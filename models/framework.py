# ======================================================================
# Unified Framework for DeepONet and MLP-IVSurface (self-contained)
# ======================================================================

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
import time
from scipy.optimize import minimize, least_squares, differential_evolution
import sys
from functorch import vmap, jacrev
import torch.nn.functional as F


# ============================================================
# Dataset Wrapper (for DeepONet per-point supervision)
# ============================================================

class IVSurfaceDataset(Dataset):
    """
    Holds per-point supervision for DeepONet:
      (branch_vector, trunk_coord) -> IV value
    """
    def __init__(self, X_branch, X_trunk, Y):
        assert len(X_branch) == len(X_trunk) == len(Y)
        self.Xb = torch.tensor(X_branch, dtype=torch.float32)
        self.Xt = torch.tensor(X_trunk, dtype=torch.float32)
        self.Y  = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.Xb[idx], self.Xt[idx], self.Y[idx]


# ============================================================
# Base Model with shared eval + persistence + scaling utilities
# ============================================================

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Stored context (set during data prep / init)
        self.strikes = None          # np.ndarray (nK,)
        self.maturities = None       # np.ndarray (nT,)
        self.input_dim = None        # int for param vector (MLP) / branch dim (DeepONet)
        self.output_shape = None     # (nT, nK) for MLP (helps reshape/check)

        # Scaling state
        self.param_bounds = None     # (lb, ub) arrays for param vector scaling to [-1, 1]
        self.output_scaler = None    # StandardScaler for IV outputs (fit on train only)

        # caches
        self._last_pred = None
        self._last_true = None
        self._last_params = None

    # -------------------- Shared helpers --------------------

    def set_grid(self, strikes, maturities):
        self.strikes = np.array(strikes, dtype=np.float32)
        self.maturities = np.array(maturities, dtype=np.float32)

    def set_io_dims(self, input_dim=None, output_shape=None):
        if input_dim is not None:
            self.input_dim = int(input_dim)
        if output_shape is not None:
            assert len(output_shape) == 2, "output_shape must be (nT, nK)"
            self.output_shape = (int(output_shape[0]), int(output_shape[1]))

    def compute_loss(self, y_pred, y_true):
        return nn.MSELoss()(y_pred, y_true)

    # -------------------- Param scaling: bounds → [-1, 1] --------------------
    @staticmethod
    def _make_param_bounds(num_knots, eta=(0.5, 4.0), rho=(0.0, 1.0), H=(0.025, 0.5), knot=(0.01, 0.16)):
        """
        Build (lb, ub) arrays for a parameter vector shaped as:
          [eta, rho, H, xi0_knots...]
        Using user-specified borders.
        """
        lb = [eta[0], rho[0], H[0]] + [knot[0]] * num_knots
        ub = [eta[1], rho[1], H[1]] + [knot[1]] * num_knots
        return np.asarray(lb, dtype=np.float32), np.asarray(ub, dtype=np.float32)

    @staticmethod
    def _scale_to_m1_p1(x, lb, ub):
        # (x - mid) * 2 / (ub - lb), where mid = (ub+lb)/2
        mid = (ub + lb) * 0.5
        return (x - mid) * (2.0 / (ub - lb))

    @staticmethod
    def _inverse_from_m1_p1(x_scaled, lb, ub):
        # x_scaled*(ub - lb)/2 + mid
        mid = (ub + lb) * 0.5
        return x_scaled * (0.5 * (ub - lb)) + mid

    def set_param_bounds(self, lb, ub):
        """Attach explicit (lb, ub) arrays to the model for param scaling."""
        lb = np.asarray(lb, dtype=np.float32)
        ub = np.asarray(ub, dtype=np.float32)
        assert lb.shape == ub.shape, "lb and ub must have the same shape"
        self.param_bounds = (lb, ub)

    def scale_params(self, param_vec):
        """Scale [eta, rho, H, xi0_knots...] to [-1, 1] using self.param_bounds."""
        assert self.param_bounds is not None, "param_bounds not set"
        lb, ub = self.param_bounds
        x = np.asarray(param_vec, dtype=np.float32)
        return BaseModel._scale_to_m1_p1(x, lb, ub)

    def inverse_scale_params(self, x_scaled):
        """Inverse of scale_params."""
        assert self.param_bounds is not None, "param_bounds not set"
        lb, ub = self.param_bounds
        x_scaled = np.asarray(x_scaled, dtype=np.float32)
        return BaseModel._inverse_from_m1_p1(x_scaled, lb, ub)

    # -------------------- Output (IV) scaling: StandardScaler --------------------
    def fit_output_scaler(self, Y_train):
        """
        Fit StandardScaler on IV outputs (train only, leakage-safe).
        Y_train: (N, nT, nK) or (N, 1) or (N, nPts, 1) for DeepONet flattened
        """
        from sklearn.preprocessing import StandardScaler
        self.output_scaler = StandardScaler()
        flat = Y_train.reshape(len(Y_train), -1)
        self.output_scaler.fit(flat)

    def transform_output(self, Y):
        """Transform IV outputs using fitted StandardScaler."""
        assert self.output_scaler is not None, "output_scaler not fitted"
        shp = Y.shape
        Y_scaled = self.output_scaler.transform(Y.reshape(len(Y), -1))
        return Y_scaled.reshape(shp)

    def inverse_transform_output_single(self, y_vec):
        """
        Inverse-transform a single vector of IV predictions.
        y_vec: shape (nPts,) or (nPts,1)
        Returns flattened numpy vector (nPts,)
        """
        assert self.output_scaler is not None, "output_scaler not fitted"
        y = np.asarray(y_vec, dtype=np.float32).reshape(1, -1)
        inv = self.output_scaler.inverse_transform(y)
        return inv.reshape(-1)

    def inverse_transform_surface(self, Y2d):
        """
        Inverse-transform a single (nT, nK) surface predicted in scaled space.
        """
        assert self.output_scaler is not None, "output_scaler not fitted"
        nT, nK = Y2d.shape
        inv = self.inverse_transform_output_single(Y2d.reshape(-1))
        return inv.reshape(nT, nK)

    # -------------------- Abstract API ----------------------
    # Each child model must implement:
    #   predict_surface(self, params_dict, store_last: bool = True) -> np.ndarray (nT, nK)

    # ====================================================
    # Shared evaluation utilities (no extra args needed)
    # ====================================================

    def plot_evaluation(self, surface_data, figsize=(15, 5), levels=30, interp_method="spline"):
        """
        Compare true vs predicted IV surfaces using contour plots.
        If the sample grid differs from the model grid, interpolate true_surface
        to match the model's grid for consistent visualization.
        """
        assert self.strikes is not None and self.maturities is not None, \
            "Model grid (strikes/maturities) not set; call set_grid or train/prepare first."

        true_surface = np.asarray(surface_data["iv_surface"], dtype=np.float32)
        grid = surface_data.get("grid", {"strikes": self.strikes, "maturities": self.maturities})
        Ks_true = np.asarray(grid["strikes"], dtype=np.float32)
        Ts_true = np.asarray(grid["maturities"], dtype=np.float32)
        params = surface_data["params"]

        # Predict surface on model's canonical grid
        strikes, maturities = self.strikes, self.maturities
        pred_surface = self.predict_surface(params)
        pred_surface = pred_surface.detach().cpu().numpy()

        # Interpolate true surface to the model's grid if needed
        if not (np.allclose(Ks_true, strikes) and np.allclose(Ts_true, maturities)):
            if interp_method == "spline":
                interp = RectBivariateSpline(Ts_true, Ks_true, true_surface)
                true_surface = interp(maturities, strikes)
            else:
                interp = RegularGridInterpolator(
                    (Ts_true, Ks_true), true_surface, method=interp_method,
                    bounds_error=False, fill_value=None
                )
                TT, KK = np.meshgrid(maturities, strikes, indexing="ij")
                coords = np.stack([TT.ravel(), KK.ravel()], axis=1)
                true_surface = interp(coords).reshape(len(maturities), len(strikes))

        # Compute difference
        diff = pred_surface - true_surface
        mae = np.mean(np.abs(diff))
        rmse = np.sqrt(np.mean(diff ** 2))
        vmax = np.max(np.abs(diff))

        # Plot
        fig, axs = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
        K, T = np.meshgrid(strikes, maturities)

        # --- True Surface ---
        c0 = axs[0].contourf(K, T, true_surface, levels=levels, cmap="viridis")
        axs[0].set_title("True Surface")
        axs[0].set_xlabel("Strike")
        axs[0].set_ylabel("Maturity")
        plt.colorbar(c0, ax=axs[0])

        # --- Predicted Surface ---
        c1 = axs[1].contourf(K, T, pred_surface, levels=levels, cmap="plasma")
        eta, rho, H = params["eta"], params["rho"], params["H"]
        axs[1].set_title(f"Predicted Surface (η={eta:.2f}, ρ={rho:.2f}, H={H:.2f})")
        axs[1].set_xlabel("Strike")
        axs[1].set_ylabel("Maturity")
        plt.colorbar(c1, ax=axs[1])

        # --- Difference ---
        c2 = axs[2].contourf(K, T, diff, levels=levels, cmap="coolwarm",
                            vmin=-vmax, vmax=vmax)
        axs[2].set_title(f"Difference (Pred - True)\nMAE={mae:.3e}, RMSE={rmse:.3e}")
        axs[2].set_xlabel("Strike")
        axs[2].set_ylabel("Maturity")
        plt.colorbar(c2, ax=axs[2])

        print(f"MAE = {mae:.6f}, RMSE = {rmse:.6f}")
        return fig
    
    def evaluate(self, surface_samples, out_dir):
        """
        Evaluates model-predicted vs. true IV surfaces by binning errors
        according to the model's base grid (self.strikes, self.maturities).

        Adds:
        - RMSE per surface
        - RMSE ECDF/Quantile plot (axes swapped)
        - RMSE heatmaps (mean/median/max)
        - RMSE vs parameter plots (eta, H, rho)
        """
        assert self.strikes is not None and self.maturities is not None, \
            "Model grid (strikes/maturities) not set; call set_grid first."
        os.makedirs(out_dir, exist_ok=True)

        nT, nK = len(self.maturities), len(self.strikes)

        # Bins
        bin_errs_rel = [[[] for _ in range(nK)] for _ in range(nT)]
        bin_errs_abs = [[[] for _ in range(nK)] for _ in range(nT)]
        bin_errs_mc  = [[[] for _ in range(nK)] for _ in range(nT)]
        bin_sqerr    = [[[] for _ in range(nK)] for _ in range(nT)]

        # Store surface-level RMSE and parameters
        rmses = []
        etas  = []
        Hs    = []
        rhos  = []

        for s in surface_samples:
            params = s["params"]
            etas.append(params["eta"])
            Hs.append(params["H"])
            rhos.append(params["rho"])

            true_surface = np.array(s["iv_surface"], dtype=np.float32)

            grid = s.get("grid", {"strikes": self.strikes, "maturities": self.maturities})
            Ks = np.asarray(grid["strikes"], dtype=np.float32)
            Ts = np.asarray(grid["maturities"], dtype=np.float32)

            pred_surface = self.predict_surface(params, grid=grid)

            # 🔥 safe conversion (outside of optimization)
            pred_surface = pred_surface.detach().cpu().numpy()

            rel_err = np.abs(true_surface - pred_surface) / np.clip(true_surface, 1e-6, None) * 100.0
            abs_err = np.abs(true_surface - pred_surface)
            sq_err  = (true_surface - pred_surface)**2
            # RMSE for this surface
            rmses.append(float(np.sqrt(np.mean(sq_err))))

            mc_rel_err = np.array(s.get("iv_rel_error", np.zeros_like(true_surface)), dtype=np.float32) * 100.0
            mc_rel_err = np.nan_to_num(mc_rel_err, nan=0.0)

            for ti, T in enumerate(Ts):
                t_idx = np.argmin(np.abs(self.maturities - T))
                for ki, K in enumerate(Ks):
                    k_idx = np.argmin(np.abs(self.strikes - K))

                    bin_errs_rel[t_idx][k_idx].append(rel_err[ti, ki])
                    bin_errs_abs[t_idx][k_idx].append(abs_err[ti, ki])
                    bin_errs_mc[t_idx][k_idx].append(mc_rel_err[ti, ki])
                    bin_sqerr[t_idx][k_idx].append(sq_err[ti, ki])

        # ---- aggregation helper ----
        def aggregate_bins(bin_errs):
            mean = np.full((nT, nK), np.nan, dtype=np.float32)
            median = np.full((nT, nK), np.nan, dtype=np.float32)
            maxv = np.full((nT, nK), np.nan, dtype=np.float32)
            for t in range(nT):
                for k in range(nK):
                    vals = bin_errs[t][k]
                    if vals:
                        mean[t, k]   = np.mean(vals)
                        median[t, k] = np.median(vals)
                        maxv[t, k]   = np.max(vals)
            global_mean = np.nanmean(mean)
            for arr in (mean, median, maxv):
                arr[np.isnan(arr)] = global_mean
            return mean, median, maxv, global_mean

        # Rel error
        mean_rel, median_rel, max_rel, global_mean_rel = aggregate_bins(bin_errs_rel)
        # Abs error
        mean_abs, median_abs, max_abs, global_mean_abs = aggregate_bins(bin_errs_abs)
        # MC error
        mean_mc, median_mc, max_mc, global_mean_mc = aggregate_bins(bin_errs_mc)

        # RMSE
        mean_sqerr, median_sqerr, max_sqerr, _ = aggregate_bins(bin_sqerr)
        mean_rmse  = np.sqrt(mean_sqerr)
        median_rmse = np.sqrt(median_sqerr)
        max_rmse    = np.sqrt(max_sqerr)
        global_mean_rmse = float(np.nanmean(mean_rmse))

        # ------------------------------------------------------------
        # plotting infra
        # ------------------------------------------------------------
        Ks_mesh, Ts_mesh = np.meshgrid(self.strikes, self.maturities, indexing="xy")

        def plot_set(data_triplet, titles, fname, label):
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for ax, data, title in zip(axes, data_triplet, titles):
                im = ax.pcolormesh(Ks_mesh, Ts_mesh, data, cmap="magma", shading="auto")
                ax.set_xlabel("Strike (K)")
                ax.set_ylabel("Maturity (T)")
                ax.set_title(title)
                ax.invert_yaxis()
                fig.colorbar(im, ax=ax, label=label)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, fname), dpi=200)
            plt.close(fig)

        # Rel error
        plot_set(
            [mean_rel, median_rel, max_rel],
            ["Mean Rel Error (Pred)", "Median Rel Error (Pred)", "Max Rel Error (Pred)"],
            "iv_error_heatmaps_binned.png",
            "%"
        )

        # Abs error
        plot_set(
            [mean_abs, median_abs, max_abs],
            ["Mean Abs Error (Pred)", "Median Abs Error (Pred)", "Max Abs Error (Pred)"],
            "iv_abs_error_heatmaps_binned.png",
            "abs(IV diff)"
        )

        # MC error
        plot_set(
            [mean_mc, median_mc, max_mc],
            ["Mean Rel Error (MC)", "Median Rel Error (MC)", "Max Rel Error (MC)"],
            "iv_mc_rel_error_heatmaps_binned.png",
            "%"
        )

        # RMSE heatmaps
        plot_set(
            [mean_rmse, median_rmse, max_rmse],
            ["Mean RMSE", "Median RMSE", "Max RMSE"],
            "iv_rmse_heatmaps_binned.png",
            "RMSE"
        )

        # ------------------------------------------------------------
        # RMSE ECDF WITH AXES SWAPPED
        # ------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(6, 5))
        sorted_r = np.sort(rmses)
        p = np.linspace(0, 1, len(sorted_r))

        # y = RMSE, x = quantile
        ax.plot(p, sorted_r, lw=2)

        ax.set_xlabel("Quantile")
        ax.set_ylabel("RMSE")
        ax.set_title("RMSE ECDF")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "iv_rmse_quantiles.png"), dpi=200)
        plt.close(fig)

        # ------------------------------------------------------------
        # RMSE VS PARAMETER SCATTERPLOTS
        # ------------------------------------------------------------
        def scatter(x, y, name_x, fname):
            fig, ax = plt.subplots(figsize=(6,5))
            ax.scatter(x, y, s=12, alpha=0.7)

            # ---- MOVING AVERAGE LINE (robust version) ----
            import numpy as np

            x = np.asarray(x)
            y = np.asarray(y)
            idx = np.argsort(x)
            xs = x[idx]
            ys = y[idx]

            window = 100
            if len(ys) > window:
                # Compute moving average
                kernel = np.ones(window) / window
                ma = np.convolve(ys, kernel, mode="valid")

                # Create xs for MA with exact matching length
                pad = (len(xs) - len(ma)) // 2
                xs_ma = xs[pad: pad + len(ma)]   # guaranteed to match ma exactly

                # Plot line
                ax.plot(xs_ma, ma, linewidth=2.5, color="red", label=f"MA({window})")

            ax.set_xlabel(name_x)
            ax.set_ylabel("RMSE")
            ax.set_title(f"RMSE vs {name_x}")
            ax.grid(True, alpha=0.3)
            ax.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, fname), dpi=200)
            plt.close(fig)


        scatter(etas, rmses, "eta", "rmse_vs_eta.png")
        scatter(Hs,   rmses, "H",   "rmse_vs_H.png")
        scatter(rhos, rmses, "rho", "rmse_vs_rho.png")

        # ------------------------------------------------------------
        # worst surfaces
        # ------------------------------------------------------------
        worst_idx = np.argsort(rmses)[-10:][::-1]

        print("\nWorst 10 surfaces by RMSE:")
        for rank, idx in enumerate(worst_idx, 1):
            print(f"{rank:2d}. index={idx}, RMSE={rmses[idx]:.6f}")

        return {
            "pred_rel": {"mean": mean_rel, "median": median_rel, "max": max_rel},
            "pred_abs": {"mean": mean_abs, "median": median_abs, "max": max_abs},
            "mc_rel":  {"mean": mean_mc, "median": median_mc, "max": max_mc},
            "rmse":    {"mean": mean_rmse, "median": median_rmse, "max": max_rmse},
            "rmses": rmses,
            "worst_indices": worst_idx.tolist(),
            "parameters": {"eta": etas, "H": Hs, "rho": rhos},
            "global_mean": {
                "pred_rel": float(global_mean_rel),
                "pred_abs": float(global_mean_abs),
                "mc_rel":   float(global_mean_mc),
                "rmse":     float(global_mean_rmse),
            }
        }




    # ============================================================
    # Calibration utilities
    # ============================================================


    def residuals_autograd(self, x_phys, true_surface, Ks, Ts):
        device = self.device

        x = torch.tensor(x_phys, dtype=torch.float32, requires_grad=True, device=device)

        def f_single(x_vec):
            params = {
                "eta": x_vec[0],
                "rho": x_vec[1],
                "H":   x_vec[2],
                "xi0_knots": x_vec[3:]
            }
            pred = self.predict_surface(params, {"strikes": Ks, "maturities": Ts})
            true_t = torch.as_tensor(true_surface, dtype=torch.float32, device=device)
            return (pred - true_t).reshape(-1)

        # jacrev returns Jacobian with efficient reuse of graph
        J_fn = jacrev(f_single)

        J = J_fn(x)  # shape (N, n_params)
        res = f_single(x)

        return res.detach().cpu().numpy(), J.detach().cpu().numpy()

    def calibrate(self, target_surface, optimiser="L-BFGS-B", bounds=None, maxiter=500, verbose=False):
        """
        Calibrate model parameters θ̂ to a given implied-volatility surface by minimizing RMSE.

        Notes
        -----
        - Works purely in physical parameter and IV space.
        - predict_surface() already returns physical implied volatilities.
        - No scaling or normalization is done anywhere in this method.
        """
        import time
        import numpy as np
        from scipy.optimize import minimize, least_squares, differential_evolution

        assert self.param_bounds is not None, "Call set_param_bounds() first."
        lb, ub = self.param_bounds if bounds is None else np.array(list(zip(*bounds)))
        lb, ub = np.asarray(lb, dtype=np.float32), np.asarray(ub, dtype=np.float32)

        true_surface = np.asarray(target_surface["iv_surface"], dtype=np.float32)
        Ks = np.asarray(target_surface["grid"]["strikes"], dtype=np.float32)
        Ts = np.asarray(target_surface["grid"]["maturities"], dtype=np.float32)
        true_params = target_surface.get("params", None)

        def rmse_objective(x_phys):
            params = {
                "eta": x_phys[0],
                "rho": x_phys[1],
                "H": x_phys[2],
                "xi0_knots": x_phys[3:]
            }
            pred = self.predict_surface(params, grid={"strikes": Ks, "maturities": Ts})
            pred = pred.detach().cpu().numpy()

            # --- Dimension check ---
            assert pred.shape == true_surface.shape, (
                f"Shape mismatch: predicted surface {pred.shape} vs true surface {true_surface.shape}"
            )
            val = np.sqrt(np.sum((pred - true_surface)**2))
            if not np.isfinite(val):
                return 1e6  # penalize NaN region
            return val

        x0 = 0.5 * (lb + ub)
        t0 = time.perf_counter()

        opt_lower = optimiser.lower()
        if opt_lower == "differential evolution":
            res = differential_evolution(rmse_objective, bounds=list(zip(lb, ub)), maxiter=maxiter, disp=verbose)
        elif opt_lower in ["levenberg-marquardt", "lm"]:
            
            def fun_ls(x):
                res_np, _ = self.residuals_autograd(x, true_surface, Ks, Ts)
                return res_np

            def jac_ls(x):
                _, J_np = self.residuals_autograd(x, true_surface, Ks, Ts)
                return J_np

            res = least_squares(
                fun_ls,
                x0,
                jac=jac_ls,
                method="trf",                # LM braucht unconstrained, aber trf akzeptiert bounds.
                bounds=(lb, ub),
                max_nfev=maxiter,
                verbose=2 if verbose else 0
            )
        else:
            res = minimize(rmse_objective, x0, method=optimiser, bounds=list(zip(lb, ub)),
                        options={"maxiter": maxiter, "disp": verbose})

        t1 = time.perf_counter()
        theta_hat = res.x

        param_names = ["eta", "rho", "H"] + [f"xi0_{i}" for i in range(len(theta_hat) - 3)]
        if true_params is not None:
            true_vec = np.concatenate([
                [true_params["eta"], true_params["rho"], true_params["H"]],
                np.array(true_params["xi0_knots"], dtype=np.float32).ravel()
            ])
            rel_errs = np.abs(theta_hat - true_vec) / np.clip(np.abs(true_vec), 1e-8, None)
            rel_err_dict = dict(zip(param_names, rel_errs))
        else:
            rel_err_dict = {k: 0.0 for k in param_names}

        rmse = rmse_objective(theta_hat)

        return {
            "theta_hat": theta_hat,
            "error_rel_dict": rel_err_dict,
            "rmse": float(rmse),
            "runtime_ms": (t1 - t0) * 1000,
            "optimizer": optimiser
        }

    def evaluate_calibrate(self, surfaces, optimiser="L-BFGS-B", maxiter=500,
                        out_dir="calibration_eval", verbose=False):
        """
        Run calibration across multiple surfaces using a single optimizer,
        producing per-parameter error statistics (mean/median/std + RMSE for absolute),
        and returning true/estimated parameter arrays.
        """

        os.makedirs(out_dir, exist_ok=True)
        print(f"\nEvaluating calibration using {optimiser} on {len(surfaces)} surfaces...")

        runtimes, rmses = [], []
        per_param_rel_errors, per_param_abs_errors = {}, {}
        true_params_all, est_params_all = [], []

        for i, s in enumerate(surfaces, start=1):
            r = self.calibrate(s, optimiser=optimiser, maxiter=maxiter, verbose=verbose)
            runtimes.append(r["runtime_ms"])
            rmses.append(r["rmse"])
            est_params_all.append(r["theta_hat"])

            tp = s.get("params", None)
            if tp is not None:
                tvec = np.concatenate([
                    [tp["eta"], tp["rho"], tp["H"]],
                    np.array(tp["xi0_knots"], dtype=np.float32).ravel()
                ])
                true_params_all.append(tvec)
            else:
                true_params_all.append(np.full_like(r["theta_hat"], np.nan))

            theta_hat = np.array(r["theta_hat"], dtype=np.float32)
            true_vec = np.array(true_params_all[-1], dtype=np.float32)
            param_names = ["eta", "rho", "H"] + [f"xi0_{j}" for j in range(len(theta_hat) - 3)]

            abs_errs = np.abs(theta_hat - true_vec)
            rel_errs = abs_errs / np.clip(np.abs(true_vec), 1e-8, None)

            for k, aerr, rerr in zip(param_names, abs_errs, rel_errs):
                per_param_abs_errors.setdefault(k, []).append(aerr)
                per_param_rel_errors.setdefault(k, []).append(rerr)

            # if verbose and (i % 50 == 0 or i == len(surfaces)):
            #     mean_rmse = np.mean(rmses)
            #     print(f"  [{i}/{len(surfaces)}] mean RMSE={mean_rmse:.5f}, "
            #         f"avg time={np.mean(runtimes):.1f} ms")

        # Convert lists to arrays
        for d in (per_param_abs_errors, per_param_rel_errors):
            for k in d:
                d[k] = np.array(d[k])

        runtimes, rmses = np.array(runtimes), np.array(rmses)
        true_params_all, est_params_all = np.array(true_params_all), np.array(est_params_all)
        avg_time = np.mean(runtimes)

        # --- Helper for summary printing ---
        def summarize(errors, kind):
            print(f"{kind.title()} Errors per Parameter:")
            for k, vals in errors.items():
                scale = 100 if kind == "relative" else 1
                mean = np.mean(vals)
                median = np.median(vals)
                std = np.std(vals)
                q95 = np.quantile(vals, 0.95)
                q99 = np.quantile(vals, 0.99)
                unit = "%" if kind == "relative" else ""

                if kind == "absolute":
                    rmse = np.sqrt(np.mean(vals ** 2))
                    print(f"   {k:<8s} | mean={mean*scale:.3f}{unit}"
                        f" | median={median*scale:.3f}{unit}"
                        f" | std={std*scale:.3f}{unit}"
                        f" | q95={q95*scale:.3f}{unit}"
                        f" | q99={q99*scale:.3f}{unit}"
                        f" | RMSE={rmse*scale:.3f}{unit}")
                else:
                    print(f"   {k:<8s} | mean={mean*scale:.3f}{unit}"
                        f" | median={median*scale:.3f}{unit}"
                        f" | std={std*scale:.3f}{unit}"
                        f" | q95={q95*scale:.3f}{unit}"
                        f" | q99={q99*scale:.3f}{unit}")

        # --- Output summary ---
        print(f"\n→ Final avg time: {avg_time:.1f} ms, mean RMSE={np.mean(rmses):.5f}\n")
        summarize(per_param_rel_errors, "relative")
        print()
        summarize(per_param_abs_errors, "absolute")

        # --- Return data (only abs RMSEs computed) ---
        per_param_abs_rmse = {k: np.sqrt(np.mean(v ** 2)) for k, v in per_param_abs_errors.items()}

        # ------------------------------------------------------------
        # Print top-5 highest absolute + relative errors per parameter
        # ------------------------------------------------------------
        print("\nTop-5 absolute & relative errors per parameter:")
        for k in per_param_abs_errors.keys():
            abs_vals = per_param_abs_errors[k]      # shape (n_surfaces,)
            rel_vals = per_param_rel_errors[k]

            # --- Top-5 absolute ---
            top5_abs_idx = np.argsort(abs_vals)[-5:][::-1]
            top5_abs_vals = abs_vals[top5_abs_idx]

            # --- Top-5 relative ---
            top5_rel_idx = np.argsort(rel_vals)[-5:][::-1]
            top5_rel_vals = rel_vals[top5_rel_idx]

            print(f"\nParameter: {k}")
            print("  Top-5 ABS errors:")
            for rank, (idx, val) in enumerate(zip(top5_abs_idx, top5_abs_vals), 1):
                print(
                    f"    {rank}. index={int(idx):4d} | abs_err={float(val):.6f} "
                    f"| surface_RMSE={float(rmses[idx]):.6f}"
                )

            print("  Top-5 REL errors:")
            for rank, (idx, val) in enumerate(zip(top5_rel_idx, top5_rel_vals), 1):
                print(
                    f"    {rank}. index={int(idx):4d} | rel_err={float(val):.6f} "
                    f"| surface_RMSE={float(rmses[idx]):.6f}"
                )


        return {
            "optimizer": optimiser,
            "avg_time_ms": float(avg_time),
            "mean_rmse": float(np.mean(rmses)),
            "per_param_rel_errors": per_param_rel_errors,
            "per_param_abs_errors": per_param_abs_errors,
            "per_param_abs_rmse": per_param_abs_rmse,
            "true_params": true_params_all,
            "est_params": est_params_all,
            "rmses": rmses,
            "runtimes": runtimes,
        }

    def count_parameters(self, trainable_only=True):
        """Return the number of (trainable) parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

# ============================================================
# DeepONet (self-contained)
# ============================================================

# (assuming BaseModel and IVSurfaceDataset are imported or defined elsewhere)


# assuming BaseModel and IVSurfaceDataset are available
class SparseMask(nn.Module):
    """Wrapper für mask_net mit integrierter L1- und Entropy-Sparsity."""
    def __init__(self, mask_net, entropy_lambda=1e-3, l1_lambda=1e-4):
        super().__init__()
        self.mask_net = mask_net
        self.entropy_lambda = entropy_lambda
        self.l1_lambda = l1_lambda
        self.loss_reg = torch.tensor(0.0)

    def forward(self, x):
        m = torch.sigmoid(self.mask_net(x))
        if self.training:
            # Regularisierung nur im Trainingsmodus berechnen
            ent = - (m * torch.log(m + 1e-8) + (1 - m) * torch.log(1 - m + 1e-8))
            self.loss_reg = (
                self.l1_lambda * m.abs().mean() +
                self.entropy_lambda * ent.mean()
            )
        else:
            self.loss_reg = torch.tensor(0.0, device=m.device)
        return m



class DeepONet(BaseModel):
    """
    Deep Operator Network for implied volatility surfaces.
    - Branch input: parameter vector θ (eta, rho, H, xi0_knots...)
    - Trunk input: 2D coords (K, T)
    - Optional learnable mask acting as contextual filter between branch and trunk
      (mask_type: 'none', 'spatial', 'channel', or 'contextual').
    """

    def __init__(self,
                 branch_in_dim=None,
                 trunk_in_dim=2,
                 latent_dim=64,
                 branch_hidden_dims=(64, 64),
                 trunk_hidden_dims=(64, 64),
                 activation="relu",
                 lr=1e-3,
                 mask_type="none"):
        super().__init__()
        self.branch_in_dim = branch_in_dim
        self.trunk_in_dim = trunk_in_dim
        self.latent_dim = latent_dim
        self.branch_hidden_dims = branch_hidden_dims
        self.trunk_hidden_dims = trunk_hidden_dims
        self.activation = activation
        self.lr = lr
        self.mask_type = mask_type.lower()

        if branch_in_dim:
            self._build_networks()
        self.to(self.device)

    # --------------------------------------------------------
    def _build_networks(self):
        act_fn = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU()
        }[self.activation]

        # --- Branch network ---
        branch_layers = []
        dims = [self.branch_in_dim] + list(self.branch_hidden_dims)
        for i in range(len(dims) - 1):
            branch_layers.append(nn.Linear(dims[i], dims[i + 1]))
            branch_layers.append(act_fn)
        branch_layers.append(nn.Linear(dims[-1], self.latent_dim))
        self.branch = nn.Sequential(*branch_layers)

        # --- Trunk network ---
        trunk_layers = []
        dims_t = [self.trunk_in_dim] + list(self.trunk_hidden_dims)
        for i in range(len(dims_t) - 1):
            trunk_layers.append(nn.Linear(dims_t[i], dims_t[i + 1]))
            trunk_layers.append(act_fn)
        trunk_layers.append(nn.Linear(dims_t[-1], self.latent_dim))
        self.trunk = nn.Sequential(*trunk_layers)

        # --- Learnable mask ---
        self.mask_net = self._build_mask(act_fn)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    # --------------------------------------------------------
    def _build_mask(self, act_fn):
        """Construct a learnable mask network depending on mask_type."""
        if self.mask_type == "none":
            return None

        elif self.mask_type == "spatial":
            base = nn.Sequential(
                nn.Linear(self.trunk_in_dim, 32),
                act_fn,
                nn.Linear(32, 16),
                act_fn,
                nn.Linear(16, 1)
            )
            return SparseMask(base)

        elif self.mask_type == "channel":
            base = nn.Sequential(
                nn.Linear(self.trunk_in_dim, self.latent_dim)
            )
            return SparseMask(base)

        elif self.mask_type == "contextual":
            base = nn.Sequential(
                nn.Linear(self.latent_dim * 2, self.latent_dim ),
                act_fn,
                nn.Linear(self.latent_dim, self.latent_dim)
            )
            return SparseMask(base)

        else:
            raise ValueError(f"Unknown mask_type '{self.mask_type}'")


    # --------------------------------------------------------
    def forward(self, xb, xt):
        b = self.branch(xb)  # (batch, latent_dim)
        t = self.trunk(xt)   # (batch, latent_dim)

        if self.mask_type == "none":
            return torch.sum(b * t, dim=1, keepdim=True)

        elif self.mask_type == "spatial":
            mask = self.mask_net(xt)  # (batch, 1)
            return torch.sum(b * t, dim=1, keepdim=True) * mask

        elif self.mask_type == "channel":
            mask = self.mask_net(xt)  # (batch, latent_dim)
            return torch.sum(b * t * mask, dim=1, keepdim=True)

        elif self.mask_type == "contextual":
            mask_input = torch.cat([b, t], dim=1)
            mask = self.mask_net(mask_input)  # (batch, latent_dim)
            return torch.sum(b * t * mask, dim=1, keepdim=True)

    # --------------------------------------------------------

    @staticmethod
    def _flatten_surfaces_for_deeponet(surfaces, enforce_shared_grid=False):
        """
        Convert list of surfaces into per-point tuples for DeepONet training.

        Each element in `surfaces` is expected to have:
            - surf["params"]: dict with eta, rho, H, xi0_knots
            - surf["iv_surface"]: 2D array (nT, nK)
            - surf["grid"]["strikes"], surf["grid"]["maturities"]

        Unlike the earlier version, this version supports variable grids per surface.
        """
        Xb_list, Xt_list, Y_list = [], [], []
        Ks_ref, Ts_ref = None, None

        for surf in surfaces:
            params = surf["params"]
            iv_surface = np.asarray(surf["iv_surface"], dtype=np.float32)
            Ks = np.asarray(surf["grid"]["strikes"], dtype=np.float32)
            Ts = np.asarray(surf["grid"]["maturities"], dtype=np.float32)

            # Optional safety check if you want to enforce shared grid (for plotting consistency)
            if enforce_shared_grid:
                if Ks_ref is None:
                    Ks_ref, Ts_ref = Ks, Ts
                else:
                    if not (np.allclose(Ks_ref, Ks) and np.allclose(Ts_ref, Ts)):
                        raise ValueError("All surfaces must share the same (K, T) grid when enforce_shared_grid=True")

            xi0_knots = np.array(params["xi0_knots"], dtype=np.float32).flatten()
            branch_vec = np.concatenate([[params["eta"], params["rho"], params["H"]], xi0_knots])

            # Build per-point training tuples
            K_mesh, T_mesh = np.meshgrid(Ks, Ts)
            trunk_coords = np.stack([K_mesh.ravel(), T_mesh.ravel()], axis=1).astype(np.float32)
            branch_repeated = np.repeat(branch_vec[None, :], len(trunk_coords), axis=0)
            Y_flat = iv_surface.ravel()[:, None].astype(np.float32)

            Xb_list.append(branch_repeated)
            Xt_list.append(trunk_coords)
            Y_list.append(Y_flat)

        X_branch = np.concatenate(Xb_list, axis=0)
        X_trunk = np.concatenate(Xt_list, axis=0)
        Y = np.concatenate(Y_list, axis=0)

        # Only return the last surface's grid for convenience (used for plotting)
        strikes = Ks_ref if Ks_ref is not None else Ks
        maturities = Ts_ref if Ts_ref is not None else Ts

        return X_branch, X_trunk, Y, strikes, maturities


    @classmethod
    def from_surfaces(cls,
                      surfaces,
                      *,
                      mask_type="none",
                      batch_size=256,
                      val_split=0.2,
                      shuffle=True,
                      shuffle_training_batches=False,
                      branch_hidden_dims=(64, 64),
                      trunk_hidden_dims=(64, 64),
                      activation="relu",
                      lr=1e-3,
                      latent_dim=64,
                      ref_strikes=None,
                      ref_maturities=None):
        """
        Build a DeepONet model + loaders with internal, leakage-safe scaling.
        Includes learnable mask specified by `mask_type`.
        """
        X_branch, X_trunk, Y, strikes, maturities = cls._flatten_surfaces_for_deeponet(surfaces)
        if ref_strikes is not None:
            strikes = np.asarray(ref_strikes, dtype=np.float32)
        if ref_maturities is not None:
            maturities = np.asarray(ref_maturities, dtype=np.float32)

        # Empirical bounds for branch normalization
        lb = np.min(X_branch, axis=0)
        ub = np.max(X_branch, axis=0)
        margin = 0.01 * (ub - lb)
        lb -= margin
        ub += margin
        Xb_scaled = BaseModel._scale_to_m1_p1(X_branch, lb, ub)

        n_total = len(Y)
        n_train = int((1 - val_split) * n_total)
        idx = np.arange(n_total)
        if shuffle:
            np.random.shuffle(idx)
        tr, va = idx[:n_train], idx[n_train:]

        model = cls(branch_in_dim=Xb_scaled.shape[1],
                    branch_hidden_dims=branch_hidden_dims,
                    trunk_hidden_dims=trunk_hidden_dims,
                    activation=activation,
                    latent_dim=latent_dim,
                    lr=lr,
                    mask_type=mask_type)

        model.set_grid(strikes, maturities)
        model.set_io_dims(input_dim=Xb_scaled.shape[1])
        model.fit_output_scaler(Y[tr])
        model.set_param_bounds(lb, ub)

        Y_tr_scaled = model.transform_output(Y[tr])
        Y_va_scaled = model.transform_output(Y[va])

        train_ds = IVSurfaceDataset(Xb_scaled[tr], X_trunk[tr], Y_tr_scaled)
        val_ds = IVSurfaceDataset(Xb_scaled[va], X_trunk[va], Y_va_scaled)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_training_batches)
        val_loader = DataLoader(val_ds, batch_size=2 * batch_size, shuffle=False)

        return model, train_loader, val_loader, strikes, maturities

    def test_mask_response(self, xb_sample, xt_grid, visualize=True, channels=None):
        """
        Evaluate and optionally visualize individual latent mask channels
        for a fixed branch vector across a grid of (K,T) coordinates.

        Parameters
        ----------
        xb_sample : array-like
            Parameter vector (eta, rho, H, xi0_knots...).
        xt_grid : np.ndarray of shape [N, 2]
            Grid of (K, T) coordinates.
        visualize : bool, default=True
            If True, plots selected mask channels as 2D heatmaps.
        channels : list[int], optional
            Specific channel indices to visualize (e.g. [0, 3, 7]).
            If None, visualizes all latent channels.

        Returns
        -------
        mask_np : np.ndarray
            Full mask array of shape [N, latent_dim].
        """
        import torch, numpy as np, matplotlib.pyplot as plt

        self.eval()
        if getattr(self, "mask_net", None) is None:
            print("No mask network defined (mask_type='none').")
            return None

        # Prepare input tensors
        xb_sample = torch.tensor(xb_sample, dtype=torch.float32, device=self.device)
        xb_sample = xb_sample.unsqueeze(0).repeat(len(xt_grid), 1)
        xt = torch.tensor(xt_grid, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            b = self.branch(xb_sample)
            t = self.trunk(xt)

            if self.mask_type == "spatial":
                mask = self.mask_net(xt)
            elif self.mask_type == "channel":
                mask = self.mask_net(xt)
            elif self.mask_type == "contextual":
                mask_input = torch.cat([b, t], dim=1)
                mask = self.mask_net(mask_input)
            else:
                mask = torch.ones(len(xt), self.latent_dim, device=self.device)

        mask_np = mask.detach().cpu().numpy()  # [N, latent_dim]
        nK = len(np.unique(xt_grid[:, 0]))
        nT = len(np.unique(xt_grid[:, 1]))
        latent_dim = mask_np.shape[1]

        # ----------------------------------------------------------
        # Visualization
        # ----------------------------------------------------------
        if visualize:
            if channels is None:
                channels = list(range(latent_dim))
            else:
                channels = [c for c in channels if c < latent_dim]

            n_show = len(channels)
            fig, axes = plt.subplots(1, n_show, figsize=(3.5 * n_show, 4))

            for idx, c in enumerate(channels):
                ax = axes[idx] if n_show > 1 else axes
                mask_c = mask_np[:, c].reshape(nT, nK)
                im = ax.imshow(mask_c, origin="lower", aspect="auto", cmap="magma")
                ax.set_title(f"Channel {c}")
                ax.set_xlabel("Strike (K)")
                ax.set_ylabel("Maturity (T)")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            plt.suptitle(f"Mask Channels ({self.mask_type})", fontsize=13)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

        return mask_np

    # --------------------------------------------------------
    def train_model(
        self,
        train_loader,
        val_loader=None,
        epochs=10,
        lr_schedule=[(0, 1e-3), (5, 5e-4), (8, 1e-4)],
    ):
        """
        Train DeepONet in scaled output space (StandardScaler),
        but additionally compute & report RMSE in ORIGINAL IV space.
        """

        lr_schedule = sorted(lr_schedule, key=lambda x: x[0])
        schedule_index = 0
        base_lr = lr_schedule[0][1]
        for g in self.optimizer.param_groups:
            g["lr"] = base_lr

        start_time = time.time()
        epoch_durations = []

        for epoch in range(epochs):
            epoch_start = time.time()

            # --- update LR ---
            if (
                schedule_index + 1 < len(lr_schedule)
                and epoch >= lr_schedule[schedule_index + 1][0]
            ):
                schedule_index += 1
                new_lr = lr_schedule[schedule_index][1]
                for g in self.optimizer.param_groups:
                    g["lr"] = new_lr
                sys.stdout.write(f"\n→ Adjusted learning rate to {new_lr:.2e} at epoch {epoch}\n")

            # --- TRAINING ---
            self.train()
            total_scaled_loss = 0.0
            total_iv_rmse = 0.0
            n_samples = len(train_loader.dataset)
            for xb, xt, y in train_loader:
                xb, xt, y = xb.to(self.device), xt.to(self.device), y.to(self.device)

                pred = self.forward(xb, xt)
                loss = self.compute_loss(pred, y)
                if hasattr(self.mask_net, "loss_reg"):
                    loss += self.mask_net.loss_reg

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # accumulate scaled loss
                batch_size = len(y)
                total_scaled_loss += loss.item() * batch_size

                # ---- compute IV RMSE ----
                pred_np = pred.detach().cpu().numpy().reshape(batch_size, -1)
                y_np    = y.detach().cpu().numpy().reshape(batch_size, -1)

                pred_iv = self.inverse_transform_output_single(pred_np)
                y_iv    = self.inverse_transform_output_single(y_np)

                total_iv_rmse += np.sum((pred_iv - y_iv)**2)

            # per-epoch metrics
            train_rmse_scaled = float(np.sqrt(total_scaled_loss / n_samples))
            train_rmse_iv     = float(np.sqrt(total_iv_rmse / n_samples))

            # --- VALIDATION ---
            if val_loader is not None:
                val_rmse_scaled, val_rmse_iv = self._validate_dual(val_loader)
                msg = (f"Epoch {epoch+1:03d} | "
                    f"train_scaled={train_rmse_scaled:.6f}, val_scaled={val_rmse_scaled:.6f}, "
                    f"train_iv={train_rmse_iv:.6f}, val_iv={val_rmse_iv:.6f}, "
                    f"lr={self.optimizer.param_groups[0]['lr']:.1e}")
            else:
                msg = (f"Epoch {epoch+1:03d} | "
                    f"train_scaled={train_rmse_scaled:.6f}, train_iv={train_rmse_iv:.6f}, "
                    f"lr={self.optimizer.param_groups[0]['lr']:.1e}")

            # ETA & timing
            epoch_time = time.time() - epoch_start
            epoch_durations.append(epoch_time)
            avg_time = np.mean(epoch_durations)
            eta = (epochs - (epoch + 1)) * avg_time
            msg += f", time={epoch_time:.2f}s, ETA={eta/60:.1f} min"

            sys.stdout.write("\r\033[K" + msg)
            sys.stdout.flush()

        print()
        total_time = time.time() - start_time
        print(f"\n✅ Training completed in {total_time/60:.2f} min")



    def _validate_dual(self, val_loader):
        """
        Return (rmse_scaled, rmse_iv) for validation.
        """
        self.eval()
        total_scaled = 0.0
        total_iv = 0.0
        n = len(val_loader.dataset)

        with torch.no_grad():
            for xb, xt, y in val_loader:
                xb, xt, y = xb.to(self.device), xt.to(self.device), y.to(self.device)
                pred = self.forward(xb, xt)

                # scaled MSE
                scaled = self.compute_loss(pred, y).item()
                batch_size = len(y)
                total_scaled += scaled * batch_size

                # IV RMSE
                pred_np = pred.detach().cpu().numpy().reshape(batch_size, -1)
                y_np    = y.detach().cpu().numpy().reshape(batch_size, -1)
                pred_iv = self.inverse_transform_output_single(pred_np)
                y_iv    = self.inverse_transform_output_single(y_np)
                total_iv +=(np.sum((pred_iv - y_iv)**2))

        return (
            float(np.sqrt(total_scaled / n)),
            float(np.sqrt(total_iv/n))
        )

    def validate(self, val_loader):
        self.eval(); total = 0.0
        with torch.no_grad():
            for xb, xt, y in val_loader:
                xb, xt, y = xb.to(self.device), xt.to(self.device), y.to(self.device)
                pred = self.forward(xb, xt)
                total += self.compute_loss(pred, y).item() * len(y)
        return total / len(val_loader.dataset)

    # --------------------------------------------------------
    ###############################################
    #   Fully Torch-Based predict_surface()
    #   Drop-in replacement for DeepONet models
    ###############################################

    def predict_surface(self, params, grid=None):
        device = self.device

        # ---------------------------------------------------------
        # 1) Choose grid
        # ---------------------------------------------------------
        if grid is None:
            strikes = torch.tensor(self.strikes, dtype=torch.float32, device=device)
            maturities = torch.tensor(self.maturities, dtype=torch.float32, device=device)
        else:
            strikes = torch.tensor(grid["strikes"], dtype=torch.float32, device=device)
            maturities = torch.tensor(grid["maturities"], dtype=torch.float32, device=device)

        # ---------------------------------------------------------
        # 2) Construct unscaled parameter vector (torch)
        # ---------------------------------------------------------
        eta = torch.as_tensor(params["eta"], dtype=torch.float32, device=device)
        rho = torch.as_tensor(params["rho"], dtype=torch.float32, device=device)
        H   = torch.as_tensor(params["H"],   dtype=torch.float32, device=device)
        xi0 = torch.as_tensor(params["xi0_knots"], dtype=torch.float32, device=device).flatten()

        x_phys = torch.cat([eta.unsqueeze(0), rho.unsqueeze(0), H.unsqueeze(0), xi0], dim=0)  # (d_branch,)

        # ---------------------------------------------------------
        # 3) Scale parameters exactly like in training (AUTOGRAD SAFE)
        # ---------------------------------------------------------
        lb = torch.as_tensor(self.param_bounds[0], dtype=torch.float32, device=device)
        ub = torch.as_tensor(self.param_bounds[1], dtype=torch.float32, device=device)

        # scale to [-1,1]
        x_scaled = 2.0 * (x_phys - lb) / (ub - lb) - 1.0     # (d_branch,)

        # repeat for each trunk point
        # will be repeated after trunk construction

        # ---------------------------------------------------------
        # 4) Construct trunk coordinates
        # ---------------------------------------------------------
        K_mesh, T_mesh = torch.meshgrid(strikes, maturities, indexing="xy")
        xt = torch.stack([K_mesh.reshape(-1), T_mesh.reshape(-1)], dim=1)  # (nPts, 2)

        # ---------------------------------------------------------
        # 5) Broadcast branch vector
        # ---------------------------------------------------------
        xb = x_scaled.unsqueeze(0).repeat(xt.shape[0], 1)   # (nPts, d_branch)

        # ---------------------------------------------------------
        # 6) Forward pass (scaled output)
        # ---------------------------------------------------------
        pred_scaled = self.forward(xb, xt).reshape(-1)

        # ---------------------------------------------------------
        # 7) Inverse output-scaling in TORCH
        # ---------------------------------------------------------
        mean = torch.as_tensor(self.output_scaler.mean_[0], dtype=torch.float32, device=device)
        std  = torch.as_tensor(self.output_scaler.scale_[0], dtype=torch.float32, device=device)

        pred_iv = pred_scaled * std + mean   # physical IV

        # ---------------------------------------------------------
        # 8) Reshape to surface
        # ---------------------------------------------------------
        nK = strikes.numel()
        nT = maturities.numel()
        return pred_iv.reshape(nT, nK)





# ============================================================
# MLP (self-contained)
# ============================================================

class MLP(BaseModel):
    """
    Simple MLP mapping parameter vector -> implied volatility surface.
    Output is the full (nT, nK) surface predicted in one shot.
    """
    def __init__(self, input_dim=None, output_shape=None, hidden_dims=(256, 256, 256),
                 activation="elu", lr=1e-3):
        super().__init__()
        self.input_dim = input_dim
        self.output_shape = output_shape  # (nT, nK)
        self.output_dim = None if output_shape is None else int(output_shape[0] * output_shape[1])
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.lr = lr

        if (input_dim is not None) and (output_shape is not None):
            self._build_network()
        self.to(self.device)

    def _build_network(self):
        act_fn = {"relu": nn.ReLU(), "gelu": nn.GELU(), "tanh": nn.Tanh(), "elu": nn.ELU()}[self.activation]
        dims = [self.input_dim] + list(self.hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act_fn)
        layers.append(nn.Linear(dims[-1], self.output_dim))
        self.net = nn.Sequential(*layers).to(self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        out = self.net(x)                    # (batch, 121)
        batch = out.shape[0]
        return out.reshape(batch, *self.output_shape)

    # --------------------------------------------------------
    @staticmethod
    def _stack_XY(surfaces, sanity_check_grids=True):
        X_list, Y_list = [], []
        Ks_ref, Ts_ref = None, None

        for surf in surfaces:
            params = surf["params"]
            iv_surface = np.array(surf["iv_surface"], dtype=np.float32)
            Ks = np.array(surf["grid"]["strikes"], dtype=np.float32)
            Ts = np.array(surf["grid"]["maturities"], dtype=np.float32)

            if sanity_check_grids:
                if Ks_ref is None:
                    Ks_ref, Ts_ref = Ks, Ts
                else:
                    if not (np.allclose(Ks_ref, Ks) and np.allclose(Ts_ref, Ts)):
                        raise ValueError("All surfaces must share the same (K, T) grid for the MLP output shape.")

            xi0_knots = np.array(params["xi0_knots"]).flatten()
            param_vec = np.concatenate([[params["eta"], params["rho"], params["H"]], xi0_knots]).astype(np.float32)
            X_list.append(param_vec)
            Y_list.append(iv_surface)

        X = np.stack(X_list, axis=0)   # (N, d)
        Y = np.stack(Y_list, axis=0)   # (N, nT, nK)
        strikes = Ks_ref if Ks_ref is not None else np.array(surfaces[0]["grid"]["strikes"], dtype=np.float32)
        maturities = Ts_ref if Ts_ref is not None else np.array(surfaces[0]["grid"]["maturities"], dtype=np.float32)
        return X, Y, strikes, maturities

    @classmethod
    def from_surfaces(cls, surfaces, batch_size=32, val_split=0.2, shuffle=True,
                    hidden_dims=(256, 256, 256), activation="gelu", lr=1e-3):
        """
        Build an MLP model + loaders with internal, leakage-safe scaling:
        - Param vector scaled to [-1, 1] using empirical min/max (+5% margin)
        - Output (IV) scaled via StandardScaler fit on train only

        Returns
        -------
        model, train_loader, val_loader, strikes, maturities
        """
        # --- Flatten all surfaces ---
        X, Y, strikes, maturities = cls._stack_XY(surfaces)
        nT, nK = Y.shape[1:]
        input_dim = X.shape[1]
        output_shape = (nT, nK)

        # --- Empirical parameter bounds (+5% margin) ---
        lb = np.min(X, axis=0)
        ub = np.max(X, axis=0)
        margin = 0.01 * (ub - lb)
        lb -= margin
        ub += margin

        X_scaled = BaseModel._scale_to_m1_p1(X, lb, ub)

        # Split
        n_total = len(Y)
        n_train = int((1 - val_split) * n_total)
        idx = np.arange(n_total)
        if shuffle:
            np.random.shuffle(idx)
        tr, va = idx[:n_train], idx[n_train:]

        # Fit output scaler on train only
        model = cls(input_dim=input_dim, output_shape=output_shape, hidden_dims=hidden_dims,
                    activation=activation, lr=lr)
        model.set_grid(strikes, maturities)
        model.set_io_dims(input_dim=input_dim, output_shape=output_shape)
        model.set_param_bounds(lb, ub)
        model.fit_output_scaler(Y[tr])

        # Transform outputs
        Y_tr_scaled = model.transform_output(Y[tr])
        Y_va_scaled = model.transform_output(Y[va])

        # Tensors
        Xtr = torch.tensor(X_scaled[tr], dtype=torch.float32)
        Ytr = torch.tensor(Y_tr_scaled, dtype=torch.float32)
        Xva = torch.tensor(X_scaled[va], dtype=torch.float32)
        Yva = torch.tensor(Y_va_scaled, dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(TensorDataset(Xva, Yva), batch_size=2 * batch_size, shuffle=False)

        # Ensure network is built (already in __init__)
        return model, train_loader, val_loader, strikes, maturities

    # --------------------------------------------------------
    def train_model(
        self,
        train_loader,
        val_loader=None,
        epochs=50,
        lr_schedule=[(0, 1e-3), (30, 5e-4), (60, 1e-4)],
    ):
        """
        Train MLP in scaled output space, but report both
        scaled RMSE and true IV-space RMSE.
        """
        lr_schedule = sorted(lr_schedule, key=lambda x: x[0])
        schedule_index = 0

        for g in self.optimizer.param_groups:
            g["lr"] = lr_schedule[0][1]

        start_time = time.time()
        epoch_durations = []

        for epoch in range(epochs):
            epoch_start = time.time()

            # --- LR update ---
            if (
                schedule_index + 1 < len(lr_schedule)
                and epoch >= lr_schedule[schedule_index + 1][0]
            ):
                schedule_index += 1
                new_lr = lr_schedule[schedule_index][1]
                for g in self.optimizer.param_groups:
                    g["lr"] = new_lr
                sys.stdout.write(f"\n→ Adjusted learning rate to {new_lr:.2e} at epoch {epoch}\n")

            # --- TRAIN ---
            self.train()
            total_scaled = 0.0
            total_iv = 0.0
            n = len(train_loader.dataset)

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                pred = self.forward(x)
                loss = self.criterion(pred, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch = len(y)
                total_scaled += loss.item() * batch
                
                # IV RMSE
                pred_np = pred.detach().cpu().numpy()     # (batch, 11,11)
                y_np    = y.detach().cpu().numpy()        # (batch, 11,11)

                batch = pred_np.shape[0]

                rmse_iv_list = []
                for i in range(batch):
                    # einzelne Oberfläche extrahieren
                    pred_surf = pred_np[i]      # (11,11)
                    y_surf    = y_np[i]         # (11,11)

                    # inverse transform auf EIN surface
                    pred_iv  = self.inverse_transform_surface(pred_surf)
                    y_iv     = self.inverse_transform_surface(y_surf)

                    rmse_iv_list.append( np.sqrt(np.mean((pred_iv - y_iv)**2)) )
                rmse_iv = float(np.mean(rmse_iv_list))
                total_iv += rmse_iv * batch

            train_scaled = float(np.sqrt(total_scaled / n))
            train_iv     = float(total_iv / n)

            # --- VAL ---
            if val_loader is not None:
                val_scaled, val_iv = self._validate_dual(val_loader)
                msg = (f"Epoch {epoch+1:03d} | "
                    f"train_scaled={train_scaled:.6f}, val_scaled={val_scaled:.6f}, "
                    f"train_iv={train_iv:.6f}, val_iv={val_iv:.6f}, "
                    f"lr={self.optimizer.param_groups[0]['lr']:.1e}")
            else:
                msg = (f"Epoch {epoch+1:03d} | "
                    f"train_scaled={train_scaled:.6f}, train_iv={train_iv:.6f}, "
                    f"lr={self.optimizer.param_groups[0]['lr']:.1e}")

            epoch_time = time.time() - epoch_start
            epoch_durations.append(epoch_time)
            eta = (epochs - (epoch + 1)) * np.mean(epoch_durations)
            msg += f", time={epoch_time:.2f}s, ETA={eta/60:.1f} min"

            sys.stdout.write("\r\033[K" + msg)
            sys.stdout.flush()

        print()
        total_time = time.time() - start_time
        print(f"\n✅ Training completed in {total_time/60:.2f} min")


    def _validate_dual(self, val_loader):
        """
        Compute BOTH:
        - RMSE in scaled output space
        - RMSE in ORIGINAL IV space
        """
        self.eval()
        total_scaled = 0.0
        total_iv = 0.0
        n = len(val_loader.dataset)

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.forward(x)

                # --- scaled RMSE accumulation ---
                scaled_loss = self.criterion(pred, y).item()
                batch = y.shape[0]
                total_scaled += scaled_loss * batch

                # -----------------------------------------
                # TRUE IV RMSE: EINE SURFACE NACH DER ANDEREN
                # -----------------------------------------
                pred_np = pred.detach().cpu().numpy()   # (batch, nT, nK)
                y_np    = y.detach().cpu().numpy()      # (batch, nT, nK)

                rmse_list = []
                for i in range(batch):
                    pred_surf = pred_np[i]        # (nT, nK)
                    y_surf    = y_np[i]           # (nT, nK)

                    pred_iv = self.inverse_transform_surface(pred_surf)
                    y_iv    = self.inverse_transform_surface(y_surf)

                    rmse_list.append(np.sqrt(np.mean((pred_iv - y_iv)**2)))

                rmse_iv = float(np.mean(rmse_list))
                total_iv += rmse_iv * batch


        return (
            float(np.sqrt(total_scaled / n)),   # scaled RMSE
            float(total_iv / n)                 # IV-RMSE
        )


    def validate(self, val_loader):
        self.eval(); total = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                total += self.criterion(self.forward(x), y).item() * len(y)
        return total / len(val_loader.dataset)   

    # --------------------------------------------------------###########################################################
    #   Fully Torch-Based predict_surface() for MLP models
    #   Drop-in replacement: autograd-friendly, numpy-free
    ###########################################################


    def predict_surface(self, params, grid=None):
        """
        Fully torch-based surface prediction for MLP models.
        Autograd-safe and GPU-accelerated.
        """

        device = self.device

        # -------------------------------------------
        # 1) Convert parameters to Torch
        # -------------------------------------------
        eta = torch.as_tensor(params["eta"], dtype=torch.float32, device=device)
        rho = torch.as_tensor(params["rho"], dtype=torch.float32, device=device)
        H   = torch.as_tensor(params["H"],  dtype=torch.float32, device=device)
        xi0 = torch.as_tensor(params["xi0_knots"], dtype=torch.float32, device=device).flatten()

        x_vec = torch.cat([eta.unsqueeze(0), rho.unsqueeze(0), H.unsqueeze(0), xi0], dim=0)

        # -------------------------------------------
        # 2) Scale parameters
        # -------------------------------------------
        lb = torch.tensor(self.param_bounds[0], dtype=torch.float32, device=device)
        ub = torch.tensor(self.param_bounds[1], dtype=torch.float32, device=device)

        x_scaled = 2.0 * (x_vec - lb) / (ub - lb) - 1.0

        # -------------------------------------------
        # 3) Forward MLP → scaled IVs
        # -------------------------------------------
        pred_scaled = self.forward(x_scaled.unsqueeze(0)).squeeze(0)

        # 4) Inverse output scaling – VOLLVEKTOR, nicht nur [0]
        nT, nK = self.output_shape
        nPts = nT * nK

        # pred_scaled: shape (nPts,)
        pred_flat = pred_scaled.view(-1)

        mean = torch.as_tensor(self.output_scaler.mean_,  dtype=torch.float32, device=device)   # (nPts,)
        std  = torch.as_tensor(self.output_scaler.scale_, dtype=torch.float32, device=device)   # (nPts,)

        pred_iv_flat = pred_flat * std + mean              # (nPts,)
        base_surface = pred_iv_flat.view(nT, nK)           # (nT, nK)


        # ---- If no interpolation requested → return base surface
        if grid is None:
            return base_surface

        # -------------------------------------------
        # 6) PyTorch interpolation using grid_sample
        # -------------------------------------------
        base_Ts = torch.tensor(self.maturities, dtype=torch.float32, device=device)
        base_Ks = torch.tensor(self.strikes, dtype=torch.float32, device=device)

        target_Ts = torch.tensor(grid["maturities"], dtype=torch.float32, device=device)
        target_Ks = torch.tensor(grid["strikes"], dtype=torch.float32, device=device)

        # --- Normalize base grid to [-1, 1]
        T_min, T_max = base_Ts.min(), base_Ts.max()
        K_min, K_max = base_Ks.min(), base_Ks.max()

        TT, KK = torch.meshgrid(target_Ts, target_Ks, indexing="ij")

        TTn = 2 * (TT - T_min) / (T_max - T_min) - 1
        KKn = 2 * (KK - K_min) / (K_max - K_min) - 1

        # grid_sample wants: (N, H, W, 2) with last dim = (x, y)
        grid_torch = torch.stack([KKn, TTn], dim=-1).unsqueeze(0)

        # grid_sample input must be 4D: (N, C, H, W)
        img = base_surface.unsqueeze(0).unsqueeze(0)

        surface_interp = F.grid_sample(
            img,
            grid_torch,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=True
        )
        return surface_interp.squeeze(0).squeeze(0)






# ============================================================
# Usage Notes (example)
# ============================================================
# DeepONet (with internal scaling):
#   model, train_loader, val_loader, Ks, Ts = DeepONet.from_surfaces(train_surfaces,
#       batch_size=256, val_split=0.2, eta=(0.5,4.0), rho=(0.0,1.0), H=(0.025,0.5), knot=(0.01,0.16))
#   model.train_model(train_loader, val_loader, epochs=100)
#   fig = model.plot_evaluation(test_surfaces[0])
#   model.evaluate(test_surfaces, out_dir="deeponet_eval")
#
# MLP (with internal scaling):
#   model, train_loader, val_loader, Ks, Ts = MLP.from_surfaces(train_surfaces,
#       batch_size=32, val_split=0.2, hidden_dims=(256,256,256), eta=(0.5,4.0), rho=(0.0,1.0), H=(0.025,0.5), knot=(0.01,0.16))
#   model.train_model(train_loader, val_loader, epochs=100)
#   fig = model.plot_evaluation(test_surfaces[0])
#   model.evaluate(test_surfaces, out_dir="mlp_eval")
