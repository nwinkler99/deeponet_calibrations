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

    def export_to_numpy(self, out_dir="exports", filename=None, autosave=True):
        """
        Export model weights + metadata for NumPy-only inference.
        Automatically saves to disk if autosave=True.

        Returns
        -------
        export_dict : dict
            A fully serializable dictionary for pure NumPy inference.
        """
        model_type = self.__class__.__name__
        export = {"model_type": model_type, "meta": {}, "layers": []}

        # --- MLP ---
        if model_type == "MLP":
            net = self.net
            for layer in net:
                if isinstance(layer, torch.nn.Linear):
                    W = layer.weight.detach().cpu().numpy().T  # (in, out)
                    b = layer.bias.detach().cpu().numpy()
                    act = None
                elif isinstance(layer, (torch.nn.ReLU, torch.nn.GELU, torch.nn.Tanh, torch.nn.ELU)):
                    W, b, act = None, None, layer.__class__.__name__.lower()
                else:
                    continue
                export["layers"].append({"W": W, "b": b, "act": act})

            export["meta"] = {
                "input_dim": self.input_dim,
                "output_shape": self.output_shape,
                "param_bounds": (
                    self.param_bounds[0].tolist() if self.param_bounds else None,
                    self.param_bounds[1].tolist() if self.param_bounds else None,
                ),
                "output_scaler": {
                    "mean_": self.output_scaler.mean_.tolist(),
                    "scale_": self.output_scaler.scale_.tolist(),
                } if self.output_scaler is not None else None,
                "strikes": self.strikes.tolist() if self.strikes is not None else None,
                "maturities": self.maturities.tolist() if self.maturities is not None else None,
            }



        # --- DeepONet ---
        elif model_type == "DeepONet":
            def _extract_seq(seq):
                layers = []
                for layer in seq:
                    if isinstance(layer, torch.nn.Linear):
                        W = layer.weight.detach().cpu().numpy().T
                        b = layer.bias.detach().cpu().numpy()
                        act = None
                    elif isinstance(layer, (torch.nn.ReLU, torch.nn.GELU, torch.nn.Tanh, torch.nn.ELU)):
                        W, b, act = None, None, layer.__class__.__name__.lower()
                    else:
                        continue
                    layers.append({"W": W, "b": b, "act": act})
                return layers

            export["branch_layers"] = _extract_seq(self.branch)
            export["trunk_layers"]  = _extract_seq(self.trunk)
            export["meta"] = {
                "input_dim": self.input_dim,
                "output_shape": self.output_shape,
                "param_bounds": (
                    self.param_bounds[0].tolist() if self.param_bounds else None,
                    self.param_bounds[1].tolist() if self.param_bounds else None,
                ),
                "output_scaler": {
                    "mean_": self.output_scaler.mean_.tolist(),
                    "scale_": self.output_scaler.scale_.tolist(),
                } if self.output_scaler is not None else None,
                "strikes": self.strikes.tolist() if self.strikes is not None else None,
                "maturities": self.maturities.tolist() if self.maturities is not None else None,
            }

        else:
            raise NotImplementedError(f"Unsupported model type {model_type}")

        # --- Autosave ---
        if autosave:
            os.makedirs(out_dir, exist_ok=True)
            if filename is None:
                filename = f"{model_type.lower()}_export.json"
            path = os.path.join(out_dir, filename)

            def _default(o):
                if isinstance(o, np.ndarray): return o.tolist()
                raise TypeError(f"Object of type {type(o)} not serializable")

            with open(path, "w") as f:
                json.dump(export, f, indent=2, default=_default)
            print(f"Model exported to {path}")

        return export

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

    def compute_grid_mse(self, surface_data, interp_method="spline"):
        """
        Compute MSE between predicted and true surfaces.
        If the sample grid differs from the model grid, interpolate the true surface
        onto the model's grid before comparison.

        Parameters
        ----------
        surface_data : dict
            {"iv_surface": 2D np.array, "grid": {"strikes","maturities"}, "params": dict}
        interp_method : {"spline","linear","nearest"}, default="spline"
            Interpolation method used if grids differ.

        Returns
        -------
        mse_grid : np.ndarray
            Elementwise squared error on the model grid.
        stats : dict
            Summary metrics (mean, std, max, RMSE, MAE, etc.)
        """
        assert self.strikes is not None and self.maturities is not None, \
            "Model grid (strikes/maturities) not set; call set_grid or train/prepare first."

        true_surface = np.asarray(surface_data["iv_surface"], dtype=np.float32)
        grid = surface_data.get("grid", {"strikes": self.strikes, "maturities": self.maturities})
        Ks_true = np.asarray(grid["strikes"], dtype=np.float32)
        Ts_true = np.asarray(grid["maturities"], dtype=np.float32)
        params = surface_data["params"]

        # Predict surface on model's own grid
        pred_surface = self.predict_surface(params)  # always on (self.maturities, self.strikes)

        # If grids differ, interpolate true_surface onto model's grid
        if not (np.allclose(Ks_true, self.strikes) and np.allclose(Ts_true, self.maturities)):
            if interp_method == "spline":
                interp = RectBivariateSpline(Ts_true, Ks_true, true_surface)
                true_surface_interp = interp(self.maturities, self.strikes)
            else:
                interp = RegularGridInterpolator(
                    (Ts_true, Ks_true), true_surface, method=interp_method,
                    bounds_error=False, fill_value=None
                )
                TT, KK = np.meshgrid(self.maturities, self.strikes, indexing="ij")
                coords = np.stack([TT.ravel(), KK.ravel()], axis=1)
                true_surface_interp = interp(coords).reshape(len(self.maturities), len(self.strikes))
            true_surface = true_surface_interp.astype(np.float32)

        # Compute per-grid-point error
        abs_err = np.abs(true_surface - pred_surface)
        mse_grid = abs_err ** 2

        # Locate largest absolute error
        idx_flat = np.argmax(abs_err)
        m_idx, k_idx = np.unravel_index(idx_flat, abs_err.shape)
        loc = {
            'maturity_index': int(m_idx),
            'strike_index': int(k_idx),
            'strike': float(self.strikes[k_idx]),
            'maturity': float(self.maturities[m_idx])
        }

        stats = {
            'mse_mean': float(np.mean(mse_grid)),
            'mse_std': float(np.std(mse_grid)),
            'mse_max': float(np.max(mse_grid)),
            'rmse': float(np.sqrt(np.mean(mse_grid))),
            'mae': float(np.mean(abs_err)),
            'max_abs_error': float(np.max(abs_err)),
            'max_error_location': loc
        }
        return mse_grid, stats


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

        Works even if each sample has an irregular or different (K,T) grid.
        Produces heatmaps of mean/median/max relative error [%],
        MC sampling error [%], and absolute error.
        """
        assert self.strikes is not None and self.maturities is not None, \
            "Model grid (strikes/maturities) not set; call set_grid first."
        os.makedirs(out_dir, exist_ok=True)

        nT, nK = len(self.maturities), len(self.strikes)

        # Initialize bins for predicted rel error, abs error, and MC rel error
        bin_errs_rel = [[[] for _ in range(nK)] for _ in range(nT)]
        bin_errs_abs = [[[] for _ in range(nK)] for _ in range(nT)]
        bin_errs_mc = [[[] for _ in range(nK)] for _ in range(nT)]

        for s in surface_samples:
            params = s["params"]
            true_surface = np.array(s["iv_surface"], dtype=np.float32)

            grid = s.get("grid", {"strikes": self.strikes, "maturities": self.maturities})
            Ks = np.asarray(grid["strikes"], dtype=np.float32)
            Ts = np.asarray(grid["maturities"], dtype=np.float32)

            pred_surface = self.predict_surface(params, grid=grid)
            rel_err = np.abs(true_surface - pred_surface) / np.clip(true_surface, 1e-6, None) * 100.0
            abs_err = np.abs(true_surface - pred_surface)

            mc_rel_err = np.array(s.get("iv_rel_error", np.zeros_like(true_surface)), dtype=np.float32) * 100.0

            # --- fill NaNs in mc_rel_err dynamically ---
            nan_mask = np.isnan(mc_rel_err)
            if np.any(nan_mask):
                mc_rel_err = np.where(nan_mask, 0.0, mc_rel_err)  # temporary fill

            # Assign errors to nearest bin on base grid
            for ti, T in enumerate(Ts):
                t_idx = np.argmin(np.abs(self.maturities - T))
                for ki, K in enumerate(Ks):
                    k_idx = np.argmin(np.abs(self.strikes - K))

                    # --- predicted errors ---
                    bin_errs_rel[t_idx][k_idx].append(rel_err[ti, ki])
                    bin_errs_abs[t_idx][k_idx].append(abs_err[ti, ki])

                    # --- MC rel error (handle NaN replacement with bin mean) ---
                    val = mc_rel_err[ti, ki]
                    if np.isnan(val):
                        prev_vals = bin_errs_mc[t_idx][k_idx]
                        if prev_vals:
                            val = float(np.mean(prev_vals))  # replace with bin mean so far
                        else:
                            val = 0.0
                    bin_errs_mc[t_idx][k_idx].append(val)

        def aggregate_bins(bin_errs):
            mean = np.full((nT, nK), np.nan, dtype=np.float32)
            median = np.full((nT, nK), np.nan, dtype=np.float32)
            maxv = np.full((nT, nK), np.nan, dtype=np.float32)
            for t in range(nT):
                for k in range(nK):
                    vals = bin_errs[t][k]
                    if vals:
                        mean[t, k] = np.mean(vals)
                        median[t, k] = np.median(vals)
                        maxv[t, k] = np.max(vals)
            global_mean = np.nanmean(mean)
            for arr in [mean, median, maxv]:
                arr[np.isnan(arr)] = global_mean
            return mean, median, maxv, global_mean

        mean_rel, median_rel, max_rel, global_mean_rel = aggregate_bins(bin_errs_rel)
        mean_abs, median_abs, max_abs, global_mean_abs = aggregate_bins(bin_errs_abs)
        mean_mc, median_mc, max_mc, global_mean_mc = aggregate_bins(bin_errs_mc)

        # --- Plotting ---
        Ks_mesh, Ts_mesh = np.meshgrid(self.strikes, self.maturities, indexing="xy")

        def plot_set(data_triplet, titles, fname, label):
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for ax, data, title in zip(axes, data_triplet, titles):
                im = ax.pcolormesh(Ks_mesh, Ts_mesh, data, cmap="magma", shading="auto")
                ax.set_xlabel("Strike (K)")
                ax.set_ylabel("Maturity (T)")
                ax.set_title(f"{title}")
                ax.invert_yaxis()
                fig.colorbar(im, ax=ax, label=label)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, fname), dpi=200)
            plt.close(fig)

        # Predicted relative error
        plot_set(
            [mean_rel, median_rel, max_rel],
            ["Mean Rel Error (Pred)", "Median Rel Error (Pred)", "Max Rel Error (Pred)"],
            "iv_error_heatmaps_binned.png",
            "%"
        )

        # Absolute error
        plot_set(
            [mean_abs, median_abs, max_abs],
            ["Mean Abs Error (Pred)", "Median Abs Error (Pred)", "Max Abs Error (Pred)"],
            "iv_abs_error_heatmaps_binned.png",
            "abs(IV diff)"
        )

        # Monte Carlo sampling relative error
        plot_set(
            [mean_mc, median_mc, max_mc],
            ["Mean Rel Error (MC)", "Median Rel Error (MC)", "Max Rel Error (MC)"],
            "iv_mc_rel_error_heatmaps_binned.png",
            "%"
        )

        return {
            "pred_rel": {"mean": mean_rel, "median": median_rel, "max": max_rel},
            "pred_abs": {"mean": mean_abs, "median": median_abs, "max": max_abs},
            "mc_rel": {"mean": mean_mc, "median": median_mc, "max": max_mc},
            "global_mean": {
                "pred_rel": float(global_mean_rel),
                "pred_abs": float(global_mean_abs),
                "mc_rel": float(global_mean_mc),
            },
        }

    # ============================================================
    # Calibration utilities
    # ============================================================
    def calibrate(self, target_surface, optimiser="L-BFGS-B", bounds=None, maxiter=1000, verbose=False):
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
            return np.sqrt(np.mean((pred - true_surface) ** 2))

        def residuals(x_phys):
            params = {
                "eta": x_phys[0],
                "rho": x_phys[1],
                "H": x_phys[2],
                "xi0_knots": x_phys[3:]
            }
            pred = self.predict_surface(params, grid={"strikes": Ks, "maturities": Ts})
            return (pred - true_surface).ravel()

        x0 = 0.5 * (lb + ub)
        
        t0 = time.perf_counter()

        opt_lower = optimiser.lower()
        if opt_lower == "differential evolution":
            res = differential_evolution(rmse_objective, bounds=list(zip(lb, ub)), maxiter=maxiter, disp=verbose)
        elif opt_lower in ["levenberg-marquardt", "lm"]:
            res = least_squares(residuals, x0, method="trf", bounds=(lb, ub), max_nfev=maxiter, verbose=2 if verbose else 0)
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



    # ------------------------------------------------------------

    def evaluate_calibrate(self, surfaces, optimiser="L-BFGS-B", maxiter=500, out_dir="calibration_eval"):
        """
        Run calibration across multiple surfaces using a single optimizer,
        producing per-parameter error statistics, CDF plots, and returning
        full true/estimated parameter arrays.

        Parameters
        ----------
        surfaces : list of dict
            Each surface dict: {"iv_surface": ..., "grid": ..., "params": ...}
        optimiser : str
            Optimizer to use (e.g. 'L-BFGS-B', 'SLSQP', etc.)
        maxiter : int
        out_dir : str
            Directory to save plots.
        """

        os.makedirs(out_dir, exist_ok=True)
        print(f"\nEvaluating calibration using {optimiser} on {len(surfaces)} surfaces...")

        runtimes, rmses = [], []
        per_param_errors = {}   # key: param_name -> list of rel errors
        true_params_all = []    # list of physical true param vectors
        est_params_all = []     # list of calibrated param vectors

        for i, s in enumerate(surfaces, start=1):
            r = self.calibrate(s, optimiser=optimiser, maxiter=maxiter)
            runtimes.append(r["runtime_ms"])
            rmses.append(r["rmse"])
            est_params_all.append(r["theta_hat"])

            # true parameters (in physical units)
            tp = s.get("params", None)
            if tp is not None:
                tvec = np.concatenate([
                    [tp["eta"], tp["rho"], tp["H"]],
                    np.array(tp["xi0_knots"], dtype=np.float32).ravel()
                ])
                true_params_all.append(tvec)
            else:
                true_params_all.append(np.full_like(r["theta_hat"], np.nan))

            # per-parameter errors
            for k, v in r["error_rel_dict"].items():
                per_param_errors.setdefault(k, []).append(v)

            # Verbose progress every 50 surfaces
            if i % 50 == 0 or i == len(surfaces):
                mean_rmse = np.mean(rmses)
                print(f"  [{i}/{len(surfaces)}]  mean RMSE={mean_rmse:.5f}  "
                    f"avg time={np.mean(runtimes):.1f} ms")
                summary_str = "     " + "  ".join(
                    [f"{k}: {np.mean(v_list)*100:.2f}%" for k, v_list in per_param_errors.items()])
                print(summary_str)

        # Convert dicts/lists to arrays
        for k in per_param_errors:
            per_param_errors[k] = np.array(per_param_errors[k])
        runtimes = np.array(runtimes)
        rmses = np.array(rmses)
        true_params_all = np.array(true_params_all)
        est_params_all = np.array(est_params_all)

        avg_time = np.mean(runtimes)
        print(f"\n→ Final avg time: {avg_time:.1f} ms, mean RMSE={np.mean(rmses):.5f}")
        print("\nMean relative errors per parameter:")
        for k, vals in per_param_errors.items():
            print(f"   {k:<8s} | mean={np.mean(vals)*100:.3f}% | median={np.median(vals)*100:.3f}% | std={np.std(vals)*100:.3f}%")

        # --- Plot ECDFs per parameter ---
        def ecdf(x):
            xs = np.sort(x)
            ys = np.linspace(0, 1, len(x))
            return xs, ys

        n_params = len(per_param_errors)
        ncols = min(4, n_params)
        nrows = int(np.ceil(n_params / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), squeeze=False)

        for i, (k, vals) in enumerate(per_param_errors.items()):
            ax = axes[i // ncols, i % ncols]
            xs, ys = ecdf(vals)
            ax.plot(ys, xs * 100)
            ax.set_title(f"{k}")
            ax.set_xlabel("Quantiles")
            ax.set_ylabel("Rel. Error [%]")
            ax.grid(True, ls=":", lw=0.5)

        for j in range(i + 1, nrows * ncols):
            axes[j // ncols, j % ncols].axis("off")

        plt.suptitle(f"Parameter Relative Error CDFs ({optimiser})")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(os.path.join(out_dir, f"param_error_cdfs_{optimiser}.png"), dpi=200)
        plt.close()

        # --- RMSE CDF ---
        plt.figure(figsize=(6, 4))
        xs, ys = ecdf(rmses)
        plt.plot(ys, xs)
        plt.axvline(0.99, ls="--", c="k")
        plt.xlabel("Quantiles")
        plt.ylabel("RMSE")
        plt.title(f"{optimiser}: Surface RMSE CDF")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"rmse_cdf_{optimiser}.png"), dpi=200)
        plt.close()

        return {
            "optimizer": optimiser,
            "avg_time_ms": float(avg_time),
            "mean_rmse": float(np.mean(rmses)),
            "per_param_errors": per_param_errors,
            "true_params": true_params_all,     # shape (N, n_params)
            "est_params": est_params_all,       # shape (N, n_params)
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

class DeepONet(BaseModel):
    """
    Deep Operator Network for implied volatility surfaces.
    - Branch input: parameter vector θ (eta, rho, H, xi0_knots...)
    - Trunk input: 2D coords (K, T)
    - Optional learnable mask acting as contextual filter between branch and trunk.
    """
    def __init__(self,
                 branch_in_dim=None,
                 trunk_in_dim=2,
                 latent_dim=64,
                 branch_hidden_dims=(64, 64),
                 trunk_hidden_dims=(64, 64),
                 activation="relu",
                 lr=1e-3,
                 mask_type="none"):  # new flag
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

        # Branch network
        branch_layers = []
        dims = [self.branch_in_dim] + list(self.branch_hidden_dims)
        for i in range(len(dims) - 1):
            branch_layers.append(nn.Linear(dims[i], dims[i + 1]))
            branch_layers.append(act_fn)
        branch_layers.append(nn.Linear(dims[-1], self.latent_dim))
        self.branch = nn.Sequential(*branch_layers)

        # Trunk network
        trunk_layers = []
        dims_t = [self.trunk_in_dim] + list(self.trunk_hidden_dims)
        for i in range(len(dims_t) - 1):
            trunk_layers.append(nn.Linear(dims_t[i], dims_t[i + 1]))
            trunk_layers.append(act_fn)
        trunk_layers.append(nn.Linear(dims_t[-1], self.latent_dim))
        self.trunk = nn.Sequential(*trunk_layers)

        # Optional mask
        self.mask_net = self._build_mask(act_fn)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    # --------------------------------------------------------
    def _build_mask(self, act_fn):
        """Construct a learnable mask network depending on mask_type."""
        if self.mask_type == "none":
            return None

        elif self.mask_type == "spatial":
            # Mask depends only on (K,T)
            return nn.Sequential(
                nn.Linear(self.trunk_in_dim, 32),
                act_fn,
                nn.Linear(32, 16),
                act_fn,
                nn.Linear(16, 1),
                nn.Sigmoid()
            )

        elif self.mask_type == "contextual":
            # Mask depends on both branch and trunk context
            return nn.Sequential(
                nn.Linear(self.latent_dim * 2, 128),
                act_fn,
                nn.Linear(128, self.latent_dim),
                nn.Sigmoid()
            )

        elif self.mask_type == "channel":
            # Mask modulates each latent feature per (K,T)
            return nn.Sequential(
                nn.Linear(self.trunk_in_dim, self.latent_dim),
                nn.Sigmoid()
            )

        else:
            raise ValueError(f"Unknown mask_type '{self.mask_type}'")

    # --------------------------------------------------------
    def forward(self, xb, xt):
        b = self.branch(xb)  # (batch, latent_dim)
        t = self.trunk(xt)   # (batch, latent_dim)

        if self.mask_type == "none":
            return torch.sum(b * t, dim=1, keepdim=True)

        elif self.mask_type == "spatial":
            mask = self.mask_net(xt)  # (batch,1)
            return torch.sum(b * t, dim=1, keepdim=True) * mask

        elif self.mask_type == "channel":
            mask = self.mask_net(xt)  # (batch,latent_dim)
            return torch.sum(b * t * mask, dim=1, keepdim=True)

        elif self.mask_type == "contextual":
            mask_input = torch.cat([b, t], dim=1)
            mask = self.mask_net(mask_input)  # (batch,latent_dim)
            return torch.sum(b * t * mask, dim=1, keepdim=True)

    # --------------------------------------------------------
    def test_mask_response(self, xb_sample, xt_grid, visualize=True):
        """
        Evaluate and optionally visualize the mask response for a fixed branch vector
        across a grid of (K,T) coordinates.

        Parameters
        ----------
        xb_sample : torch.Tensor or np.ndarray
            Single branch input vector θ.
        xt_grid : np.ndarray
            Array of (K,T) points with shape (nT*nK, 2).
        visualize : bool
            If True, plot the resulting mask surface.
        """
        import matplotlib.pyplot as plt

        self.eval()
        if self.mask_net is None:
            print("No mask network defined (mask_type='none').")
            return None

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
                mask = torch.mean(mask, dim=1, keepdim=True)  # avg per channel
            elif self.mask_type == "contextual":
                mask_input = torch.cat([b, t], dim=1)
                mask = self.mask_net(mask_input)
                mask = torch.mean(mask, dim=1, keepdim=True)
            else:
                mask = torch.ones(len(xt), 1, device=self.device)

        mask_np = mask.detach().cpu().numpy().reshape(-1)

        if visualize:
            nK = len(np.unique(xt_grid[:, 0]))
            nT = len(np.unique(xt_grid[:, 1]))
            mask_surf = mask_np.reshape(nT, nK)
            plt.figure(figsize=(6, 4))
            plt.imshow(mask_surf, origin="lower", aspect="auto", cmap="magma")
            plt.colorbar(label="Mask Intensity")
            plt.title(f"Mask Response ({self.mask_type})")
            plt.xlabel("Strike Index")
            plt.ylabel("Maturity Index")
            plt.tight_layout()
            plt.show()

        return mask_np

    # --------------------------------------------------------
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

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

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
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
        Train the DeepONet model with optional learning-rate schedule, timing, and ETA display.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader (xb, xt, y) batches.
        val_loader : DataLoader, optional
            Validation data loader.
        epochs : int, default=10
            Total number of training epochs.
        lr_schedule : list of (int, float), optional
            List of (epoch, lr) tuples defining the learning-rate schedule.
            Example: [(0, 1e-3), (30, 5e-4), (60, 1e-4)]
            Learning rate changes at each specified epoch threshold.
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

            # --- Learning rate schedule update ---
            if (
                schedule_index + 1 < len(lr_schedule)
                and epoch >= lr_schedule[schedule_index + 1][0]
            ):
                schedule_index += 1
                new_lr = lr_schedule[schedule_index][1]
                for g in self.optimizer.param_groups:
                    g["lr"] = new_lr
                print(f"→ Adjusted learning rate to {new_lr:.2e} at epoch {epoch}")

            # --- Training loop ---
            self.train()
            total_loss = 0.0
            for xb, xt, y in tqdm(train_loader, desc=f"Train {epoch+1}", leave=False):
                xb, xt, y = xb.to(self.device), xt.to(self.device), y.to(self.device)
                pred = self.forward(xb, xt)
                loss = self.compute_loss(pred, y)
                if hasattr(self.mask_net, "loss_reg"):
                    loss += self.mask_net.loss_reg
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * len(y)

            train_rmse = float(np.sqrt(total_loss / len(train_loader.dataset)))

            # --- Validation ---
            if val_loader is not None:
                val_rmse = float(np.sqrt(self.validate(val_loader)))
                msg = (f"Epoch {epoch+1:03d} | "
                    f"train_rmse={train_rmse:.6f}, val_rmse={val_rmse:.6f}, "
                    f"lr={self.optimizer.param_groups[0]['lr']:.1e}")
            else:
                msg = (f"Epoch {epoch+1:03d} | "
                    f"train_rmse={train_rmse:.6f}, "
                    f"lr={self.optimizer.param_groups[0]['lr']:.1e}")

            # --- Timing & ETA ---
            epoch_time = time.time() - epoch_start
            epoch_durations.append(epoch_time)
            avg_time = np.mean(epoch_durations)
            remaining_epochs = epochs - (epoch + 1)
            eta = remaining_epochs * avg_time

            msg += f", time={epoch_time:.2f}s, ETA={eta/60:.2f} min"
            print(msg)

        # --- Summary ---
        total_time = time.time() - start_time
        print(f"\n✅ Training completed in {total_time/60:.2f} min "
            f"(avg {total_time/epochs:.2f}s per epoch)")

        # --- Automatic export ---
        self.export_to_numpy(
            out_dir="exports",
            filename=f"{self.__class__.__name__.lower()}_final.json"
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
    def predict_surface(self, params, grid=None):
        """
        Predict implied vol surface for a given parameter set and optional grid.

        Parameters
        ----------
        params : dict
            Model parameters (eta, rho, H, xi0_knots)
        grid : dict, optional
            Custom grid with "strikes" and "maturities" arrays.
            If None, uses the model's stored self.strikes/self.maturities.

        Returns
        -------
        surface : np.ndarray
            Predicted IV surface in ORIGINAL scale, shape (nT, nK)
        """
        # Determine which grid to use
        if grid is not None:
            strikes = np.asarray(grid["strikes"], dtype=np.float32)
            maturities = np.asarray(grid["maturities"], dtype=np.float32)
        else:
            assert self.strikes is not None and self.maturities is not None, \
                "No grid provided and no default grid stored; set one via set_grid()."
            strikes, maturities = self.strikes, self.maturities

        # Parameter scaling
        assert self.param_bounds is not None, "param_bounds must be set for scaling"
        assert self.output_scaler is not None, "output_scaler must be fitted"

        xi0_knots = np.array(params["xi0_knots"], dtype=np.float32).flatten()
        param_vec = np.concatenate([[params["eta"], params["rho"], params["H"]], xi0_knots]).astype(np.float32)
        xb_np = self.scale_params(param_vec)  # [-1, 1] scaling

        # Build trunk coords on the chosen grid
        K_mesh, T_mesh = np.meshgrid(strikes, maturities)
        trunk_coords = np.stack([K_mesh.ravel(), T_mesh.ravel()], axis=1).astype(np.float32)

        # Torch forward pass
        with torch.no_grad():
            xb = torch.tensor(xb_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            xb = xb.repeat(len(trunk_coords), 1)
            xt = torch.tensor(trunk_coords, dtype=torch.float32, device=self.device)
            pred_scaled = self.forward(xb, xt)

        pred_scaled_np = pred_scaled.detach().cpu().numpy().reshape(-1)
        pred_iv = self.inverse_transform_output_single(pred_scaled_np)
        surface = pred_iv.reshape(len(maturities), len(strikes))
        return surface



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
        out = self.net(x)
        return out.view(-1, *self.output_shape)

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
        Train the MLP model with optional learning rate schedule, timing, and ETA display.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader.
        val_loader : DataLoader, optional
            Validation data loader.
        epochs : int, default=50
            Total number of training epochs.
        lr_schedule : list of (int, float), optional
            List of (epoch, lr) tuples defining when to update the learning rate.
            Example: [(0, 1e-3), (30, 5e-4), (60, 1e-4)]
            Learning rate changes at each specified epoch (>= threshold).
        """
        lr_schedule = sorted(lr_schedule, key=lambda x: x[0])
        schedule_index = 0

        # Initialize to first LR
        base_lr = lr_schedule[0][1]
        for g in self.optimizer.param_groups:
            g["lr"] = base_lr

        start_time = time.time()
        epoch_durations = []

        for epoch in range(epochs):
            epoch_start = time.time()

            # --- Learning rate schedule update ---
            if (
                schedule_index + 1 < len(lr_schedule)
                and epoch >= lr_schedule[schedule_index + 1][0]
            ):
                schedule_index += 1
                new_lr = lr_schedule[schedule_index][1]
                for g in self.optimizer.param_groups:
                    g["lr"] = new_lr
                print(f"→ Adjusted learning rate to {new_lr:.2e} at epoch {epoch}")

            # --- Training ---
            self.train()
            total = 0.0
            for x, y in tqdm(train_loader, desc=f"Train {epoch+1}", leave=False):
                x, y = x.to(self.device), y.to(self.device)
                pred = self.forward(x)
                loss = self.criterion(pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total += loss.item() * len(y)

            train_rmse = float(np.sqrt(total / len(train_loader.dataset)))

            # --- Validation ---
            if val_loader is not None:
                val_rmse = float(np.sqrt(self.validate(val_loader)))
                msg = (f"Epoch {epoch+1:03d} | "
                    f"train_rmse={train_rmse:.6f}, val_rmse={val_rmse:.6f}, "
                    f"lr={self.optimizer.param_groups[0]['lr']:.1e}")
            else:
                msg = (f"Epoch {epoch+1:03d} | "
                    f"train_rmse={train_rmse:.6f}, "
                    f"lr={self.optimizer.param_groups[0]['lr']:.1e}")

            # --- Timing and ETA ---
            epoch_time = time.time() - epoch_start
            epoch_durations.append(epoch_time)
            avg_time = np.mean(epoch_durations)
            remaining_epochs = epochs - (epoch + 1)
            eta = remaining_epochs * avg_time

            msg += f", time={epoch_time:.2f}s, ETA={eta/60:.2f} min"
            print(msg)

        total_time = time.time() - start_time
        print(f"\n✅ Training completed in {total_time/60:.2f} min "
            f"(avg {total_time/epochs:.2f}s per epoch)")
        
        # --- Automatic export after training ---
        self.export_to_numpy(
            out_dir="exports",
            filename=f"{self.__class__.__name__.lower()}_final.json"
        )




    def validate(self, val_loader):
        self.eval(); total = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                total += self.criterion(self.forward(x), y).item() * len(y)
        return total / len(val_loader.dataset)   

    # --------------------------------------------------------
    def predict_surface(self, params, grid=None, interp_method="spline"):
        """
        Predict implied volatility surface for given parameters, optionally interpolated
        onto a custom (strike, maturity) grid.

        Parameters
        ----------
        params : dict
            {"eta", "rho", "H", "xi0_knots"} defining the parameter vector.
        grid : dict, optional
            If provided, must contain "strikes" and "maturities" arrays defining
            the target evaluation grid.
            If None, uses the model's own self.strikes / self.maturities grid.
        interp_method : {"spline", "linear", "nearest"}, default="spline"
            Interpolation method used when transforming to a custom grid.

        Returns
        -------
        surface : np.ndarray
            Predicted IV surface in ORIGINAL (unscaled) space,
            shape (len(grid["maturities"]), len(grid["strikes"])).
        """
        assert self.output_shape is not None, "MLP needs output_shape set."
        assert self.strikes is not None and self.maturities is not None, \
            "MLP needs self.strikes/self.maturities set; call set_grid or train/prepare first."
        assert self.param_bounds is not None, "param_bounds must be set for scaling"
        assert self.output_scaler is not None, "output_scaler must be fitted"

        # 1️⃣ Prepare scaled parameter vector
        xi0_knots = np.array(params["xi0_knots"], dtype=np.float32).flatten()
        x = np.concatenate([[params["eta"], params["rho"], params["H"]], xi0_knots]).astype(np.float32)
        x_scaled = self.scale_params(x)

        # 2️⃣ Predict on the model's native base grid
        with torch.no_grad():
            x_t = torch.tensor(x_scaled, dtype=torch.float32, device=self.device).unsqueeze(0)
            pred_scaled = self.forward(x_t)  # (1, nT, nK)
        pred_scaled_np = pred_scaled.squeeze(0).detach().cpu().numpy()
        base_surface = self.inverse_transform_surface(pred_scaled_np)  # (nT, nK)
        base_Ks, base_Ts = self.strikes, self.maturities

        # 3️⃣ If no grid given → return base surface directly
        if grid is None:
            return base_surface

        # 4️⃣ Otherwise interpolate to the target grid
        Ks_target = np.asarray(grid["strikes"], dtype=np.float32)
        Ts_target = np.asarray(grid["maturities"], dtype=np.float32)

        same_shape = (
            base_Ks is not None
            and base_Ts is not None
            and len(base_Ks) == len(Ks_target)
            and len(base_Ts) == len(Ts_target)
        )

        if same_shape:
            if np.allclose(Ks_target, base_Ks) and np.allclose(Ts_target, base_Ts):
                # identical grid → no interpolation
                return base_surface

        # --- Interpolation ---
        if interp_method == "spline":
            interp = RectBivariateSpline(base_Ts, base_Ks, base_surface)
            surface_interp = interp(Ts_target, Ks_target)
        else:
            from scipy.interpolate import RegularGridInterpolator
            kind = interp_method  # "linear" or "nearest"
            interp = RegularGridInterpolator(
                (base_Ts, base_Ks), base_surface, method=kind, bounds_error=False, fill_value=None
            )
            TT, KK = np.meshgrid(Ts_target, Ks_target, indexing="ij")
            coords = np.stack([TT.ravel(), KK.ravel()], axis=1)
            surface_interp = interp(coords).reshape(len(Ts_target), len(Ks_target))

        return surface_interp.astype(np.float32)


class NumpyModel:
    """
    Lightweight NumPy-only model for inference from exported PyTorch MLP or DeepONet.
    Reconstructs the architecture from exported weights, applies scaling, and predicts.
    """

    # ------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------
    def __init__(self, export_dict):
        self.model_type = export_dict["model_type"]
        self.meta = export_dict["meta"]

        # ---------------- Load layer weights ----------------
        if self.model_type == "MLP":
            self.layers = export_dict["layers"]
        elif self.model_type == "DeepONet":
            self.branch_layers = export_dict["branch_layers"]
            self.trunk_layers = export_dict["trunk_layers"]
        else:
            raise ValueError(f"Unsupported model type {self.model_type}")

        # ---------------- Load param bounds -----------------
        pb = self.meta.get("param_bounds")
        if pb and all(pb):
            self.param_bounds = (
                np.array(pb[0], dtype=np.float32),
                np.array(pb[1], dtype=np.float32)
            )
        else:
            self.param_bounds = None

        # ---------------- Load output scaler ----------------
        scaler = self.meta.get("output_scaler")
        if scaler is not None:
            self.out_mean = np.array(scaler["mean_"], dtype=np.float32)
            self.out_scale = np.array(scaler["scale_"], dtype=np.float32)
        else:
            self.out_mean = None
            self.out_scale = None

        # Cached shape info
        shape = self.meta.get("output_shape", None)
        self.output_shape = tuple(shape) if isinstance(shape, (list, tuple)) else None
        self.input_dim = self.meta.get("input_dim", None)

        # Add base grid info
        self.strikes = np.array(self.meta.get("strikes"), dtype=np.float32) if self.meta.get("strikes") is not None else None
        self.maturities = np.array(self.meta.get("maturities"), dtype=np.float32) if self.meta.get("maturities") is not None else None


    # ------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------
    @classmethod
    def load(cls, path):
        """Load a JSON export produced by BaseModel.export_to_numpy()."""
        with open(path, "r") as f:
            data = json.load(f)
        print(f"Loaded NumPy model from {path} ({data['model_type']})")
        return cls(data)

    # ------------------------------------------------------------
    # Scaling utilities
    # ------------------------------------------------------------
    def _scale_to_m1_p1(self, x, lb, ub):
        mid = 0.5 * (ub + lb)
        return (x - mid) * (2.0 / (ub - lb))

    def _inverse_from_m1_p1(self, x_scaled, lb, ub):
        mid = 0.5 * (ub + lb)
        return x_scaled * (0.5 * (ub - lb)) + mid

    def scale_params(self, x_raw):
        """Scale raw physical params [eta, rho, H, xi0...] → [-1,1]."""
        assert self.param_bounds is not None, "param_bounds missing in export"
        lb, ub = self.param_bounds
        x_raw = np.asarray(x_raw, dtype=np.float32)
        return self._scale_to_m1_p1(x_raw, lb, ub)

    def inverse_scale_params(self, x_scaled):
        """Inverse of scale_params."""
        assert self.param_bounds is not None, "param_bounds missing in export"
        lb, ub = self.param_bounds
        x_scaled = np.asarray(x_scaled, dtype=np.float32)
        return self._inverse_from_m1_p1(x_scaled, lb, ub)

    # ------------------------------------------------------------
    # Output scaling (StandardScaler)
    # ------------------------------------------------------------
    def transform_output(self, Y):
        """Apply StandardScaler normalization."""
        if self.out_mean is None or self.out_scale is None:
            return Y
        return (Y - self.out_mean) / self.out_scale

    def inverse_transform_output(self, Y_scaled):
        """Revert StandardScaler normalization (scaled → original)."""
        if self.out_mean is None or self.out_scale is None:
            return Y_scaled
        return Y_scaled * self.out_scale + self.out_mean

    # ------------------------------------------------------------
    # Activation functions
    # ------------------------------------------------------------
    def _activation(self, name, x):
        if name is None:
            return x
        name = name.lower()
        if name == "relu":
            return np.maximum(x, 0)
        elif name == "tanh":
            return np.tanh(x)
        elif name == "gelu":
            return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
        elif name == "elu":
            return np.where(x > 0, x, np.exp(x) - 1)
        else:
            raise ValueError(f"Unsupported activation: {name}")

    # ------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------
    def _forward_seq(self, x, layers):
        """Forward through a serialized layer list (weights + activations)."""
        a = np.array(x, dtype=np.float32)
        for layer in layers:
            W, b, act = layer["W"], layer["b"], layer["act"]
            if W is not None and b is not None:
                a = a @ np.array(W, dtype=np.float32) + np.array(b, dtype=np.float32)
            if act:
                a = self._activation(act, a)
        return a

    # ------------------------------------------------------------
    # Predict (core)
        # ------------------------------------------------------------

    def predict_surface(self, params, grid=None, interp_method="spline", verbose=True):
        """
        Unified prediction interface for MLP and DeepONet NumPy models.
        Measures per-step runtime if verbose=True.
        """
        # --- Scale branch/param inputs ---
        x_raw = np.concatenate(
            [[params["eta"], params["rho"], params["H"]], params["xi0_knots"]]
        )[None, :]
        x_scaled = self.scale_params(x_raw)

        # =====================================================
        # MLP case
        # =====================================================
        if self.model_type == "MLP":
            assert self.output_shape is not None, "MLP export missing output_shape."

            t0 = time.perf_counter()
            y_scaled = self._forward_seq(x_scaled, self.layers)
            t1 = time.perf_counter()

            y = self.inverse_transform_output(y_scaled)
            t2 = time.perf_counter()

            nT, nK = self.output_shape
            surface = y.reshape(-1, nT, nK)[0]
            t3 = time.perf_counter()

            base_Ks = self.strikes
            base_Ts = self.maturities

            # --- Optional interpolation to custom grid ---
            if isinstance(grid, dict):
                Ks_target = np.asarray(grid["strikes"], dtype=np.float32)
                Ts_target = np.asarray(grid["maturities"], dtype=np.float32)

                same_shape = (
                    base_Ks is not None and base_Ts is not None
                    and len(base_Ks) == len(Ks_target)
                    and len(base_Ts) == len(Ts_target)
                )

                if same_shape and np.allclose(Ks_target, base_Ks) and np.allclose(Ts_target, base_Ts):
                    if verbose:
                        print(f"[MLP timing]")
                        print(f"  Forward        : {t1 - t0:.4f} s")
                        print(f"  Inverse scale  : {t2 - t1:.4f} s")
                        print(f"  Reshape        : {t3 - t2:.4f} s")
                        print(f"  Total          : {t3 - t0:.4f} s\n")
                    return surface

                # otherwise, perform interpolation
                if interp_method == "spline":
                    from scipy.interpolate import RectBivariateSpline
                    interp = RectBivariateSpline(base_Ts, base_Ks, surface)
                    surface_interp = interp(Ts_target, Ks_target)
                else:
                    from scipy.interpolate import RegularGridInterpolator
                    interp = RegularGridInterpolator(
                        (base_Ts, base_Ks),
                        surface,
                        method=interp_method,
                        bounds_error=False,
                        fill_value=None,
                    )
                    TT, KK = np.meshgrid(Ts_target, Ks_target, indexing="ij")
                    coords = np.stack([TT.ravel(), KK.ravel()], axis=1)
                    surface_interp = interp(coords).reshape(len(Ts_target), len(Ks_target))

                t4 = time.perf_counter()
                if verbose:
                    print(f"[MLP timing]")
                    print(f"  Forward        : {t1 - t0:.4f} s")
                    print(f"  Inverse scale  : {t2 - t1:.4f} s")
                    print(f"  Reshape        : {t3 - t2:.4f} s")
                    print(f"  Interpolation  : {t4 - t3:.4f} s")
                    print(f"  Total          : {t4 - t0:.4f} s\n")
                return surface_interp.astype(np.float32)

            if verbose:
                print(f"[MLP timing]")
                print(f"  Forward        : {t1 - t0:.4f} s")
                print(f"  Inverse scale  : {t2 - t1:.4f} s")
                print(f"  Reshape        : {t3 - t2:.4f} s")
                print(f"  Total          : {t3 - t0:.4f} s\n")

            return surface.astype(np.float32)

        # =====================================================
        # DeepONet case
        # =====================================================
        elif self.model_type == "DeepONet":
            assert isinstance(grid, dict), \
                "For DeepONet, provide grid={'strikes': ..., 'maturities': ...}"

            Ks = np.asarray(grid["strikes"], dtype=np.float32)
            Ts = np.asarray(grid["maturities"], dtype=np.float32)
            K_mesh, T_mesh = np.meshgrid(Ks, Ts, indexing="xy")
            trunk_coords = np.stack([K_mesh.ravel(), T_mesh.ravel()], axis=1)

            t0 = time.perf_counter()
            B = self._forward_seq(x_scaled, self.branch_layers)
            t1 = time.perf_counter()
            T = self._forward_seq(trunk_coords, self.trunk_layers)
            t2 = time.perf_counter()
            y_scaled = np.sum(B * T, axis=1, keepdims=True)
            t3 = time.perf_counter()
            y = self.inverse_transform_output(y_scaled).reshape(len(Ts), len(Ks))
            t4 = time.perf_counter()

            if verbose:
                print(f"[DeepONet timing]")
                print(f"  Branch forward : {t1 - t0:.4f} s")
                print(f"  Trunk forward  : {t2 - t1:.4f} s")
                print(f"  Fusion (B*T)   : {t3 - t2:.4f} s")
                print(f"  Inverse scale  : {t4 - t3:.4f} s")
                print(f"  Total          : {t4 - t0:.4f} s\n")

            return y.astype(np.float32)

        else:
            raise ValueError(f"Unsupported model type {self.model_type}")




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
