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
from torch.func import jacrev
import torch.nn.functional as F
import time
import numpy as np
from scipy.optimize import least_squares, differential_evolution, minimize
import QuantLib as ql
from generation.rbergomi import bsinv  # Pfad ggf. anpassen

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

        # -------------------------
        # Shared contextual info
        # -------------------------
        self.strikes = None
        self.maturities = None
        self.input_dim = None
        self.output_shape = None

        self.param_bounds = None      # (lb, ub)
        self.output_scaler = None     # StandardScaler

        self.param_names = None      # list[str]
        self.param_slices = None     # dict[str, slice|int]

        # -------------------------
        # Hyperparameter placeholders
        # (covering MLP & DeepONet)
        # -------------------------

        # MLP
        self.hidden_dims = None
        self.activation = None
        self.lr = None

        # DeepONet
        self.branch_in_dim = None
        self.trunk_in_dim = None
        self.latent_dim = None
        self.branch_hidden_dims = None
        self.trunk_hidden_dims = None
        self.mask_type = None

        # caches (not saved)
        self._last_pred = None
        self._last_true = None
        self._last_params = None

    # ============================================================
    #   UNIVERSAL SAVE / LOAD FOR MLP + DEEPONET
    # ============================================================

    def save(self, path):
        """
        Save model architecture + weights + scaler + grid + bounds.
        Automatically detects whether model is MLP or DeepONet.
        """
        import torch
        from sklearn.preprocessing import StandardScaler
        from models.framework import DeepONet, MLP

        # --------------------------
        # 1) Detect model type
        # --------------------------
        if isinstance(self, DeepONet):
            init_kwargs = {
                "branch_in_dim": self.branch_in_dim,
                "trunk_in_dim": self.trunk_in_dim,
                "latent_dim": self.latent_dim,
                "branch_hidden_dims": self.branch_hidden_dims,
                "trunk_hidden_dims": self.trunk_hidden_dims,
                "activation": self.activation,
                "lr": self.lr,
                "mask_type": self.mask_type,
            }
            class_name = "DeepONet"

        elif isinstance(self, MLP):
            init_kwargs = {
                "input_dim": self.input_dim,
                "output_shape": self.output_shape,
                "hidden_dims": self.hidden_dims,
                "activation": self.activation,
                "lr": self.lr,
            }
            class_name = "MLP"

        else:
            raise ValueError(f"Unsupported model type for saving: {self.__class__.__name__}")

        # --------------------------
        # 2) Build checkpoint dict
        # --------------------------
        payload = {
            "class_name": class_name,
            "state_dict": self.state_dict(),
            "init_kwargs": init_kwargs,

            # grid + normalisation
            "strikes": self.strikes,
            "maturities": self.maturities,
            "param_bounds": self.param_bounds,

            # parameter layout
            "param_names": self.param_names,
            "param_slices": self.param_slices,
            
            # scaler (if exists)
            "scaler_mean": None if self.output_scaler is None else self.output_scaler.mean_,
            "scaler_scale": None if self.output_scaler is None else self.output_scaler.scale_,
        }

        # --------------------------
        # 3) SAVE
        # --------------------------
        torch.save(payload, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path):
        """
        Auto-loads either DeepONet or MLP based on checkpoint metadata.
        """
        import torch
        from sklearn.preprocessing import StandardScaler
        from models.framework import DeepONet, MLP

        # Important: allow full pickle loading (PyTorch 2.6!)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        class_name = checkpoint["class_name"]
        init_kwargs = checkpoint["init_kwargs"]

        # --------------------------------------------------
        # Instantiate correct class
        # --------------------------------------------------
        if class_name == "DeepONet":
            model = DeepONet(**init_kwargs)
        elif class_name == "MLP":
            model = MLP(**init_kwargs)
        else:
            raise ValueError(f"Unknown class_name '{class_name}' in checkpoint")

        # --------------------------------------------------
        # Restore weights
        # --------------------------------------------------
        model.load_state_dict(checkpoint["state_dict"])

        # --------------------------------------------------
        # Restore grid + param bounds
        # --------------------------------------------------
        model.set_grid(checkpoint["strikes"], checkpoint["maturities"])

        if checkpoint["param_bounds"] is not None:
            lb, ub = checkpoint["param_bounds"]
            model.set_param_bounds(lb, ub)

        # --------------------------------------------------
        # Restore parameter layout (if present)
        # --------------------------------------------------
        param_names  = checkpoint.get("param_names", None)
        param_slices = checkpoint.get("param_slices", None)
        if param_names is not None and param_slices is not None:
            model.set_param_structure(param_names, param_slices)


        # --------------------------------------------------
        # Restore scaler (if present)
        # --------------------------------------------------
        if checkpoint["scaler_mean"] is not None:
            scaler = StandardScaler()
            scaler.mean_  = checkpoint["scaler_mean"]
            scaler.scale_ = checkpoint["scaler_scale"]
            model.output_scaler = scaler

        print(f"Loaded {class_name} from {path}")
        return model





    def set_grid(self, strikes, maturities):
        self.strikes = np.array(strikes, dtype=np.float32)
        self.maturities = np.array(maturities, dtype=np.float32)

    def set_io_dims(self, input_dim=None, output_shape=None):
        if input_dim is not None:
            self.input_dim = int(input_dim)
        if output_shape is not None:
            assert len(output_shape) == 2, "output_shape must be (nT, nK)"
            self.output_shape = (int(output_shape[0]), int(output_shape[1]))


    # -----------------------------------------------------------
    # Parameter-Layout / Mapping (modellagnostisch)
    # -----------------------------------------------------------
    def set_param_structure(self, param_names, param_slices):
        """
        Speichert die vollständige Param-Layout-Struktur am Modell.
        param_names : list[str] – geflattete Namen (für Plots etc.)
        param_slices: dict[str, slice|int] – Mapping von originalen Keys zu Positionen.
        """
        self.param_names = list(param_names)
        self.param_slices = dict(param_slices)

    def _params_dict_to_vec_np(self, params_dict):
        """
        Mappe params_dict -> np.float32 Vektor gemäß self.param_names / self.param_slices.
        Erwartet, dass self.param_slices alle Keys aus params_dict abdeckt, die wir benutzen.
        """
        import numpy as np
        assert self.param_names is not None and self.param_slices is not None, \
            "param_names / param_slices nicht gesetzt. Ruf infer_param_structure_from_surfaces + set_param_structure auf."

        vec = np.zeros(len(self.param_names), dtype=np.float32)

        for key, sl in self.param_slices.items():
            if key not in params_dict:
                raise KeyError(f"Missing parameter '{key}' in params dict.")
            v = params_dict[key]
            if isinstance(sl, slice):
                vec[sl] = np.asarray(v, dtype=np.float32).ravel()
            else:
                vec[sl] = float(v)

        return vec

    def _params_dict_to_vec_tensor(self, params_dict, requires_grad=False):
        """
        Wie _params_dict_to_vec_np, aber als 1D torch.Tensor auf self.device.
        """
        import torch
        assert self.param_names is not None and self.param_slices is not None, \
            "param_names / param_slices nicht gesetzt. Ruf infer_param_structure_from_surfaces + set_param_structure auf."

        device = self.device
        vec = torch.zeros(len(self.param_names), dtype=torch.float32, device=device)

        for key, sl in self.param_slices.items():
            if key not in params_dict:
                raise KeyError(f"Missing parameter '{key}' in params dict.")
            v = params_dict[key]
            if isinstance(v, torch.Tensor):
                v_t = v.to(device=device, dtype=torch.float32)
            else:
                v_t = torch.tensor(v, dtype=torch.float32, device=device)

            if isinstance(sl, slice):
                vec[sl] = v_t.flatten()
            else:
                vec[sl] = v_t.flatten()[0]

        if requires_grad:
            vec.requires_grad_(True)
        return vec

    def params_dict_to_vec(self, p):
        """Komfort-Wrapper auf _params_dict_to_vec_np."""
        return self._params_dict_to_vec_np(p)

    def params_vec_to_dict(self, vec):
        """
        Inverse Mapping: Vektor -> dict mit denselben Keys wie self.param_slices.
        """
        import numpy as np
        assert self.param_slices is not None, "param_slices nicht gesetzt."
        vec = np.asarray(vec, dtype=np.float32)
        out = {}
        for key, sl in self.param_slices.items():
            if isinstance(sl, slice):
                out[key] = vec[sl].tolist()
            else:
                out[key] = float(vec[sl])
        return out

    
    @staticmethod
    def infer_param_structure_from_surfaces(surfaces):
        """
        Detects full parameter structure from surfaces[0]["params"].
        Produces:
            param_names: list of flattened parameter names in order
            param_slices: dict mapping key -> slice or index
        Works with:
        - any number of scalar params
        - any number of vector params
        - any shapes of vectors (flattened)
        """
        import numpy as np

        first = surfaces[0]["params"]

        param_names = []
        param_slices = {}

        pos = 0
        for key, val in first.items():
            # vector parameter (list/np array/tuple)
            if isinstance(val, (list, tuple, np.ndarray)):
                length = len(val)
                param_slices[key] = slice(pos, pos + length)
                for i in range(length):
                    param_names.append(f"{key}_{i}")
                pos += length

            # scalar parameter
            else:
                param_slices[key] = pos
                param_names.append(key)
                pos += 1

        return param_names, param_slices





    def compute_loss(self, y_pred, y_true):
        return nn.MSELoss()(y_pred, y_true)

    # -------------------- Param scaling: bounds → [-1, 1] --------------------

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
    
    def _interpolate_market_to_model(self, market_surface, Ks_market, Ts_market):
        """
        Bilinear interpolation: market IVs -> model-grid IVs.
        Everything stays in torch (differentiable but no gradients needed here).
        """
        device = self.device

        # model grid (log-space)
        base_Ts = torch.exp(
            torch.as_tensor(self.maturities, dtype=torch.float32, device=device)
        )
        base_Ks = torch.exp(
            torch.as_tensor(self.strikes, dtype=torch.float32, device=device)
        )

        # market grid
        mTs = torch.exp(torch.as_tensor(Ts_market, dtype=torch.float32, device=device))
        mKs = torch.exp(torch.as_tensor(Ks_market, dtype=torch.float32, device=device))
        mIV = torch.as_tensor(market_surface, dtype=torch.float32, device=device)

        T_min, T_max = mTs.min(), mTs.max()
        K_min, K_max = mKs.min(), mKs.max()

        TT, KK = torch.meshgrid(base_Ts, base_Ks, indexing="ij")
        TTn = 2 * (TT - T_min) / (T_max - T_min) - 1
        KKn = 2 * (KK - K_min) / (K_max - K_min) - 1

        grid_torch = torch.stack([KKn, TTn], dim=-1).unsqueeze(0)
        img = mIV.unsqueeze(0).unsqueeze(0)

        interp = torch.nn.functional.grid_sample(
            img,
            grid_torch,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(0).squeeze(0)

        return interp.detach().cpu().numpy()   # calibration needs NumPy

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

        title = "Predicted Surface"

        if isinstance(params, dict) and self.param_names is not None:
            parts = []
            for name in self.param_names:
                if "_" in name:
                    base, idx = name.rsplit("_", 1)
                    idx = int(idx)
                    if base in params:
                        try:
                            parts.append(f"{name}={float(params[base][idx]):.3g}")
                        except Exception:
                            pass
                else:
                    if name in params:
                        try:
                            parts.append(f"{name}={float(params[name]):.3g}")
                        except Exception:
                            pass

            # Optional: nur ein paar Parameter anzeigen, sonst wird Titel zu lang
            if len(parts) > 6:
                parts = parts[:6] + ["..."]

            title += " (" + ", ".join(parts) + ")"

        # --- Difference ---
        c2 = axs[2].contourf(
            K,
            T,
            diff,
            levels=levels,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
        )
        axs[2].set_title(f"Difference (Pred - True)\nMAE={mae:.3e}, RMSE={rmse:.3e}")
        axs[2].set_xlabel("Strike")
        axs[2].set_ylabel("Maturity")
        plt.colorbar(c2, ax=axs[2])

        print(f"MAE = {mae:.6f}, RMSE = {rmse:.6f}")
        return fig

        
    def evaluate(self, surface_samples, out_dir):
        """
        Model-agnostic evaluation:
        - RMSE per surface
        - Error heatmaps
        - MC error heatmaps (if provided)
        - RMSE ECDF
        - RMSE vs parameter scatterplots (for ALL param_names)
        - Prediction latency stats (avg/median/p95)
        """
        assert self.strikes is not None and self.maturities is not None, \
            "Model grid (strikes/maturities) not set; call set_grid first."
        os.makedirs(out_dir, exist_ok=True)

        nT, nK = len(self.maturities), len(self.strikes)

        if self.param_names is not None:
            param_names = list(self.param_names)
        else:
            param_names = [f"theta_{i}" for i in range(len(surface_samples[0]["params"]))]

        param_values = {name: [] for name in param_names}

        bin_errs_rel = [[[] for _ in range(nK)] for _ in range(nT)]
        bin_errs_abs = [[[] for _ in range(nK)] for _ in range(nT)]
        bin_errs_mc  = [[[] for _ in range(nK)] for _ in range(nT)]
        bin_sqerr    = [[[] for _ in range(nK)] for _ in range(nT)]

        rmses = []

        # -------------------------
        # Prediction timing tracking
        # -------------------------
        pred_times_ms = []  # per-surface prediction latency in milliseconds

        # Helper: sync for accurate GPU timings
        def _sync_if_cuda():
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass

        # ==========================================================
        # Loop über alle Surfaces
        # ==========================================================
        for s in surface_samples:
            params = s["params"]

            true_vec = self._params_dict_to_vec_np(params)
            for name, val in zip(param_names, true_vec):
                param_values[name].append(float(val))

            true_surface = np.asarray(s["iv_surface"], dtype=np.float32)

            grid = s.get("grid", {"strikes": self.strikes, "maturities": self.maturities})
            Ks = np.asarray(grid["strikes"], dtype=np.float32)
            Ts = np.asarray(grid["maturities"], dtype=np.float32)

            # PRED (timed)
            _sync_if_cuda()
            t0 = time.perf_counter()
            with torch.inference_mode():
                pred_surface = self.predict_surface(params, grid=grid)
            _sync_if_cuda()
            t1 = time.perf_counter()

            pred_times_ms.append((t1 - t0) * 1e3)

            pred_surface = pred_surface.detach().cpu().numpy()

            rel_err = np.abs(true_surface - pred_surface) / np.clip(true_surface, 1e-6, None) * 100.0
            abs_err = np.abs(true_surface - pred_surface)
            sq_err  = (true_surface - pred_surface) ** 2

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

        rmses = np.array(rmses, dtype=np.float32)
        pred_times_ms = np.array(pred_times_ms, dtype=np.float64)

        # -------------------------
        # Prediction timing summary
        # -------------------------
        pred_time_mean_ms = float(np.mean(pred_times_ms)) if len(pred_times_ms) else float("nan")
        pred_time_median_ms = float(np.median(pred_times_ms)) if len(pred_times_ms) else float("nan")
        pred_time_p95_ms = float(np.percentile(pred_times_ms, 95)) if len(pred_times_ms) else float("nan")
        pred_time_min_ms = float(np.min(pred_times_ms)) if len(pred_times_ms) else float("nan")
        pred_time_max_ms = float(np.max(pred_times_ms)) if len(pred_times_ms) else float("nan")

        print("\nPrediction latency (ms) per surface:")
        print(f"  mean={pred_time_mean_ms:.3f}, median={pred_time_median_ms:.3f}, p95={pred_time_p95_ms:.3f}, "
            f"min={pred_time_min_ms:.3f}, max={pred_time_max_ms:.3f}")

        # ===============================
        # Aggregation helper
        # ===============================
        def aggregate_bins(bin_errs):
            mean = np.full((nT, nK), np.nan, dtype=np.float32)
            median = np.full((nT, nK), np.nan, dtype=np.float32)
            maxv = np.full((nT, nK), np.nan, dtype=np.float32)

            for t in range(nT):
                for k in range(nK):
                    vals = bin_errs[t][k]
                    if vals:
                        vals = np.asarray(vals, dtype=np.float32)
                        mean[t, k]   = float(np.mean(vals))
                        median[t, k] = float(np.median(vals))
                        maxv[t, k]   = float(np.max(vals))

            # missing → global mean
            global_mean = float(np.nanmean(mean))
            for arr in (mean, median, maxv):
                arr[np.isnan(arr)] = global_mean
            return mean, median, maxv, global_mean

        # REL
        mean_rel, median_rel, max_rel, global_mean_rel = aggregate_bins(bin_errs_rel)
        # ABS
        mean_abs, median_abs, max_abs, global_mean_abs = aggregate_bins(bin_errs_abs)
        # MC
        mean_mc, median_mc, max_mc, global_mean_mc = aggregate_bins(bin_errs_mc)
        # RMSE
        mean_sqerr, median_sqerr, max_sqerr, _ = aggregate_bins(bin_sqerr)
        mean_rmse = np.sqrt(mean_sqerr)
        median_rmse = np.sqrt(median_sqerr)
        max_rmse = np.sqrt(max_sqerr)
        global_mean_rmse = float(np.nanmean(mean_rmse))

        # ===============================
        # Plotting helpers
        # ===============================
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

        # rel / abs / mc / rmse heatmaps
        plot_set(
            [mean_rel, median_rel, max_rel],
            ["Mean Rel Error (Pred)", "Median Rel Error (Pred)", "Max Rel Error (Pred)"],
            "iv_error_heatmaps_binned.png",
            "%"
        )

        plot_set(
            [mean_abs, median_abs, max_abs],
            ["Mean Abs Error (Pred)", "Median Abs Error (Pred)", "Max Abs Error (Pred)"],
            "iv_abs_error_heatmaps_binned.png",
            "abs(IV diff)"
        )

        plot_set(
            [mean_mc, median_mc, max_mc],
            ["Mean Rel Error (MC)", "Median Rel Error (MC)", "Max Rel Error (MC)"],
            "iv_mc_rel_error_heatmaps_binned.png",
            "%"
        )

        plot_set(
            [mean_rmse, median_rmse, max_rmse],
            ["Mean RMSE", "Median RMSE", "Max RMSE"],
            "iv_rmse_heatmaps_binned.png",
            "RMSE"
        )

        # ===============================
        # RMSE ECDF
        # ===============================
        fig, ax = plt.subplots(figsize=(6, 5))
        sorted_r = np.sort(rmses)
        p = np.linspace(0, 1, len(sorted_r))
        ax.plot(p, sorted_r, lw=2)
        ax.set_xlabel("Quantile")
        ax.set_ylabel("RMSE")
        ax.set_title("RMSE ECDF")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "iv_rmse_quantiles.png"), dpi=200)
        plt.close(fig)

        # ===============================
        # RMSE vs parameter (für alle!)
        # ===============================
        def scatter(x, y, name_x, fname):
            fig, ax = plt.subplots(figsize=(6,5))
            ax.scatter(x, y, s=12, alpha=0.7)

            # polynomial fit (2nd degree)
            x = np.asarray(x)
            y = np.asarray(y)
            idx = np.argsort(x)
            xs, ys = x[idx], y[idx]

            if len(xs) > 10:  # safety
                # Fit: y ≈ a x^2 + b x + c
                coeffs = np.polyfit(xs, ys, deg=2)
                poly = np.poly1d(coeffs)

                xs_fit = np.linspace(xs.min(), xs.max(), 300)
                ys_fit = poly(xs_fit)

                ax.plot(xs_fit, ys_fit, linewidth=2.5, color="red",
                        label="Quadratic Fit")

            ax.set_xlabel(name_x)
            ax.set_ylabel("RMSE")
            ax.set_title(f"RMSE vs {name_x}")
            ax.grid(True, alpha=0.3)
            ax.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, fname), dpi=200)
            plt.close(fig)

        # Für alle param_names plotten
        for name in param_names:
            scatter(param_values[name], rmses, name, f"rmse_vs_{name}.png")


        mean_rmse_surface = float(np.mean(rmses))
        median_rmse_surface = float(np.median(rmses))

        print("\nRMSE summary over surfaces:")
        print(f"  Mean RMSE   = {mean_rmse_surface:.6f}")
        print(f"  Median RMSE = {median_rmse_surface:.6f}")

        # ===============================
        # Worst 10 Surfaces
        # ===============================
        worst_idx = np.argsort(rmses)[-10:][::-1]
        print("\nWorst 10 surfaces by RMSE:")
        for rank, idx in enumerate(worst_idx, 1):
            print(f"{rank:2d}. index={idx}, RMSE={rmses[idx]:.6f}")

        # ===============================
        # Rückgabe
        # ===============================
        return {
            "pred_rel": {"mean": mean_rel, "median": median_rel, "max": max_rel},
            "pred_abs": {"mean": mean_abs, "median": median_abs, "max": max_abs},
            "mc_rel":  {"mean": mean_mc, "median": median_mc, "max": max_mc},
            "rmse":    {"mean": mean_rmse, "median": median_rmse, "max": max_rmse},
            "rmses": rmses,
            "worst_indices": worst_idx.tolist(),
            "parameters": param_values,
            "prediction_time_ms": {
                "per_surface": pred_times_ms,   # numpy array
                "mean": pred_time_mean_ms,
                "median": pred_time_median_ms,
                "p95": pred_time_p95_ms,
                "min": pred_time_min_ms,
                "max": pred_time_max_ms,
            },
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

    @staticmethod
    def _heston_iv_surface_quantlib(
        x_phys,
        Ks_log,
        Ts_log,
        S0,
        r=0.0,
        q=0.0,
        mask_2d=None,
        dtype=np.float32,
    ):
        """
        Erzeuge eine Heston-IV-Surface via QuantLib AnalyticHestonEngine.

        x_phys: [kappa, theta, v0, sigma, rho]
        Ks_log: 1D array (log(K/S0) oder log(K)-ähnlich)
        Ts_log: 1D array (log(T))
        S0:     Spot
        mask_2d: optional bool-Array (nT, nK) – nur True-Punkte werden bepreist
        """

        kappa, theta, v0, sigma, rho = [float(v) for v in x_phys]

        day_counter = ql.Actual365Fixed()
        calendar = ql.NullCalendar()

        today = ql.Date(1, ql.January, 2000)
        ql.Settings.instance().evaluationDate = today

        spot = ql.QuoteHandle(ql.SimpleQuote(float(S0)))
        rTS = ql.YieldTermStructureHandle(
            ql.FlatForward(today, float(r), day_counter)
        )
        qTS = ql.YieldTermStructureHandle(
            ql.FlatForward(today, float(q), day_counter)
        )

        process = ql.HestonProcess(
            rTS, qTS, spot, float(v0), float(kappa), float(theta), float(sigma), float(rho)
        )
        model = ql.HestonModel(process)
        engine = ql.AnalyticHestonEngine(model)

        # log -> linear
        Ks_pricing = np.exp(np.asarray(Ks_log, dtype=np.float64)) * float(S0)
        Ts_pricing = np.exp(np.asarray(Ts_log, dtype=np.float64))

        nT, nK = len(Ts_pricing), len(Ks_pricing)
        iv_surf = np.full((nT, nK), np.nan, dtype=dtype)

        if mask_2d is None:
            mask_2d = np.ones((nT, nK), dtype=bool)
        else:
            mask_2d = np.asarray(mask_2d, dtype=bool)
            assert mask_2d.shape == (nT, nK)

        for ti, T in enumerate(Ts_pricing):
            # wenn in dieser Zeile kein Punkt gebraucht wird, skip
            if not mask_2d[ti, :].any():
                continue

            T_float = float(T)
            if T_float <= 0.0:
                continue

            days = max(1, int(round(T_float * 365)))
            maturity_date = today + days
            exercise = ql.EuropeanExercise(maturity_date)

            for ki, K in enumerate(Ks_pricing):
                if not mask_2d[ti, ki]:
                    continue

                K_float = float(K)
                payoff_type = ql.Option.Call if K_float >= S0 else ql.Option.Put
                payoff = ql.PlainVanillaPayoff(payoff_type, K_float)
                option = ql.VanillaOption(payoff, exercise)
                option.setPricingEngine(engine)

                price = option.NPV()
                try:
                    if payoff_type == ql.Option.Call:
                        iv = bsinv(price, S0, K_float, T_float, o="call")
                    else:
                        iv = bsinv(price, S0, K_float, T_float, o="put")
                except Exception:
                    iv = np.nan

                iv_surf[ti, ki] = dtype(iv)

        return iv_surf

    def calibrate_heston_analytic(
        self,
        target_surface,
        optimiser="lm",
        maxiter=500,
        verbose=False,
    ):
        """
        Analytische Heston-Kalibrierung via QuantLib (AnalyticHestonEngine).

        Gleiche Output-Form wie self.calibrate:
            - "theta_hat"
            - "error_rel_dict"
            - "rmse"
            - "runtime_ms"
            - "optimizer"
        (plus ggf. zusätzliche Felder, falls du willst)
        """
        import time
        from scipy.optimize import least_squares, differential_evolution, minimize

        assert self.param_bounds is not None, "Call set_param_bounds() first."

        # --- Bounds ---
        lb, ub = self.param_bounds
        lb = np.asarray(lb, dtype=np.float64)
        ub = np.asarray(ub, dtype=np.float64)

        # --- Ziel-Surface + Grid (log-space) ---
        true_surface = np.asarray(target_surface["iv_surface"], dtype=np.float64)  # (nT, nK)
        Ks_log = np.asarray(target_surface["grid"]["strikes"], dtype=np.float64)
        Ts_log = np.asarray(target_surface["grid"]["maturities"], dtype=np.float64)

        nT, nK = true_surface.shape

        S0 = float(getattr(self, "S0", target_surface.get("S0", 1.0)))
        r = float(getattr(self, "r", target_surface.get("r", 0.0)))
        q = float(getattr(self, "q", target_surface.get("q", 0.0)))

        true_params = target_surface.get("params", None)

        # --- Mask ---
        mask_np = np.isfinite(true_surface)  # (nT, nK)
        mask_flat = mask_np.reshape(-1)
        true_flat = true_surface.reshape(-1)
        true_masked = true_flat[mask_flat]

        # --- Weights optional ---
        weights = target_surface.get("weights", None)
        if weights is not None:
            weights_full = np.asarray(weights, dtype=np.float64)
            weights_flat = weights_full.reshape(-1)
            weights_masked = weights_flat[mask_flat]
        else:
            weights_full = None
            weights_masked = None

        # --- Startwert ---
        if true_params is not None:
            param_names = getattr(self, "param_names", ["kappa", "theta", "v0", "sigma", "rho"])
            guess_list = []
            ok = True
            for name in param_names:
                if name in true_params:
                    guess_list.append(float(true_params[name]))
                else:
                    ok = False
                    break
            if ok and len(guess_list) == len(lb):
                x0 = np.array(guess_list, dtype=np.float64)
            else:
                x0 = 0.5 * (lb + ub)
        else:
            x0 = 0.5 * (lb + ub)

        # --- Residuals ---
        def residuals(x_phys):
            iv_pred = self._heston_iv_surface_quantlib(
                x_phys,
                Ks_log=Ks_log,
                Ts_log=Ts_log,
                S0=S0,
                r=r,
                q=q,
                mask_2d=mask_np,  # nur relevante Punkte berechnen
                dtype=np.float64,
            )
            pred_flat = iv_pred.reshape(-1)
            diff = pred_flat[mask_flat] - true_masked
            if weights_masked is not None:
                diff = diff * weights_masked
            diff = np.where(np.isfinite(diff), diff, 0.0)
            return diff

        def rmse_objective(x_phys):
            iv_pred = self._heston_iv_surface_quantlib(
                x_phys,
                Ks_log=Ks_log,
                Ts_log=Ts_log,
                S0=S0,
                r=r,
                q=q,
                mask_2d=mask_np,
                dtype=np.float64,
            )

            pred_flat = iv_pred.reshape(-1)
            diff = pred_flat[mask_flat] - true_masked

            if weights_masked is not None:
                w = weights_masked ** 2     # <- hier brauchst du das echte w, NICHT sqrt(w)
                sq = w * (diff ** 2)
                val = np.sqrt(np.sum(sq) / np.sum(w))
            else:
                val = np.sqrt(np.mean(diff**2))
            if not np.isfinite(val):
                return 1e6

            return val

        t0 = time.perf_counter()
        opt_lower = optimiser.lower()

        # --- Optimizer Auswahl ---
        if opt_lower in ["differential evolution", "de"]:
            res = differential_evolution(
                rmse_objective,
                bounds=list(zip(lb, ub)),
                maxiter=maxiter,
                disp=verbose
            )

        elif opt_lower in ["levenberg-marquardt", "lm", "trf", "dogbox"]:
            # least_squares – ohne expliziten Jacobian
            method = "trf" if opt_lower in ["levenberg-marquardt", "lm"] else opt_lower
            res = least_squares(
                residuals,
                x0,
                method=method,
                bounds=(lb, ub),
                max_nfev=maxiter,
                verbose=2 if verbose else 0
            )

        else:
            res = minimize(
                rmse_objective,
                x0,
                method=optimiser,
                bounds=list(zip(lb, ub)),
                options={"maxiter": maxiter, "disp": verbose}
            )

        t1 = time.perf_counter()

        theta_hat = res.x
        runtime_ms = (t1 - t0) * 1000.0
        rmse = rmse_objective(theta_hat)

        # --- Fehlerauswertung wie in calibrate ---
        if self.param_names is not None:
            param_names = list(self.param_names)
        else:
            param_names = [f"theta_{i}" for i in range(len(theta_hat))]

        if true_params is not None:
            true_vec = self._params_dict_to_vec_np(true_params)
            abs_err = np.abs(theta_hat - true_vec)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_err = np.where(true_vec != 0, abs_err / np.abs(true_vec), np.nan)
            error_rel_dict = {
                name: float(err) for name, err in zip(param_names, rel_err)
            }
        else:
            error_rel_dict = None

        if verbose:
            print("Heston analytic calibration finished.")
            print("theta_hat:", theta_hat)
            print("RMSE:", rmse)
            print("runtime_ms:", runtime_ms)

        return {
            "theta_hat":      theta_hat,
            "error_rel_dict": error_rel_dict,
            "rmse":           float(rmse),
            "runtime_ms":     float(runtime_ms),
            "optimizer":      optimiser,
        }



    def residuals_autograd(
        self,
        x_phys,
        true_surface,
        Ks,
        Ts,
        weights_full=None,
        mask_flat=None,
    ):
        """
        Compute residuals and Jacobian wrt param vector x_phys using autograd.

        x_phys: 1D numpy array (physische Parameter, gleiche Reihenfolge wie param_bounds)
        """
        import torch

        device = self.device

        true_np = np.asarray(true_surface, dtype=np.float32)
        nT, nK = true_np.shape
        true_flat = true_np.reshape(-1)

        if mask_flat is not None:
            mask_flat = np.asarray(mask_flat, dtype=bool).reshape(-1)
            true_masked = true_flat[mask_flat]
        else:
            true_masked = true_flat

        if weights_full is not None:
            w_flat = np.asarray(weights_full, dtype=np.float32).reshape(-1)
            weights_masked = w_flat[mask_flat] if mask_flat is not None else w_flat
        else:
            weights_masked = None

        Ks_t = torch.as_tensor(Ks, dtype=torch.float32, device=device)
        Ts_t = torch.as_tensor(Ts, dtype=torch.float32, device=device)

        def f_single(x_tensor):
            # x_tensor: 1D tensor, physical param vector
            pred = self.predict_surface(
                x_tensor,
                grid={"strikes": Ks_t, "maturities": Ts_t},
                mask_flat=mask_flat,
            )  # (n_valid,) or (nT, nK)
            pred_flat = pred.reshape(-1)
            diff = pred_flat[mask_flat] - torch.as_tensor(true_masked, dtype=torch.float32, device=device)
            if weights_masked is not None:
                w = torch.as_tensor(weights_masked, dtype=torch.float32, device=device)
                diff = diff * w
            return diff

        x0_t = torch.tensor(x_phys, dtype=torch.float32, device=device, requires_grad=True)
        res_t = f_single(x0_t)
        J_fn = jacrev(f_single)
        J_t = J_fn(x0_t)  # (n_res, n_params)

        r_np = res_t.detach().cpu().numpy()
        J_np = J_t.detach().cpu().numpy()
        return r_np, J_np




    def calibrate(self, target_surface, optimiser="lm", bounds=None, maxiter=500, verbose=False):
        """
        Calibrate model parameters θ̂ to a given implied-volatility surface by minimizing
        a masked (optionally weighted) L2-error (RMSE up to normalization).

        - Ignores NaNs in `iv_surface`.
        - Supports optional root-`weights` on the same grid (e.g. sqrt(volume+1)).
        """
        import time
        import numpy as np
        from scipy.optimize import minimize, least_squares, differential_evolution

        assert self.param_bounds is not None, "Call set_param_bounds() first."
        lb, ub = self.param_bounds if bounds is None else np.array(list(zip(*bounds)))
        lb, ub = np.asarray(lb, dtype=np.float32), np.asarray(ub, dtype=np.float32)

        # --- Load grid + surface ---
        true_surface = np.asarray(target_surface["iv_surface"], dtype=np.float32)  # (nT, nK)
        Ks = np.asarray(target_surface["grid"]["strikes"], dtype=np.float32)
        Ts = np.asarray(target_surface["grid"]["maturities"], dtype=np.float32)

        true_params = target_surface.get("params", None)

        # --------------------------------------------------------------
        # Mask for valid (finite) data points
        # --------------------------------------------------------------
        mask_np   = ~np.isnan(true_surface)          # (nT, nK) bool
        mask_flat = mask_np.reshape(-1)              # (nT*nK,) bool

        true_flat   = true_surface.reshape(-1)
        true_masked = true_flat[mask_flat]          # (n_valid,)

        # --------------------------------------------------------------
        # Optional weights (must align with surface flattening)
        # --------------------------------------------------------------
        weights = target_surface.get("weights", None)
        if weights is not None:
            weights_full = np.asarray(weights, dtype=np.float32)      # (nT, nK)
            weights_flat = weights_full.reshape(-1)                   # (nT*nK,)
            weights_masked = weights_flat[mask_flat]                  # (n_valid,)
        else:
            weights_full = None
            weights_masked = None

        # --------------------------------------------------------------
        # RMSE-like objective with masking + optional weighting
        # --------------------------------------------------------------
        def rmse_objective(x_phys):
            # x_phys: physical parameter vector
            pred = self.predict_surface(
                x_phys,
                grid={"strikes": Ks, "maturities": Ts},
                mask_flat=mask_flat,
            )
            pred = pred.detach().cpu().numpy()  # (nT, nK)

            pred_flat = pred.reshape(-1)
            diff = pred_flat[mask_flat] - true_masked

            if weights_masked is not None:
                w = weights_masked ** 2     # <- hier brauchst du das echte w, NICHT sqrt(w)
                sq = w * (diff ** 2)
                val = np.sqrt(np.sum(sq) / np.sum(w))
            else:
                val = np.sqrt(np.mean(diff**2))
            if not np.isfinite(val):
                return 1e6
            return val

        # --------------------------------------------------------------
        # Initial guess
        # --------------------------------------------------------------
        x0 = 0.5 * (lb + ub)
        t0 = time.perf_counter()

        # --------------------------------------------------------------
        # Optimizer selection
        # --------------------------------------------------------------
        opt_lower = optimiser.lower()

        if opt_lower == "differential evolution":
            res = differential_evolution(
                rmse_objective,
                bounds=list(zip(lb, ub)),
                maxiter=maxiter,
                disp=verbose
            )

        elif opt_lower in ["levenberg-marquardt", "lm"]:
            # least_squares expects residuals + Jacobian
            def fun_ls(x):
                r_np, _ = self.residuals_autograd(
                    x,
                    true_surface,     # full 2D (nT, nK), masking inside residuals_autograd
                    Ks,
                    Ts,
                    weights_full,
                    mask_flat      # full 2D or None
                )
                return r_np

            def jac_ls(x):
                _, J_np = self.residuals_autograd(
                    x,
                    true_surface,
                    Ks,
                    Ts,
                    weights_full,
                    mask_flat
                )
                return J_np

            res = least_squares(
                fun_ls,
                x0,
                jac=jac_ls,
                method="trf",
                bounds=(lb, ub),
                max_nfev=maxiter,
                verbose=2 if verbose else 0
            )

        else:
            res = minimize(
                rmse_objective,
                x0,
                method=optimiser,
                bounds=list(zip(lb, ub)),
                options={"maxiter": maxiter, "disp": verbose}
            )

        # --------------------------------------------------------------
        # Results
        # --------------------------------------------------------------
        t1 = time.perf_counter()
        theta_hat = res.x

        if self.param_names is not None:
            param_names = list(self.param_names)
        else:
            param_names = [f"theta_{i}" for i in range(len(theta_hat))]

        # Optionale Fehlerauswertung vs true_params (falls vorhanden)
        true_params = target_surface.get("params", None)
        if true_params is not None:
            true_vec = self._params_dict_to_vec_np(true_params)
            abs_err = np.abs(theta_hat - true_vec)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_err = np.where(true_vec != 0, abs_err / np.abs(true_vec), np.nan)
            error_rel_dict = {
                name: float(err) for name, err in zip(param_names, rel_err)
            }
        else:
            error_rel_dict = None

        runtime_ms = (t1 - t0) * 1000.0

        rmse = rmse_objective(theta_hat)

        return {
            "theta_hat":      theta_hat,
            "error_rel_dict": error_rel_dict,
            "rmse":           float(rmse),
            "runtime_ms":     runtime_ms,
            "optimizer":      optimiser,
        }




    def evaluate_calibrate(
        self,
        surfaces,
        optimiser="lm",
        maxiter=500,
        verbose=False,
        analytic=False
    ):
        print(f"\nEvaluating calibration using {optimiser} on {len(surfaces)} surfaces...")

        runtimes, rmses = [], []
        per_param_rel_errors, per_param_abs_errors = {}, {}
        true_params_all, est_params_all = [], []

        # globale Param-Reihenfolge des Modells
        if self.param_names is not None:
            param_names = list(self.param_names)
        else:
            # fallback
            param_names = [f"theta_{j}" for j in range(len(surfaces[0]["params"]))]

        for i, s in enumerate(surfaces, start=1):
            if analytic:
                r = self.calibrate_heston_analytic(
                    s,
                    optimiser=optimiser,
                    maxiter=maxiter,
                    verbose=verbose,
            )            
            else:
                r = self.calibrate(s, optimiser=optimiser, maxiter=maxiter, verbose=verbose)
            # -------------------------------------------------------------
            # Store results
            # -------------------------------------------------------------
            runtimes.append(r["runtime_ms"])
            rmses.append(r["rmse"])
            theta_hat = np.asarray(r["theta_hat"], dtype=np.float32)
            est_params_all.append(theta_hat)

            # ---- true phys. Parameter-Vektor (modellagnostisch)
            tp = s.get("params", None)
            if tp is not None:
                true_vec = self._params_dict_to_vec_np(tp)
            else:
                true_vec = np.full_like(theta_hat, np.nan, dtype=np.float32)

            true_params_all.append(true_vec)

            # Fehler
            abs_errs = np.abs(theta_hat - true_vec)
            rel_errs = abs_errs / np.clip(np.abs(true_vec), 1e-8, None)

            for k, aerr, rerr in zip(param_names, abs_errs, rel_errs):
                per_param_abs_errors.setdefault(k, []).append(float(aerr))
                per_param_rel_errors.setdefault(k, []).append(float(rerr))

        # -------------------------------------------------------------
        # Convert lists to arrays
        # -------------------------------------------------------------
        for d in (per_param_abs_errors, per_param_rel_errors):
            for k in d:
                d[k] = np.array(d[k], dtype=np.float32)

        runtimes = np.array(runtimes, dtype=np.float32)
        rmses = np.array(rmses, dtype=np.float32)
        true_params_all = np.array(true_params_all, dtype=np.float32)
        est_params_all = np.array(est_params_all, dtype=np.float32)
        avg_time = float(np.mean(runtimes))

        # -------------------------------------------------------------
        # Summary helper
        # -------------------------------------------------------------
        def summarize(errors, kind):
            print(f"{kind.title()} Errors per Parameter:")
            for k, vals in errors.items():
                scale = 100 if kind == "relative" else 1
                mean = float(np.mean(vals))
                median = float(np.median(vals))
                std = float(np.std(vals))
                q95 = float(np.quantile(vals, 0.95))
                q99 = float(np.quantile(vals, 0.99))
                unit = "%" if kind == "relative" else ""

                if kind == "absolute":
                    rmse = float(np.sqrt(np.mean(vals**2)))
                    print(
                        f"   {k:<10s} | mean={mean*scale:.3f}{unit}"
                        f" | median={median*scale:.3f}{unit}"
                        f" | std={std*scale:.3f}{unit}"
                        f" | q95={q95*scale:.3f}{unit}"
                        f" | q99={q99*scale:.3f}{unit}"
                        f" | RMSE={rmse*scale:.3f}{unit}"
                    )
                else:
                    print(
                        f"   {k:<10s} | mean={mean*scale:.3f}{unit}"
                        f" | median={median*scale:.3f}{unit}"
                        f" | std={std*scale:.3f}{unit}"
                        f" | q95={q95*scale:.3f}{unit}"
                        f" | q99={q99*scale:.3f}{unit}"
                    )

        # -------------------------------------------------------------
        # Print summary
        # -------------------------------------------------------------
        print(f"\n→ Final avg time: {avg_time:.1f} ms, mean RMSE={np.mean(rmses):.5f}\n")
        summarize(per_param_rel_errors, "relative")
        print()
        summarize(per_param_abs_errors, "absolute")

        # -------------------------------------------------------------
        # Top-5 errors per parameter
        # -------------------------------------------------------------
        print("\nTop-5 absolute & relative errors per parameter:")
        for k in param_names:
            abs_vals = per_param_abs_errors[k]
            rel_vals = per_param_rel_errors[k]

            top5_abs_idx = np.argsort(abs_vals)[-5:][::-1]
            top5_abs_vals = abs_vals[top5_abs_idx]

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

        per_param_abs_rmse = {
            k: float(np.sqrt(np.mean(v**2))) for k, v in per_param_abs_errors.items()
        }

        return {
            "optimizer": optimiser,
            "avg_time_ms": avg_time,
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
            return torch.sum(b * t, dim=1, keepdim=True).squeeze(-1)

        elif self.mask_type == "spatial":
            mask = self.mask_net(xt)  # (batch, 1)
            return (torch.sum(b * t, dim=1, keepdim=True) * mask).squeeze(-1)

        elif self.mask_type == "channel":
            mask = self.mask_net(xt)  # (batch, latent_dim)
            return torch.sum(b * t * mask, dim=1, keepdim=True).squeeze(-1)

        elif self.mask_type == "contextual":
            mask_input = torch.cat([b, t], dim=1)
            mask = self.mask_net(mask_input)  # (batch, latent_dim)
            return torch.sum(b * t * mask, dim=1, keepdim=True).squeeze(-1)

    # --------------------------------------------------------

    @staticmethod
    def _flatten_surfaces_for_deeponet(
        surfaces,
        param_names,
        param_slices,
    ):
        """
        Flatten surfaces using param_names/param_slices.
        Dies ist die finale, model-agnostische Parameter-Flattung.

        Rückgabe:
            X_branch : (N, d)
            X_trunk  : (N, 2)
            Y        : (N,)
            strikes  : 1D
            maturities: 1D
        """
        import numpy as np

        Xb_list, Xt_list, Y_list = [], [], []
        Ks_ref, Ts_ref = None, None

        for surf in surfaces:
            params = surf["params"]
            iv_surface = np.asarray(surf["iv_surface"], dtype=np.float32)
            Ks = np.asarray(surf["grid"]["strikes"], dtype=np.float32)
            Ts = np.asarray(surf["grid"]["maturities"], dtype=np.float32)

            # Build parameter vector via param_slices
            vec = np.zeros(len(param_names), dtype=np.float32)
            for key, sl in param_slices.items():
                v = params[key]
                if isinstance(sl, slice):
                    vec[sl] = np.asarray(v, dtype=np.float32).flatten()
                else:
                    vec[sl] = float(v)

            KK, TT = np.meshgrid(Ks, Ts, indexing="xy")
            coords = np.stack([KK.reshape(-1), TT.reshape(-1)], axis=1).astype(np.float32)

            Xb_list.append(np.repeat(vec[None, :], len(coords), axis=0))
            Xt_list.append(coords)
            Y_list.append(iv_surface.reshape(-1))

            Ks_ref, Ts_ref = Ks, Ts   # letzter Grid reicht

        X_branch = np.concatenate(Xb_list, axis=0)
        X_trunk  = np.concatenate(Xt_list, axis=0)
        Y        = np.concatenate(Y_list, axis=0)

        return X_branch, X_trunk, Y, Ks_ref, Ts_ref




    @classmethod
    def from_surfaces(
        cls,
        surfaces,
        batch_size=256,
        val_split=0.2,
        shuffle=True,
        branch_hidden_dims=(256, 256),
        trunk_hidden_dims=(256, 256),
        latent_dim=128,
        ref_strikes=None,
        ref_maturities=None,
        activation="gelu",
        lr=1e-3,
        mask_type="none",
        shuffle_training_batches=True,
    ):

        """
        Factory: build DeepONet + loaders from a list of surfaces.

        """
        # --- Flatten to per-point data ---
    # ------------------------------------------------------------
    # 0) AUTO-INFER PARAMETER STRUCTURE IF NOT PROVIDED
    # ------------------------------------------------------------
        # ------------------------------------------------------------
        # 0) PARAM-STRUKTUR AUTOMATISCH AUS ERSTEM SURFACE INFERIEREN
        # ------------------------------------------------------------
        param_names, param_slices = BaseModel.infer_param_structure_from_surfaces(surfaces)

        # ------------------------------------------------------------
        # 1) Flatten Surfaces (per-point)
        # ------------------------------------------------------------
        X_branch, X_trunk, Y, strikes, maturities = cls._flatten_surfaces_for_deeponet(
            surfaces,
            param_names=param_names,
            param_slices=param_slices,
        )

        if ref_strikes is not None:
            strikes = np.asarray(ref_strikes, dtype=np.float32)
        if ref_maturities is not None:
            maturities = np.asarray(ref_maturities, dtype=np.float32)

        input_dim = X_branch.shape[1]
        
        # Empirical bounds for branch normalization
        lb = np.min(X_branch, axis=0)
        ub = np.max(X_branch, axis=0)
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
        model.set_param_structure(param_names, param_slices)
        model.set_param_bounds(lb, ub)

        Y_tr_scaled = model.transform_output(Y[tr])
        Y_va_scaled = model.transform_output(Y[va])

        train_ds = IVSurfaceDataset(Xb_scaled[tr], X_trunk[tr], Y_tr_scaled)
        val_ds = IVSurfaceDataset(Xb_scaled[va], X_trunk[va], Y_va_scaled)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_training_batches)
        val_loader = DataLoader(val_ds, batch_size=2 * batch_size, shuffle=False)

        return model, train_loader, val_loader


    def test_mask_response(self, xb_sample, xt_grid, visualize=True, channels=None):
        """
        Evaluate and optionally visualize individual latent mask channels
        for a fixed branch vector across a grid of (K,T) coordinates.
        """
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        import math

        self.eval()
        if getattr(self, "mask_net", None) is None:
            print("No mask network defined (mask_type='none').")
            return None

        # ----------------------------------------------------------
        # Prepare input tensors
        # ----------------------------------------------------------
        xb_sample = torch.tensor(xb_sample, dtype=torch.float32, device=self.device)
        xb_sample = xb_sample.unsqueeze(0).repeat(len(xt_grid), 1)
        xt = torch.tensor(xt_grid, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            b = self.branch(xb_sample)
            t = self.trunk(xt)

            if self.mask_type in {"spatial", "channel"}:
                mask = self.mask_net(xt)
            elif self.mask_type == "contextual":
                mask_input = torch.cat([b, t], dim=1)
                mask = self.mask_net(mask_input)
            else:
                mask = torch.ones(len(xt), self.latent_dim, device=self.device)

        mask_np = mask.detach().cpu().numpy()  # [N, latent_dim]

        # ----------------------------------------------------------
        # Grid reconstruction
        # ----------------------------------------------------------
        K_vals = np.unique(xt_grid[:, 0])
        T_vals = np.unique(xt_grid[:, 1])

        nK = len(K_vals)
        nT = len(T_vals)
        latent_dim = mask_np.shape[1]

        # ----------------------------------------------------------
        # Visualization
        # ----------------------------------------------------------
        if visualize:
            # Clip mask values explicitly to [0, 1]
            mask_np = np.clip(mask_np, 0.0, 1.0)

            if channels is None:
                channels = list(range(latent_dim))
            else:
                channels = [c for c in channels if c < latent_dim]

            n_show = len(channels)
            n_cols = 4
            n_rows = math.ceil(n_show / n_cols)

            fig, axes = plt.subplots(
                n_rows, n_cols,
                figsize=(3.6 * n_cols, 3.2 * n_rows),
                squeeze=False
            )

            for i, c in enumerate(channels):
                r, col = divmod(i, n_cols)
                ax = axes[r, col]

                mask_c = mask_np[:, c].reshape(nT, nK)

                im = ax.imshow(
                    mask_c,
                    origin="lower",
                    aspect="auto",
                    cmap="magma",
                    vmin=0.0,
                    vmax=1.0,
                    extent=[
                        K_vals.min(), K_vals.max(),
                        T_vals.min(), T_vals.max()
                    ]
                )

                ax.set_title(f"Channel {c}", fontsize=10)
                ax.set_xlabel("Strike $K$")
                ax.set_ylabel("Maturity $T$")
                ax.invert_yaxis()

                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Turn off unused axes
            for j in range(n_show, n_rows * n_cols):
                r, col = divmod(j, n_cols)
                axes[r, col].axis("off")

            plt.tight_layout(rect=[0, 0, 1, 0.96])
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
    ###############################################
    #   Fully Torch-Based predict_surface()
    ###############################################
    def predict_surface(self, params, grid=None, mask_flat=None):
        """
        DeepONet prediction:
        - Always returns full (nT, nK) surface
        - If mask_flat is given:
            → compute ONLY masked points
            → fill output matrix with NaNs at unmasked positions

        """
        import torch

        device = self.device

        # ----- 1) Grid -----
        if grid is None:
            strikes = torch.as_tensor(self.strikes, dtype=torch.float32, device=device)
            maturities = torch.as_tensor(self.maturities, dtype=torch.float32, device=device)
        else:
            strikes = torch.as_tensor(grid["strikes"], dtype=torch.float32, device=device)
            maturities = torch.as_tensor(grid["maturities"], dtype=torch.float32, device=device)

        nK = strikes.numel()
        nT = maturities.numel()

        # ----- 2) Parametervektor -----
        if isinstance(params, dict):
            x_phys = self._params_dict_to_vec_tensor(params)
        else:
            if isinstance(params, torch.Tensor):
                x_phys = params.to(device=device, dtype=torch.float32).flatten()
            else:
                x_phys = torch.as_tensor(params, dtype=torch.float32, device=device).flatten()

        lb = torch.as_tensor(self.param_bounds[0], dtype=torch.float32, device=device)
        ub = torch.as_tensor(self.param_bounds[1], dtype=torch.float32, device=device)

        x_scaled = 2.0 * (x_phys - lb) / (ub - lb) - 1.0

        # ----- 3) Trunk coords -----
        K_mesh, T_mesh = torch.meshgrid(strikes, maturities, indexing="xy")
        xt_full = torch.stack([K_mesh.reshape(-1), T_mesh.reshape(-1)], dim=1)   # (nK*nT, 2)

        if mask_flat is None:
            xt = xt_full
        else:
            mask_t = torch.as_tensor(mask_flat, dtype=torch.bool, device=device).reshape(-1)
            xt = xt_full[mask_t]     # only masked coords

        # ----- 4) Branch broadcast -----
        xb = x_scaled.unsqueeze(0).expand(xt.shape[0], -1)

        # ----- 5) Forward pass -----
        pred_scaled = self.forward(xb, xt).reshape(-1)

        # inverse scaling
        mean = torch.as_tensor(self.output_scaler.mean_[0], dtype=torch.float32, device=device)
        std  = torch.as_tensor(self.output_scaler.scale_[0], dtype=torch.float32, device=device)
        pred_iv = pred_scaled * std + mean

        # ----- 6) Reconstruct full (nT, nK) with NaNs -----
        out = torch.full((nT * nK,), float("nan"), dtype=torch.float32, device=device)

        if mask_flat is None:
            out[:] = pred_iv
        else:
            out[mask_t] = pred_iv

        return out.reshape(nT, nK)




import time
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset



class MLP(BaseModel):
    """
    Simple MLP mapping parameter vector -> implied volatility surface.
    Output is the full (nT, nK) surface predicted in one shot.

    interp_mode:
        "base"            -> keine Interpolation, immer Modellgrid zurückgeben
        "model_to_market" -> MLP auf Modellgrid, dann Modell->Market Grid Interpolation
        "market_to_model" -> Market-IVs auf Modellgrid interpolieren (für Daten/Analyse)
    """

    def __init__(
        self,
        input_dim=None,
        output_shape=None,
        hidden_dims=(256, 256, 256),
        activation="elu",
        lr=1e-3
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_shape = output_shape  # (nT, nK)
        self.output_dim = (
            None if output_shape is None else int(output_shape[0] * output_shape[1])
        )
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.lr = lr

        if (input_dim is not None) and (output_shape is not None):
            self._build_network()
        self.to(self.device)

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------
    def _build_network(self):
        act_fn = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
        }[self.activation]

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
        out = self.net(x)  # (batch, output_dim)
        batch = out.shape[0]
        return out.reshape(batch, *self.output_shape)

    # ------------------------------------------------------------------
    # Data stacking
    # ------------------------------------------------------------------
    @staticmethod
    def _stack_XY(surfaces, param_names, param_slices, sanity_check_grids=True):
        import numpy as np

        X_list, Y_list = [], []
        Ks_ref, Ts_ref = None, None

        for surf in surfaces:
            params = surf["params"]
            iv_surface = np.asarray(surf["iv_surface"], dtype=np.float32)
            Ks = np.asarray(surf["grid"]["strikes"], dtype=np.float32)
            Ts = np.asarray(surf["grid"]["maturities"], dtype=np.float32)

            if sanity_check_grids:
                if Ks_ref is None:
                    Ks_ref, Ts_ref = Ks, Ts
                else:
                    if not (np.allclose(Ks_ref, Ks) and np.allclose(Ts_ref, Ts)):
                        raise ValueError(
                            "All surfaces must share the same (K,T) grid for MLP output."
                        )

            # Paramvektor über param_slices
            vec = np.zeros(len(param_names), dtype=np.float32)
            for key, sl in param_slices.items():
                v = params[key]
                if isinstance(sl, slice):
                    vec[sl] = np.asarray(v, dtype=np.float32).flatten()
                else:
                    vec[sl] = float(v)

            X_list.append(vec)
            Y_list.append(iv_surface)

        X = np.stack(X_list, axis=0)
        Y = np.stack(Y_list, axis=0)

        strikes   = Ks_ref if Ks_ref is not None else Ks
        maturities = Ts_ref if Ts_ref is not None else Ts

        return X, Y, strikes, maturities




    # ------------------------------------------------------------------
    # Factory from_surfaces
    # ------------------------------------------------------------------
    @classmethod
    def from_surfaces(
        cls,
        surfaces,
        batch_size=32,
        val_split=0.2,
        shuffle=True,
        hidden_dims=(256, 256, 256),
        activation="gelu",
        lr=1e-3,
        shuffle_training_batches=True,
    ):

        """
        Build an MLP model + loaders with internal, leakage-safe scaling:
        - Param vector scaled to [-1, 1] using empirical min/max (+1% margin)
        - Output (IV) scaled via StandardScaler fit on train only

        Returns
        -------
        model, train_loader, val_loader, strikes, maturities
        """
        # --- PARAM-STRUKTUR aus erstem Surface ziehen ---
        param_names, param_slices = BaseModel.infer_param_structure_from_surfaces(surfaces)

        # --- Flatten all surfaces ---
        X, Y, strikes, maturities = cls._stack_XY(
            surfaces,
            param_names=param_names,
            param_slices=param_slices,
        )
        nT, nK = Y.shape[1:]
        input_dim = X.shape[1]
        output_shape = (nT, nK)

        # --- Empirical parameter bounds (+1% margin) ---
        lb = np.min(X, axis=0)
        ub = np.max(X, axis=0)

        X_scaled = BaseModel._scale_to_m1_p1(X, lb, ub)

        # Split
        n_total = len(Y)
        n_train = int((1 - val_split) * n_total)
        idx = np.arange(n_total)
        if shuffle:
            np.random.shuffle(idx)
        tr, va = idx[:n_train], idx[n_train:]

        # Build model
        model = cls(
            input_dim=input_dim,
            output_shape=output_shape,
            hidden_dims=hidden_dims,
            activation=activation,
            lr=lr,
        )
        model.set_grid(strikes, maturities)
        model.set_io_dims(input_dim=input_dim, output_shape=output_shape)
        model.set_param_bounds(lb, ub)
        model.set_param_structure(param_names, param_slices)
        model.fit_output_scaler(Y[tr])

        # Transform outputs (scaled IV space)
        Y_tr_scaled = model.transform_output(Y[tr])
        Y_va_scaled = model.transform_output(Y[va])

        # Tensors
        Xtr = torch.tensor(X_scaled[tr], dtype=torch.float32)
        Ytr = torch.tensor(Y_tr_scaled, dtype=torch.float32)
        Xva = torch.tensor(X_scaled[va], dtype=torch.float32)
        Yva = torch.tensor(Y_va_scaled, dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(Xtr, Ytr), batch_size=batch_size, shuffle=shuffle_training_batches
        )
        val_loader = DataLoader(
            TensorDataset(Xva, Yva), batch_size=2 * batch_size, shuffle=False
        )

        return model, train_loader, val_loader, strikes, maturities


    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_model(
        self,
        train_loader,
        val_loader=None,
        epochs=50,
        lr_schedule=[(0, 1e-3), (30, 5e-4), (60, 1e-4)],
    ):
        """
        Standard MLP training. Tracks both scaled-RMSE and true-IV-RMSE.
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

            # --- LR update ---
            if (
                schedule_index + 1 < len(lr_schedule)
                and epoch >= lr_schedule[schedule_index + 1][0]
            ):
                schedule_index += 1
                new_lr = lr_schedule[schedule_index][1]
                for g in self.optimizer.param_groups:
                    g["lr"] = new_lr
                sys.stdout.write(
                    f"\n→ Adjusted learning rate to {new_lr:.2e} at epoch {epoch}\n"
                )

            # --- TRAINING ---
            self.train()
            total_scaled_loss = 0.0
            total_iv_rmse = 0.0
            n_samples = len(train_loader.dataset)

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                pred = self.forward(x)
                loss = self.criterion(pred, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_size = len(y)
                total_scaled_loss += loss.item() * batch_size

                # --- true IV RMSE ---
                pred_np = pred.detach().cpu().numpy()
                y_np = y.detach().cpu().numpy()

                batch_rmse = []
                for i in range(batch_size):
                    pi = self.inverse_transform_surface(pred_np[i])
                    yi = self.inverse_transform_surface(y_np[i])
                    batch_rmse.append(
                        np.sqrt(np.mean((pi - yi) ** 2, dtype=np.float64))
                    )

                total_iv_rmse += float(np.mean(batch_rmse)) * batch_size

            train_rmse_scaled = float(np.sqrt(total_scaled_loss / n_samples))
            train_rmse_iv = float(total_iv_rmse / n_samples)

            # --- VALIDATION ---
            if val_loader is not None:
                val_rmse_scaled, val_rmse_iv = self._validate_dual(val_loader)
                msg = (
                    f"Epoch {epoch+1:03d} | "
                    f"train_scaled={train_rmse_scaled:.6f}, val_scaled={val_rmse_scaled:.6f}, "
                    f"train_iv={train_rmse_iv:.6f}, val_iv={val_rmse_iv:.6f}, "
                    f"lr={self.optimizer.param_groups[0]['lr']:.1e}"
                )
            else:
                msg = (
                    f"Epoch {epoch+1:03d} | "
                    f"train_scaled={train_rmse_scaled:.6f}, train_iv={train_rmse_iv:.6f}, "
                    f"lr={self.optimizer.param_groups[0]['lr']:.1e}"
                )

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

                # scaled loss
                scaled_loss = self.criterion(pred, y).item()
                batch = y.shape[0]
                total_scaled += scaled_loss * batch

                pred_np = pred.detach().cpu().numpy()
                y_np = y.detach().cpu().numpy()

                rmse_list = []
                for i in range(batch):
                    pred_iv = self.inverse_transform_surface(pred_np[i])
                    y_iv = self.inverse_transform_surface(y_np[i])
                    rmse_list.append(np.sqrt(np.mean((pred_iv - y_iv) ** 2)))

                total_iv += float(np.mean(rmse_list)) * batch

        return (
            float(np.sqrt(total_scaled / n)),  # scaled RMSE
            float(total_iv / n),  # IV-RMSE
        )

    def validate(self, val_loader):
        self.eval()
        total = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                total += self.criterion(self.forward(x), y).item() * len(y)
        return total / len(val_loader.dataset)

    # ------------------------------------------------------------------
    # predict_surface with interpolation modes
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # predict_surface with interpolation modes (modellagnostisch)
    # ------------------------------------------------------------------
    def predict_surface(self, params, grid=None, mask_flat=None):
        """
        Predict surface on model-grid OR interpolate model→market.

        Supports masked interpolation:
            If mask_flat is not None:
                - only compute values at masked coordinates
                - return full (nT, nK) with NaNs elsewhere
        """
        import torch
        device = self.device

        # ---------------------------------------------------------
        # 1) PARAM → x_scaled
        # ---------------------------------------------------------
        if isinstance(params, dict):
            x_vec = self._params_dict_to_vec_tensor(params)
        else:
            if isinstance(params, torch.Tensor):
                x_vec = params.to(device=device, dtype=torch.float32).flatten()
            else:
                x_vec = torch.tensor(params, dtype=torch.float32, device=device).flatten()

        lb = torch.as_tensor(self.param_bounds[0], dtype=torch.float32, device=device)
        ub = torch.as_tensor(self.param_bounds[1], dtype=torch.float32, device=device)
        x_scaled = 2.0 * (x_vec - lb) / (ub - lb) - 1.0

        # Base prediction on model grid (scaled → unscaled)
        pred_scaled = self.forward(x_scaled.unsqueeze(0)).squeeze(0)  # (nT0, nK0)

        pred_flat  = pred_scaled.reshape(-1)
        mean = torch.as_tensor(self.output_scaler.mean_,  dtype=torch.float32, device=device)
        std  = torch.as_tensor(self.output_scaler.scale_, dtype=torch.float32, device=device)
        base_surface = (pred_flat * std + mean).reshape(*self.output_shape)  # (nT0, nK0)

        # If no interpolation requested → return model base-grid
        if grid is None:
            return base_surface

        # ---------------------------------------------------------
        # 2) Interpolation setup
        # ---------------------------------------------------------
        base_T = torch.as_tensor(self.maturities, dtype=torch.float32, device=device)
        base_K = torch.as_tensor(self.strikes,    dtype=torch.float32, device=device)

        tgt_T = torch.as_tensor(grid["maturities"], dtype=torch.float32, device=device)
        tgt_K = torch.as_tensor(grid["strikes"],    dtype=torch.float32, device=device)

        nT, nK = tgt_T.numel(), tgt_K.numel()

        # Normalized coordinates for bilinear interpolation
        T_min, T_max = base_T.min(), base_T.max()
        K_min, K_max = base_K.min(), base_K.max()

        # normalize target coordinates to [-1,1]
        TT, KK = torch.meshgrid(tgt_T, tgt_K, indexing="ij")
        TTn = 2.0 * (TT - T_min) / (T_max - T_min) - 1.0
        KKn = 2.0 * (KK - K_min) / (K_max - K_min) - 1.0

        # Full target grid as (nT*nK, 2)
        coord_full = torch.stack([KKn.reshape(-1), TTn.reshape(-1)], dim=1)  # [K-first, T-second]

        # ---------------------------------------------------------
        # 3) If mask_flat=None → normal full interpolation
        # ---------------------------------------------------------
        if mask_flat is None:
            grid_torch = torch.stack([KKn, TTn], dim=-1).unsqueeze(0)  # (1, nT, nK, 2)
            img = base_surface.unsqueeze(0).unsqueeze(0)               # (1, 1, nT0, nK0)

            interp = torch.nn.functional.grid_sample(
                img,
                grid_torch,
                mode="bilinear",
                padding_mode="reflection",
                align_corners=True,
            )
            return interp.squeeze(0).squeeze(0)  # (nT, nK)

        # ---------------------------------------------------------
        # 4) Masked interpolation: only compute for masked coords
        # ---------------------------------------------------------
        mask_t = torch.as_tensor(mask_flat, dtype=torch.bool, device=device).reshape(-1)
        masked_coords = coord_full[mask_t]  # (n_masked, 2)

        # Reshape into a grid_sample query (batch=1)
        grid_masked = masked_coords.unsqueeze(0).unsqueeze(0)  # (1, 1, n_masked, 2)

        img = base_surface.unsqueeze(0).unsqueeze(0)  # (1, 1, H0, W0)

        interp_masked = torch.nn.functional.grid_sample(
            img,
            grid_masked,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=True,
        ).squeeze(0).squeeze(0)  # shape (n_masked)

        # ---------------------------------------------------------
        # 5) Reconstruct full (nT, nK) output with NaNs
        # ---------------------------------------------------------
        out = torch.full((nT * nK,), float('nan'), dtype=torch.float32, device=device)
        out[mask_t] = interp_masked  # fill only masked coords

        return out.reshape(nT, nK)














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
