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

    def compute_grid_mse(self, surface_data):
        """
        surface_data: dict with keys: "iv_surface", "grid", "params"
        Uses model's own grid + predict_surface.
        Returns: (mse_grid, stats)
        """
        assert self.strikes is not None and self.maturities is not None, \
            "Model grid (strikes/maturities) not set; call set_grid or train/prepare first."

        true_surface = np.array(surface_data["iv_surface"], dtype=np.float32)
        params = surface_data["params"]

        pred_surface = self.predict_surface(params)
        abs_err = np.abs(true_surface - pred_surface)
        mse_grid = abs_err ** 2

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

    def plot_evaluation(self, surface_data, figsize=(15, 5), levels=30):
        """
        Compare true vs. predicted IV surfaces using contour plots.
        Shows:
            [True Surface] [Predicted Surface] [Difference (Pred - True)]
        """
        assert self.strikes is not None and self.maturities is not None, \
            "Model grid (strikes/maturities) not set; call set_grid or train/prepare first."

        true_surface = np.array(surface_data["iv_surface"], dtype=np.float32)
        params = surface_data["params"]
        strikes, maturities = self.strikes, self.maturities

        # Predict surface
        pred_surface = self.predict_surface(params)
        diff = pred_surface - true_surface
        mae = np.mean(np.abs(diff))
        rmse = np.sqrt(np.mean(diff ** 2))

        # Compute symmetric color scale for difference
        vmax = np.max(np.abs(diff))

        # Make figure
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
        Evaluates model-predicted vs. precomputed relative IV errors across samples.

        For each (K,T) grid point:
        - Aggregates errors across all samples (mean, median, max)
        - Replaces NaNs by the mean of that (K,T) location across samples
        - Expresses results in percent

        Produces up to 6 heatmaps:
        Row 1 -> Predicted relative errors (computed on the fly)
        Row 2 -> Precomputed relative errors (sample["iv_rel_error"], if available)
        Maturity axis (T) is inverted: shorter maturities at the top.
        """
        assert self.strikes is not None and self.maturities is not None, \
            "Model grid (strikes/maturities) not set; call set_grid or train/prepare first."
        os.makedirs(out_dir, exist_ok=True)

        pred_rel_errs, precomp_rel_errs = [], []

        # Collect both predicted and stored errors
        for sample in surface_samples:
            params = sample["params"]
            true_surface = np.array(sample["iv_surface"], dtype=np.float32)
            pred_surface = self.predict_surface(params)

            # Compute predicted relative errors (%)
            rel_err_pred = (
                np.abs(true_surface - pred_surface) / np.clip(true_surface, 1e-6, None)
            ) * 100.0
            pred_rel_errs.append(rel_err_pred)

            # Only collect precomputed if available
            if "iv_rel_error" in sample and sample["iv_rel_error"] is not None:
                rel_err_pre = np.array(sample["iv_rel_error"], dtype=np.float32) * 100.0
                precomp_rel_errs.append(rel_err_pre)

        # Stack predictions
        pred_rel_errs = np.stack(pred_rel_errs, axis=0)

        # Fill NaNs by mean at that (T,K) position
        def fill_nans_by_mean(arr):
            mean_over_samples = np.nanmean(arr, axis=0)
            return np.where(np.isnan(arr), np.expand_dims(mean_over_samples, 0), arr)

        pred_rel_errs = fill_nans_by_mean(pred_rel_errs)

        # Aggregate predicted errors
        mean_pred, median_pred, max_pred = (
            np.mean(pred_rel_errs, axis=0),
            np.median(pred_rel_errs, axis=0),
            np.max(pred_rel_errs, axis=0),
        )

        # Precomputed branch only if available
        has_precomp = len(precomp_rel_errs) > 0
        if has_precomp:
            precomp_rel_errs = np.stack(precomp_rel_errs, axis=0)
            precomp_rel_errs = fill_nans_by_mean(precomp_rel_errs)
            mean_pre, median_pre, max_pre = (
                np.mean(precomp_rel_errs, axis=0),
                np.median(precomp_rel_errs, axis=0),
                np.max(precomp_rel_errs, axis=0),
            )
        else:
            mean_pre = median_pre = max_pre = None

        # Plot
        rows = 2 if has_precomp else 1
        fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
        Ks, Ts = np.meshgrid(self.strikes, self.maturities, indexing="xy")

        # Row 1 — predicted relative errors
        for ax, data, title in zip(
            axes[0] if has_precomp else axes,
            [mean_pred, median_pred, max_pred],
            ["Mean Rel Error (Pred)", "Median Rel Error (Pred)", "Max Rel Error (Pred)"]
        ):
            im = ax.pcolormesh(Ks, Ts, data, cmap="magma", shading="auto")
            ax.set_xlabel("Strike (K)")
            ax.set_ylabel("Maturity (T)")
            ax.set_title(f"{title} [%]")
            ax.invert_yaxis()
            fig.colorbar(im, ax=ax, label="%")

        # Row 2 — precomputed relative errors (only if available)
        if has_precomp:
            for ax, data, title in zip(
                axes[1],
                [mean_pre, median_pre, max_pre],
                ["Mean Rel Error (Precomp)", "Median Rel Error (Precomp)", "Max Rel Error (Precomp)"]
            ):
                im = ax.pcolormesh(Ks, Ts, data, cmap="viridis", shading="auto")
                ax.set_xlabel("Strike (K)")
                ax.set_ylabel("Maturity (T)")
                ax.set_title(f"{title} [%]")
                ax.invert_yaxis()
                fig.colorbar(im, ax=ax, label="%")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "iv_error_heatmaps.png"), dpi=200)

        return {
            "pred_rel": {"mean": mean_pred, "median": median_pred, "max": max_pred},
            "precomp_rel": (
                {"mean": mean_pre, "median": median_pre, "max": max_pre} if has_precomp else None
            ),
        }



# ============================================================
# DeepONet (self-contained)
# ============================================================

class DeepONet(BaseModel):
    """
    Deep Operator Network for implied volatility surfaces.
    - Branch input: parameter vector θ (eta, rho, H, xi0_knots...)
    - Trunk input: 2D coords (K, T)
    - Output: IV value per (K, T)
    """
    def __init__(self, branch_in_dim=None, trunk_in_dim=2, latent_dim=64, hidden_dim=64, lr=1e-3):
        super().__init__()
        self.branch_in_dim = branch_in_dim
        self.trunk_in_dim = trunk_in_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.lr = lr

        if branch_in_dim:
            self._build_networks()
        self.to(self.device)

    def _build_networks(self):
        self.branch = nn.Sequential(
            nn.Linear(self.branch_in_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim)
        )
        self.trunk = nn.Sequential(
            nn.Linear(self.trunk_in_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, xb, xt):
        return torch.sum(self.branch(xb) * self.trunk(xt), dim=1, keepdim=True)

    # --------------------------------------------------------
    @staticmethod
    def _flatten_surfaces_for_deeponet(surfaces, sanity_check_grids=True):
        Xb_list, Xt_list, Y_list = [], [], []
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
                        raise ValueError("All surfaces must share the same (K, T) grid.")

            xi0_knots = np.array(params["xi0_knots"]).flatten().astype(np.float32)
            branch_vec = np.concatenate([[params["eta"], params["rho"], params["H"]], xi0_knots]).astype(np.float32)

            K_mesh, T_mesh = np.meshgrid(Ks, Ts)
            trunk_coords = np.stack([K_mesh.flatten(), T_mesh.flatten()], axis=1).astype(np.float32)
            branch_repeated = np.repeat(branch_vec[None, :], len(trunk_coords), axis=0)
            Y_flat = iv_surface.flatten()[:, None].astype(np.float32)

            Xb_list.append(branch_repeated)
            Xt_list.append(trunk_coords)
            Y_list.append(Y_flat)

        X_branch = np.concatenate(Xb_list, axis=0)
        X_trunk  = np.concatenate(Xt_list, axis=0)
        Y        = np.concatenate(Y_list, axis=0)

        strikes = Ks_ref
        maturities = Ts_ref
        return X_branch, X_trunk, Y, strikes, maturities

    @classmethod
    def from_surfaces(cls, surfaces, batch_size=256, val_split=0.2, shuffle=True,
                      eta=(0.5, 4.0), rho=(0.0, 1.0), H=(0.025, 0.5), knot=(0.01, 0.16)):
        """
        Build a DeepONet model + loaders with internal, leakage-safe scaling:
          - Branch inputs scaled to [-1, 1] using provided bounds
          - Output (IV) scaled via StandardScaler fit on train only
        Returns: model, train_loader, val_loader, strikes, maturities
        """
        # Flatten to per-point supervision
        X_branch, X_trunk, Y, strikes, maturities = cls._flatten_surfaces_for_deeponet(surfaces)

        # Build param bounds from branch vector length
        num_knots = X_branch.shape[1] - 3
        lb, ub = BaseModel._make_param_bounds(num_knots, eta=eta, rho=rho, H=H, knot=knot)

        # Scale branch inputs by bounds to [-1,1]
        Xb_scaled = BaseModel._scale_to_m1_p1(X_branch, lb, ub)

        # Split
        n_total = len(Y)
        n_train = int((1 - val_split) * n_total)
        idx = np.arange(n_total)
        if shuffle:
            np.random.shuffle(idx)
        tr, va = idx[:n_train], idx[n_train:]

        # Fit output scaler on train only
        model = cls(branch_in_dim=Xb_scaled.shape[1])
        model.set_grid(strikes, maturities)
        model.set_io_dims(input_dim=Xb_scaled.shape[1])
        model.set_param_bounds(lb, ub)
        model.fit_output_scaler(Y[tr])

        # Transform outputs
        Y_tr_scaled = model.transform_output(Y[tr])
        Y_va_scaled = model.transform_output(Y[va])

        # Build datasets
        train_ds = IVSurfaceDataset(Xb_scaled[tr], X_trunk[tr], Y_tr_scaled)
        val_ds   = IVSurfaceDataset(Xb_scaled[va], X_trunk[va], Y_va_scaled)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        val_loader   = DataLoader(val_ds, batch_size=2*batch_size, shuffle=False)

        # Build networks now that dims are known
        model._build_networks()
        return model, train_loader, val_loader, strikes, maturities

    # --------------------------------------------------------
    def train_model(
        self,
        train_loader,
        val_loader=None,
        epochs=10,
        lr_schedule=[(0, 1e-3), (5, 5e-4), (8, 1e-4)],
    ):
        """
        Train the DeepONet model with optional learning-rate schedule.

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
        # Sort and initialize LR
        lr_schedule = sorted(lr_schedule, key=lambda x: x[0])
        schedule_index = 0
        base_lr = lr_schedule[0][1]
        for g in self.optimizer.param_groups:
            g["lr"] = base_lr

        for epoch in range(epochs):
            # Update LR if we reached next threshold
            if (
                schedule_index + 1 < len(lr_schedule)
                and epoch >= lr_schedule[schedule_index + 1][0]
            ):
                schedule_index += 1
                new_lr = lr_schedule[schedule_index][1]
                for g in self.optimizer.param_groups:
                    g["lr"] = new_lr
                print(f"→ Adjusted learning rate to {new_lr:.2e} at epoch {epoch}")

            self.train()
            total_loss = 0.0
            for xb, xt, y in tqdm(train_loader, desc=f"Train {epoch+1}", leave=False):
                xb, xt, y = xb.to(self.device), xt.to(self.device), y.to(self.device)
                pred = self.forward(xb, xt)
                loss = self.compute_loss(pred, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * len(y)

            train_rmse = float(np.sqrt(total_loss / len(train_loader.dataset)))

            if val_loader is not None:
                val_rmse = float(np.sqrt(self.validate(val_loader)))
                print(
                    f"Epoch {epoch+1:03d} | "
                    f"train_rmse={train_rmse:.6f}, val_rmse={val_rmse:.6f}, "
                    f"lr={self.optimizer.param_groups[0]['lr']:.1e}"
                )
            else:
                print(
                    f"Epoch {epoch+1:03d} | "
                    f"train_rmse={train_rmse:.6f}, "
                    f"lr={self.optimizer.param_groups[0]['lr']:.1e}"
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
    def predict_surface(self, params):
        """
        params: dict with keys ("eta", "rho", "H", "xi0_knots")
        Uses self.strikes / self.maturities to build trunk coords.
        Returns (nT, nK) numpy array in ORIGINAL IV scale (inverse-transformed).
        """
        assert self.strikes is not None and self.maturities is not None, \
            "DeepONet needs self.strikes/self.maturities set; call set_grid or train/prepare first."
        assert self.param_bounds is not None, "param_bounds must be set for scaling"
        assert self.output_scaler is not None, "output_scaler must be fitted"

        xi0_knots = np.array(params["xi0_knots"]).flatten().astype(np.float32)
        param_vec = np.concatenate([[params["eta"], params["rho"], params["H"]], xi0_knots]).astype(np.float32)
        xb_np = self.scale_params(param_vec)  # [-1,1]

        K_mesh, T_mesh = np.meshgrid(self.strikes, self.maturities)
        trunk_coords = np.stack([K_mesh.flatten(), T_mesh.flatten()], axis=1).astype(np.float32)

        with torch.no_grad():
            xb = torch.tensor(xb_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            xb = xb.repeat(len(trunk_coords), 1)
            xt = torch.tensor(trunk_coords, dtype=torch.float32, device=self.device)
            pred_scaled = self.forward(xb, xt)  # (nPts, 1) in scaled IV space

        pred_scaled_np = pred_scaled.detach().cpu().numpy().reshape(-1)  # (nPts,)
        pred_iv = self.inverse_transform_output_single(pred_scaled_np)   # back to original IV
        surface = pred_iv.reshape(len(self.maturities), len(self.strikes))
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
                      hidden_dims=(256, 256, 256), activation="gelu", lr=1e-3,
                      eta=(0.5, 4.0), rho=(0.0, 1.0), H=(0.025, 0.5), knot=(0.01, 0.16)):
        """
        Build an MLP model + loaders with internal, leakage-safe scaling:
          - Param vector scaled to [-1, 1] using provided bounds
          - Output (IV) scaled via StandardScaler fit on train only
        Returns: model, train_loader, val_loader, strikes, maturities
        """
        X, Y, strikes, maturities = cls._stack_XY(surfaces)
        nT, nK = Y.shape[1:]
        input_dim = X.shape[1]
        output_shape = (nT, nK)

        # Build param bounds for the vector [eta, rho, H, knots...]
        num_knots = input_dim - 3
        lb, ub = BaseModel._make_param_bounds(num_knots, eta=eta, rho=rho, H=H, knot=knot)
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
        Train the MLP model with optional learning rate schedule.

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
        # Sort schedule for safety
        lr_schedule = sorted(lr_schedule, key=lambda x: x[0])
        schedule_index = 0

        # Initialize to first LR
        base_lr = lr_schedule[0][1]
        for g in self.optimizer.param_groups:
            g["lr"] = base_lr

        for epoch in range(epochs):
            # Check if learning rate needs to be updated
            if (
                schedule_index + 1 < len(lr_schedule)
                and epoch >= lr_schedule[schedule_index + 1][0]
            ):
                schedule_index += 1
                new_lr = lr_schedule[schedule_index][1]
                for g in self.optimizer.param_groups:
                    g["lr"] = new_lr
                print(f"→ Adjusted learning rate to {new_lr:.2e} at epoch {epoch}")

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

            if val_loader is not None:
                val_rmse = float(np.sqrt(self.validate(val_loader)))
                print(
                    f"Epoch {epoch+1:03d} | "
                    f"train_rmse={train_rmse:.6f}, val_rmse={val_rmse:.6f}, "
                    f"lr={self.optimizer.param_groups[0]['lr']:.1e}"
                )
            else:
                print(
                    f"Epoch {epoch+1:03d} | "
                    f"train_rmse={train_rmse:.6f}, "
                    f"lr={self.optimizer.param_groups[0]['lr']:.1e}"
                )

    def validate(self, val_loader):
        self.eval(); total = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                total += self.criterion(self.forward(x), y).item() * len(y)
        return total / len(val_loader.dataset)   

    # --------------------------------------------------------
    def predict_surface(self, params):
        """
        params: dict with keys ("eta", "rho", "H", "xi0_knots")
        Uses stored output shape and grid.
        Returns (nT, nK) numpy array in ORIGINAL IV scale (inverse-transformed).
        """
        assert self.output_shape is not None, "MLP needs output_shape set."
        assert self.strikes is not None and self.maturities is not None, \
            "MLP needs self.strikes/self.maturities set; call set_grid or train/prepare first."
        assert self.param_bounds is not None, "param_bounds must be set for scaling"
        assert self.output_scaler is not None, "output_scaler must be fitted"

        xi0_knots = np.array(params["xi0_knots"]).flatten().astype(np.float32)
        x = np.concatenate([[params["eta"], params["rho"], params["H"]], xi0_knots]).astype(np.float32)
        x_scaled = self.scale_params(x)

        with torch.no_grad():
            x_t = torch.tensor(x_scaled, dtype=torch.float32, device=self.device).unsqueeze(0)
            pred_scaled = self.forward(x_t)  # (1, nT, nK) in scaled IV space

        pred_scaled_np = pred_scaled.squeeze(0).detach().cpu().numpy()
        surface = self.inverse_transform_surface(pred_scaled_np)
        # Safety: ensure shape matches grid
        nT, nK = self.output_shape
        if surface.shape != (nT, nK):
            surface = surface.reshape(nT, nK)
        return surface


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
