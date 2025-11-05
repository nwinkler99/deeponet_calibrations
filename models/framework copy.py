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
# Base Model with shared eval + persistence utilities
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
        self._last_pred = None       # cache last predicted surface
        self._last_true = None       # cache last true surface
        self._last_params = None     # cache last params dict

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

        Produces 6 heatmaps total:
        Row 1 -> Predicted relative errors (computed on the fly)
        Row 2 -> Precomputed relative errors (sample["iv_rel_error"])
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

            # Compute relative errors (%)
            rel_err_pred = (
                np.abs(true_surface - pred_surface) / np.clip(true_surface, 1e-6, None)
            ) * 100.0
            pred_rel_errs.append(rel_err_pred)

            rel_err_pre = np.array(sample["iv_rel_error"], dtype=np.float32) * 100.0
            precomp_rel_errs.append(rel_err_pre)

        pred_rel_errs = np.stack(pred_rel_errs, axis=0)       # (num_samples, nT, nK)
        precomp_rel_errs = np.stack(precomp_rel_errs, axis=0) # same shape

        # Fill NaNs by mean at that (T,K) position
        def fill_nans_by_mean(arr):
            mean_over_samples = np.nanmean(arr, axis=0)
            return np.where(np.isnan(arr), np.expand_dims(mean_over_samples, 0), arr)

        pred_rel_errs = fill_nans_by_mean(pred_rel_errs)
        precomp_rel_errs = fill_nans_by_mean(precomp_rel_errs)

        # Aggregate per (K,T)
        mean_pred, median_pred, max_pred = (
            np.mean(pred_rel_errs, axis=0),
            np.median(pred_rel_errs, axis=0),
            np.max(pred_rel_errs, axis=0),
        )
        mean_pre, median_pre, max_pre = (
            np.mean(precomp_rel_errs, axis=0),
            np.median(precomp_rel_errs, axis=0),
            np.max(precomp_rel_errs, axis=0),
        )

        # Plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        Ks, Ts = np.meshgrid(self.strikes, self.maturities, indexing="xy")

        # Row 1 — predicted relative errors
        for ax, data, title in zip(
            axes[0],
            [mean_pred, median_pred, max_pred],
            ["Mean Rel Error (Pred)", "Median Rel Error (Pred)", "Max Rel Error (Pred)"]
        ):
            im = ax.pcolormesh(Ks, Ts, data, cmap="magma", shading="auto")
            ax.set_xlabel("Strike (K)")
            ax.set_ylabel("Maturity (T)")
            ax.set_title(f"{title} [%]")
            ax.invert_yaxis()  # shorter maturities at the top
            fig.colorbar(im, ax=ax, label="%")

        # Row 2 — precomputed relative errors
        for ax, data, title in zip(
            axes[1],
            [mean_pre, median_pre, max_pre],
            ["Mean Rel Error (Precomp)", "Median Rel Error (Precomp)", "Max Rel Error (Precomp)"]
        ):
            im = ax.pcolormesh(Ks, Ts, data, cmap="viridis", shading="auto")
            ax.set_xlabel("Strike (K)")
            ax.set_ylabel("Maturity (T)")
            ax.set_title(f"{title} [%]")
            ax.invert_yaxis()  # shorter maturities at the top
            fig.colorbar(im, ax=ax, label="%")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "iv_error_heatmaps.png"), dpi=200)
        #plt.close(fig)

        return {
            "pred_rel": {"mean": mean_pred, "median": median_pred, "max": max_pred},
            "precomp_rel": {"mean": mean_pre, "median": median_pre, "max": max_pre},
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

    def fit_scalers(self, X_train, Y_train):
        """
        Fits scalers on training data only (prevents data leakage).
        Stores them as self.input_scaler and self.output_scaler.
        """
        from sklearn.preprocessing import StandardScaler

        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        self.input_scaler.fit(X_train)
        self.output_scaler.fit(Y_train.reshape(len(Y_train), -1))

    def transform_input(self, X):
        return self.input_scaler.transform(X)

    def inverse_transform_input(self, X_scaled):
        return self.input_scaler.inverse_transform(X_scaled)

    def transform_output(self, Y):
        shp = Y.shape
        Y_scaled = self.output_scaler.transform(Y.reshape(len(Y), -1))
        return Y_scaled.reshape(shp)

    def inverse_transform_output(self, Y_scaled):
        shp = Y_scaled.shape
        Y = self.output_scaler.inverse_transform(Y_scaled.reshape(len(Y_scaled), -1))
        return Y.reshape(shp)
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
    def prepare_data(surfaces, batch_size=256, val_split=0.2, shuffle=True):
        """
        Flattens surfaces into per-point supervision for DeepONet.
        Returns: train_loader, val_loader, branch_dim, strikes, maturities
        """
        Xb_list, Xt_list, Y_list = [], [], []
        Ks_ref, Ts_ref = None, None

        for surf in surfaces:
            params = surf["params"]
            iv_surface = np.array(surf["iv_surface"], dtype=np.float32)
            Ks = np.array(surf["grid"]["strikes"], dtype=np.float32)
            Ts = np.array(surf["grid"]["maturities"], dtype=np.float32)

            xi0_knots = np.array(params["xi0_knots"]).flatten()
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

        # Split
        n_total = len(Y)
        n_train = int((1 - val_split) * n_total)
        idx = np.arange(n_total)
        if shuffle:
            np.random.shuffle(idx)
        train_idx, val_idx = idx[:n_train], idx[n_train:]

        train_ds = IVSurfaceDataset(X_branch[train_idx], X_trunk[train_idx], Y[train_idx])
        val_ds   = IVSurfaceDataset(X_branch[val_idx], X_trunk[val_idx], Y[val_idx])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=2*batch_size, shuffle=False)

        branch_dim = X_branch.shape[1]
        strikes = Ks_ref if Ks_ref is not None else np.array(surfaces[0]["grid"]["strikes"], dtype=np.float32)
        maturities = Ts_ref if Ts_ref is not None else np.array(surfaces[0]["grid"]["maturities"], dtype=np.float32)

        return train_loader, val_loader, branch_dim, strikes, maturities

    # --------------------------------------------------------
    def train_model(self, train_loader, val_loader=None, epochs=10):
        for epoch in range(epochs):
            self.train(); total_loss = 0.0
            for xb, xt, y in tqdm(train_loader, desc=f"Train {epoch+1}", leave=False):
                xb, xt, y = xb.to(self.device), xt.to(self.device), y.to(self.device)
                pred = self.forward(xb, xt)
                loss = self.compute_loss(pred, y)
                self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
                total_loss += loss.item() * len(y)
            train_rmse = float(np.sqrt(total_loss / len(train_loader.dataset)))

            if val_loader is not None:
                val_rmse = float(np.sqrt(self.validate(val_loader)))
                print(f"Epoch {epoch+1:03d} | train_rmse={train_rmse:.6f}, val_rmse={val_rmse:.6f}")
            else:
                print(f"Epoch {epoch+1:03d} | train_rmse={train_rmse:.6f}")

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
        Returns (nT, nK) numpy array.
        """
        assert self.strikes is not None and self.maturities is not None, \
            "DeepONet needs self.strikes/self.maturities set; call set_grid or train/prepare first."

        xi0_knots = np.array(params["xi0_knots"]).flatten()
        branch_vec = np.concatenate([[params["eta"], params["rho"], params["H"]], xi0_knots]).astype(np.float32)

        K_mesh, T_mesh = np.meshgrid(self.strikes, self.maturities)
        trunk_coords = np.stack([K_mesh.flatten(), T_mesh.flatten()], axis=1).astype(np.float32)

        with torch.no_grad():
            xb = torch.tensor(branch_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            xb = xb.repeat(len(trunk_coords), 1)
            xt = torch.tensor(trunk_coords, dtype=torch.float32, device=self.device)
            pred = self.forward(xb, xt)  # (nPts, 1)

        surface = pred.detach().cpu().numpy().reshape(len(self.maturities), len(self.strikes))

        return surface


# ============================================================
# MLP_IVSurface (self-contained)
# ============================================================

class MLP(BaseModel):
    """
    Simple MLP mapping parameter vector -> implied volatility surface.
    Output is the full (nT, nK) surface predicted in one shot.
    """
    def __init__(self, input_dim=None, output_shape=None, hidden_dims=(256, 256, 256),
                 activation="gelu", lr=1e-3):
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
    def prepare_data(surfaces, batch_size=32, val_split=0.2, shuffle=True, sanity_check_grids=True):
        """
        Converts list of surfaces into param vectors + full surfaces.
        Returns: train_loader, val_loader, input_dim, output_shape, strikes, maturities
        """
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
        nT, nK = Y.shape[1:]
        input_dim = X.shape[1]
        output_shape = (nT, nK)

        # Split
        n_total = len(Y)
        n_train = int((1 - val_split) * n_total)
        idx = np.arange(n_total)
        if shuffle:
            np.random.shuffle(idx)
        tr, va = idx[:n_train], idx[n_train:]

        # --- Fit scalers on training only ---
        model_scaler = BaseModel()  # temporary just to use the methods
        model_scaler.fit_scalers(X[tr], Y[tr])
        Xtr_scaled = model_scaler.transform_input(X[tr])
        Xva_scaled = model_scaler.transform_input(X[va])
        Ytr_scaled = model_scaler.transform_output(Y[tr])
        Yva_scaled = model_scaler.transform_output(Y[va])

        # Tensors
        Xtr = torch.tensor(Xtr_scaled, dtype=torch.float32)
        Ytr = torch.tensor(Ytr_scaled, dtype=torch.float32)
        Xva = torch.tensor(Xva_scaled, dtype=torch.float32)
        Yva = torch.tensor(Yva_scaled, dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(TensorDataset(Xva, Yva), batch_size=2 * batch_size, shuffle=False)

        return train_loader, val_loader, input_dim, output_shape, model_scaler


    # --------------------------------------------------------
    def train_model(self, train_loader, val_loader=None, epochs=50):
        for epoch in range(epochs):
            self.train(); total = 0.0
            for x, y in tqdm(train_loader, desc=f"Train {epoch+1}", leave=False):
                x, y = x.to(self.device), y.to(self.device)
                pred = self.forward(x)
                loss = self.criterion(pred, y)
                self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
                total += loss.item() * len(y)
            train_rmse = float(np.sqrt(total / len(train_loader.dataset)))

            if val_loader is not None:
                val_rmse = float(np.sqrt(self.validate(val_loader)))
                print(f"Epoch {epoch+1:03d} | train_rmse={train_rmse:.6f}, val_rmse={val_rmse:.6f}")
            else:
                print(f"Epoch {epoch+1:03d} | train_rmse={train_rmse:.6f}")

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
        Returns (nT, nK) numpy array.
        """
        assert self.output_shape is not None, "MLP needs output_shape set."
        assert self.strikes is not None and self.maturities is not None, \
            "MLP needs self.strikes/self.maturities set; call set_grid or train/prepare first."

        xi0_knots = np.array(params["xi0_knots"]).flatten()
        x = np.concatenate([[params["eta"], params["rho"], params["H"]], xi0_knots]).astype(np.float32)
        x = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            pred = self.forward(x)  # (1, nT, nK)

        surface = pred.squeeze(0).detach().cpu().numpy()

        # Safety: ensure shape matches grid
        nT, nK = self.output_shape
        if surface.shape != (nT, nK):
            surface = surface.reshape(nT, nK)

        return surface


# ============================================================
# Usage Notes (example)
# ============================================================
# DeepONet:
#   train_loader, val_loader, bdim, Ks, Ts = DeepONet.prepare_data(train_surfaces)
#   model = DeepONet(branch_in_dim=bdim, latent_dim=64, hidden_dim=64, lr=1e-3)
#   model.set_grid(Ks, Ts)
#   model.set_io_dims(input_dim=bdim)
#   model.train_model(train_loader, val_loader, epochs=100)
#   fig = model.plot_evaluation(test_surfaces[0])  # no extra args needed
#   model.evaluate_and_save(test_surfaces, out_dir="deeponet_eval")

# MLP:
#   train_loader, val_loader, in_dim, out_shape, Ks, Ts = MLP_IVSurface.prepare_data(train_surfaces)
#   mlp = MLP_IVSurface(input_dim=in_dim, output_shape=out_shape, hidden_dims=(256,256,256), lr=1e-3)
#   mlp.set_grid(Ks, Ts)
#   mlp.set_io_dims(input_dim=in_dim, output_shape=out_shape)
#   mlp.train_model(train_loader, val_loader, epochs=100)
#   fig = mlp.plot_evaluation(test_surfaces[0])
#   mlp.evaluate_and_save(test_surfaces, out_dir="mlp_eval")
