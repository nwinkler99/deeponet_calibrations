# ======================================================================
# DeepONet Framework (with integrated data preparation)
# ======================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm

# ============================================================
# Dataset Wrapper
# ============================================================

class IVSurfaceDataset(Dataset):
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
# Base Model
# ============================================================

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def configure_optimizer(self, lr=1e-3):
        return optim.Adam(self.parameters(), lr=lr)

    def compute_loss(self, y_pred, y_true):
        return nn.MSELoss()(y_pred, y_true)


# ============================================================
# DeepONet Model (with built-in data preparation)
# ============================================================

class DeepONet(BaseModel):
    """Deep Operator Network for implied volatility surfaces."""
    def __init__(self, branch_in_dim=None, trunk_in_dim=2, latent_dim=64, hidden_dim=64):
        super().__init__()
        self.branch_in_dim = branch_in_dim
        self.trunk_in_dim = trunk_in_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Define networks only after knowing input dims
        if branch_in_dim:
            self._build_networks()

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

    def forward(self, xb, xt):
        b_out = self.branch(xb)
        t_out = self.trunk(xt)
        return torch.sum(b_out * t_out, dim=1, keepdim=True)

    # ========================================================
    # Architecture-specific data preparation
    # ========================================================
    @staticmethod
    def prepare_data(
        surfaces,
        batch_size=256,
        val_split=0.2,
        shuffle=True,
        normalize=False
    ):
        """
        Convert Rough Bergomi generator surfaces into train/test DataLoaders.
        """
        Xb_list, Xt_list, Y_list = [], [], []

        for surf in surfaces:
            params = surf["params"]
            iv_surface = surf["iv_surface"]
            Ks = surf["grid"]["strikes"]
            Ts = surf["grid"]["maturities"]

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

        # Only normalize X_trunk based on training data
        if normalize:
            X_trunk_mean, X_trunk_std = X_trunk.mean(0), X_trunk.std(0)
            X_trunk = (X_trunk - X_trunk_mean) / (X_trunk_std + 1e-8)
        else:
            # no normalization of inputs either
            X_trunk_mean = X_trunk_std = None

        # Split train/test
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

        print(f"Prepared data → {len(train_ds)} train / {len(val_ds)} val samples")
        print(f"Branch dim: {branch_dim}, Trunk dim: {X_trunk.shape[1]}")

        # Return only loaders and branch dim (no normalization stats by default)
        return train_loader, val_loader, branch_dim


# ============================================================
# Trainer
# ============================================================

class Trainer:
    def __init__(self, model, train_loader, val_loader=None, lr=1e-3, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = model.configure_optimizer(lr)
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for xb, xt, y in tqdm(self.train_loader, desc="Train", leave=False):
            xb, xt, y = xb.to(self.device), xt.to(self.device), y.to(self.device)
            y_pred = self.model(xb, xt)
            loss = self.model.compute_loss(y_pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(y)
        return total_loss / len(self.train_loader.dataset)

    def validate_epoch(self):
        if self.val_loader is None:
            return None
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for xb, xt, y in self.val_loader:
                xb, xt, y = xb.to(self.device), xt.to(self.device), y.to(self.device)
                y_pred = self.model(xb, xt)
                loss = self.model.compute_loss(y_pred, y)
                total_loss += loss.item() * len(y)
        return total_loss / len(self.val_loader.dataset)

    def fit(self, epochs=10):
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            
            # Calculate RMSE directly (already in IV space)
            train_rmse = float(np.sqrt(train_loss))
            val_rmse = float(np.sqrt(val_loss)) if val_loss is not None else None

            if val_loss is not None:
                print(f"Epoch {epoch+1:02d}: train_rmse={train_rmse:.6f}, val_rmse={val_rmse:.6f}")
            else:
                print(f"Epoch {epoch+1:02d}: train_rmse={train_rmse:.6f}")


# ============================================================
# Model Evaluator
# ============================================================

class ModelEvaluator:
    def __init__(self, model, x_trunk_mean=None, x_trunk_std=None, device=None):
        """Initialize the evaluator with a trained model and optional X normalization stats.

        If X normalization stats are provided they will be used for input normalization.
        Otherwise, inputs are used as-is.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

        # X normalization stats from training (optional)
        self.x_trunk_mean = np.array(x_trunk_mean) if x_trunk_mean is not None else None
        self.x_trunk_std = np.array(x_trunk_std) if x_trunk_std is not None else None

    def set_normalization(self, x_trunk_mean, x_trunk_std):
        """Set X normalization parameters explicitly from training stats."""
        self.x_trunk_mean = np.array(x_trunk_mean)
        self.x_trunk_std = np.array(x_trunk_std)

    def predict_surface(self, branch_input, strikes, maturities, normalize_input=False):
        """Generate predictions for a full surface given branch input and grid.

        Input normalization uses training statistics if available and normalize_input=True.
        Otherwise inputs are used as-is.
        """
        K_mesh, T_mesh = np.meshgrid(strikes, maturities)
        trunk_coords = np.stack([K_mesh.flatten(), T_mesh.flatten()], axis=1).astype(np.float32)

        if normalize_input and (self.x_trunk_mean is not None) and (self.x_trunk_std is not None):
            trunk_coords = (trunk_coords - self.x_trunk_mean) / (self.x_trunk_std + 1e-8)

        branch_input = torch.tensor(branch_input, dtype=torch.float32).unsqueeze(0)
        trunk_coords = torch.tensor(trunk_coords, dtype=torch.float32)

        with torch.no_grad():
            branch_repeated = branch_input.repeat(len(trunk_coords), 1).to(self.device)
            trunk_coords = trunk_coords.to(self.device)
            predictions = self.model(branch_repeated, trunk_coords)

        return predictions.cpu().numpy().reshape(len(maturities), len(strikes))

    def compute_grid_mse(self, true_surface, pred_surface, strikes=None, maturities=None):
        """Compute MSE grid and return summary statistics.

        Returns (mse_grid, stats_dict) where stats include mean/std/max RMSE/MAE and location of max error.
        """
        abs_err = np.abs(true_surface - pred_surface)
        mse_grid = abs_err ** 2

        idx_flat = np.argmax(abs_err)
        idx = np.unravel_index(idx_flat, abs_err.shape)
        loc = None
        if (strikes is not None) and (maturities is not None):
            # idx -> (maturity_index, strike_index)
            maturity_idx, strike_idx = idx
            loc = {
                'maturity_index': int(maturity_idx),
                'strike_index': int(strike_idx),
                'strike': float(strikes[strike_idx]),
                'maturity': float(maturities[maturity_idx])
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

    def plot_evaluation(self, surface_data, figsize=(16, 10)):
        """Plot true surface, predicted surface, MSE heatmap and absolute error heatmap. Prints stats."""
        strikes = surface_data["grid"]["strikes"]
        maturities = surface_data["grid"]["maturities"]
        true_surface = surface_data["iv_surface"]
        params = surface_data["params"]

        # Prepare branch input
        xi0_knots = np.array(params["xi0_knots"]).flatten()
        branch_vec = np.concatenate([[params["eta"], params["rho"], params["H"]], xi0_knots])

        # Generate prediction (do not perform per-surface normalization by default)
        pred_surface = self.predict_surface(branch_vec, strikes, maturities, normalize_input=False)

        mse_grid, stats = self.compute_grid_mse(true_surface, pred_surface, strikes=strikes, maturities=maturities)

        # Print stats
        print("MSE stats:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

        # Create a 2x2 figure: true, pred (3D), mse heatmap, abs error heatmap
        fig = plt.figure(figsize=figsize)

        # True surface
        ax1 = fig.add_subplot(221, projection='3d')
        K_mesh, T_mesh = np.meshgrid(strikes, maturities)
        surf1 = ax1.plot_surface(K_mesh, T_mesh, true_surface, cmap=cm.viridis)
        ax1.set_title('True Surface')
        ax1.set_xlabel('Strike')
        ax1.set_ylabel('Maturity')
        ax1.set_zlabel('IV')
        plt.colorbar(surf1, ax=ax1, shrink=0.5, aspect=8)

        # Predicted surface
        ax2 = fig.add_subplot(222, projection='3d')
        surf2 = ax2.plot_surface(K_mesh, T_mesh, pred_surface, cmap=cm.viridis)
        ax2.set_title('Predicted Surface')
        ax2.set_xlabel('Strike')
        ax2.set_ylabel('Maturity')
        ax2.set_zlabel('IV')
        plt.colorbar(surf2, ax=ax2, shrink=0.5, aspect=8)

        # MSE heatmap
        ax3 = fig.add_subplot(223)
        im = ax3.imshow(mse_grid, cmap='hot',
                        extent=[strikes[0], strikes[-1], maturities[0], maturities[-1]],
                        aspect='auto', origin='lower')
        ax3.set_title('MSE Heatmap')
        ax3.set_xlabel('Strike')
        ax3.set_ylabel('Maturity')
        plt.colorbar(im, ax=ax3)

        # Absolute error heatmap
        ax4 = fig.add_subplot(224)
        abs_err = np.abs(true_surface - pred_surface)
        im2 = ax4.imshow(abs_err, cmap='inferno',
                         extent=[strikes[0], strikes[-1], maturities[0], maturities[-1]],
                         aspect='auto', origin='lower')
        ax4.set_title('Absolute Error Heatmap')
        ax4.set_xlabel('Strike')
        ax4.set_ylabel('Maturity')
        plt.colorbar(im2, ax=ax4)

        plt.tight_layout()
        return fig

    def evaluate_samples(self, surface_samples, save_path=None):
        """Evaluate and plot results for multiple samples."""
        for i, surface in enumerate(surface_samples):
            fig = self.plot_evaluation(surface)
            if save_path:
                plt.savefig(f"{save_path}/surface_eval_{i}.png")
                plt.close()
            else:
                plt.show()