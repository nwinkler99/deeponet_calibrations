# ======================================================================
# DeepONet Framework (with integrated data preparation)
# ======================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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
        normalize=True
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

        # Optional normalization for stability
        if normalize:
            X_trunk_mean, X_trunk_std = X_trunk.mean(0), X_trunk.std(0)
            Y_mean, Y_std = Y.mean(), Y.std()
            X_trunk = (X_trunk - X_trunk_mean) / (X_trunk_std + 1e-8)
            Y = (Y - Y_mean) / (Y_std + 1e-8)
        else:
            X_trunk_mean = X_trunk_std = Y_mean = Y_std = None

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
            if val_loss is not None:
                print(f"Epoch {epoch+1:02d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
            else:
                print(f"Epoch {epoch+1:02d}: train_loss={train_loss:.6f}")




