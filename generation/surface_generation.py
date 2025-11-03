# Rough Bergomi IV surfaces via rDonsker fBm — Stable IV Extraction + OTM Options
# --------------------------------------------------------------------------------------------------------
# Features:
# - Constant time grid (consistent discretization)
# - Randomized Maturities (±15%)
# - OTM pricing (Calls for K>=S0, Puts for K<S0)
# - Stable implied vol inversion (clipping + NaN handling)
# - Piecewise constant xi0, Antithetic variates, batch seeding
# --------------------------------------------------------------------------------------------------------

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from scipy.stats import norm
import pandas as pd
from datetime import datetime
import os
import pickle

import numpy as np, os, pickle
from datetime import datetime
from typing import List, Dict
from scipy.stats import qmc   # Latin Hypercube sampler

import numpy as np
import os
import pickle
from datetime import datetime
from typing import List, Dict
from scipy.stats import qmc

from scipy.stats import qmc
import numpy as np
import os, pickle, numpy as np
from datetime import datetime
from typing import List, Dict
# -------------------------------------------------------------
# Parameter structures
# -------------------------------------------------------------

@dataclass
class RBergomiParams:
    eta: float
    rho: float
    H: float
    xi0_knots: np.ndarray

@dataclass
class SimulationConfig:
    M: int = 20000
    n: int = 1200
    T_max: float = 2.0
    S0: float = 1.0
    strikes: np.ndarray = None
    maturities: np.ndarray = None
    batch_size: int = 5000
    G: int = 10

    def __post_init__(self):
        if self.strikes is None:
            self.strikes = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
        if self.maturities is None:
            self.maturities = np.array([0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.0])


def sample_param_sets_lhs(num_sets: int) -> list["RBergomiParams"]:
    """
    Latin Hypercube sample of Rough Bergomi parameters.
    - η   ∈ [0.5, 4.0]
    - ρ   ∈ [-0.95, -0.1]
    - H   ∈ [0.025, 0.5]
    - xi₀_knots (8 values) ∈ [0.01, 0.16], random per set
    """
    sampler = qmc.LatinHypercube(d=3)
    sample = sampler.random(num_sets)  # shape (num_sets, 3)

    # scale entire matrix at once
    lower = np.array([0.5, -0.95, 0.025])
    upper = np.array([4.0, -0.1, 0.5])
    scaled = qmc.scale(sample, lower, upper)  # shape (num_sets, 3)

    eta_vals = scaled[:, 0]
    rho_vals = scaled[:, 1]
    H_vals   = scaled[:, 2]

    param_sets = []
    for i in range(num_sets):
        xi0_knots = np.random.uniform(0.01, 0.16, size=8)
        param_sets.append(RBergomiParams(
            eta=float(eta_vals[i]),
            rho=float(rho_vals[i]),
            H=float(H_vals[i]),
            xi0_knots=xi0_knots
        ))
    return param_sets


def jitter_grid(base_grid, grid_jitter=0.25, min_spacing=0.1):
    """
    Apply additive jitter to a monotonic grid (e.g. strikes or maturities),
    randomly distributing a total jitter budget while preserving order and spacing.

    Parameters
    ----------
    base_grid : np.ndarray
        Sorted 1D base grid (e.g. strikes or maturities).
    grid_jitter : float
        Fraction of total jitter budget relative to total grid range.
    min_spacing : float
        Minimum allowed distance between adjacent grid points.

    Returns
    -------
    np.ndarray
        Jittered, sorted grid with the same length as the input.
    """
    base_grid = np.sort(np.array(base_grid, dtype=float))
    n = len(base_grid)
    lo, hi = base_grid[0], base_grid[-1]
    total_range = hi - lo

    # define total jitter budget
    total_budget = grid_jitter * total_range

    for _ in range(100):  # retry loop to ensure spacing
        # randomly split the total budget into positive/negative perturbations
        signs = np.random.choice([-1, 1], size=n)
        weights = np.random.uniform(0, 1, size=n)
        weights /= np.sum(weights) + 1e-12  # normalize weights to sum to 1

        jitter = signs * weights * total_budget
        grid_shifted = base_grid + jitter

        # enforce boundaries
        grid_shifted = np.clip(grid_shifted, lo, hi)
        grid_shifted = np.sort(grid_shifted)

        # ensure no overlaps / minimum spacing
        if np.all(np.diff(grid_shifted) > min_spacing):
            return np.round(grid_shifted, 6)
    return np.round(grid_shifted, 6)


def repair_edges_local_directional(iv_surface, maturities, strikes, t_threshold=0.51):
    """
    Repairs implied-vol surfaces at short maturities by:
    1️⃣ Filling NaNs via 2D extrapolation.
    2️⃣ Applying directional edge correction (left/right up to 3 strikes inward).
    3️⃣ Replacing near-zero (<0.05) points with local neighbor mean.
    """

    iv = iv_surface.copy()
    nT, nK = iv.shape

    # ===============================================
    # Step 1: Fill NaNs via simple 2D extrapolation
    # ===============================================
    for i in range(nT):
        row = iv[i]
        if np.any(np.isnan(row)):
            valid = ~np.isnan(row)
            if np.any(valid):
                iv[i] = np.interp(
                    np.arange(nK),
                    np.arange(nK)[valid],
                    row[valid],
                    left=row[valid][0],
                    right=row[valid][-1],
                )
            else:
                iv[i] = np.nanmean(iv)

    if np.isnan(iv).any():
        col_means = np.nanmean(iv, axis=0)
        for j in range(nK):
            col = iv[:, j]
            nan_idx = np.isnan(col)
            if np.any(nan_idx):
                iv[nan_idx, j] = col_means[j]

    iv = np.nan_to_num(iv, nan=np.nanmean(iv))

    # ===============================================
    # Step 2: Apply directional edge correction (extended inward)
    # ===============================================
    short_idx = np.where(maturities <= t_threshold)[0]
    for i in reversed(short_idx):
        if i >= nT - 1:
            continue  # skip last maturity

        # --- Left side (in→out) ---
        for offset in reversed(range(3)):  # 2,1,0
            j = offset
            if j + 1 < nK:
                iv[i, j] = max(iv[i, j], iv[i, j + 1])

        # --- Right side (in→out) ---
        for offset in reversed(range(3)):  # 2,1,0
            j = nK - 1 - offset
            if j - 1 >= 0:
                iv[i, j] = max(iv[i, j], iv[i, j - 1])

    # ===============================================
    # Step 3: Replace near-zero values with local mean
    # ===============================================
    threshold = 0.05
    low_mask = iv < threshold
    if np.any(low_mask):
        iv_padded = np.pad(iv, 1, mode='edge')
        for i in range(nT):
            for j in range(nK):
                if low_mask[i, j]:
                    # extract 3x3 neighborhood
                    neighborhood = iv_padded[i:i+3, j:j+3]
                    iv[i, j] = np.mean(neighborhood)

    return iv



from .rbergomi import rBergomi
from .utils import bs, bsinv
import numpy as np, os, pickle
from datetime import datetime
from typing import List, Dict

def generate_surfaces(
    num_sets=1,
    forward_curves_per_set=1,
    cfg=SimulationConfig(),
    seed=42,
    randomize_grid=False,
    grid_jitter=0.25,
    save_every=200
) -> List[Dict]:
    """
    Generate implied-volatility surfaces from the true rBergomi model
    using the hybrid-scheme path generator and exact BS inversion.
    Retains the original batching and saving logic.
    """

    # --- Setup save structure ---
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    save_dir = os.path.join("data", date_str)
    os.makedirs(save_dir, exist_ok=True)

    timestamped_path = os.path.join(save_dir, f"surfaces_{time_str}.pkl")
    progress_path = "data/surfaces_progress.pkl"
    os.makedirs(os.path.dirname(progress_path), exist_ok=True)

    rng = np.random.RandomState(seed)
    results: List[Dict] = []

    # --- Base time grid ---
    n, T_max = cfg.n, cfg.T_max

    # --- Sample parameter sets ---
    param_sets = sample_param_sets_lhs(num_sets)

    for s, params in enumerate(param_sets):
        set_rng = np.random.RandomState(seed + 10_000 * s)
        eta, rho, H = float(params.eta), float(params.rho), float(params.H)
        a = H - 0.5

        # --- Initialise rBergomi simulator for this parameter world ---
        rb = rBergomi(n=n, N=cfg.M, T=T_max, a=a)

        # --- Generate Brownian increments once per world ---
        dW1 = rb.dW1()
        dW2 = rb.dW2()
        dB  = rb.dB(dW1, dW2, rho=rho)
        Y   = rb.Y(dW1)

        # --------------------------------------------------------------
        # Forward curves and grids
        # --------------------------------------------------------------
        for j in range(forward_curves_per_set):
            
            strikes_base = cfg.strikes
            maturities_base = cfg.maturities

            xi0_knots = set_rng.uniform(0.01, 0.16, size=8)
        # # knots on [0, T_max] at equal spacing
        #     K = len(xi0_knots)
        #     bin_edges = np.linspace(0.0, T_max, K + 1)  # length K+1

        #     # for each t, find bin index i with bin_edges[i] <= t < bin_edges[i+1]
        #     idx = np.searchsorted(bin_edges, rb.t.flatten(), side="right") - 1
        #     idx = np.clip(idx, 0, K - 1)
        #     xi_t = xi0_knots[idx]

            #e0 according to maturities
            T_max = maturities_base[-1]
            K = len(xi0_knots)
            # --- build bin edges that match the maturity grid ---
            # First bin goes from 0 to first maturity, next between consecutive maturities, etc.
            # If you have more knots than maturities, we interpolate extra bins proportionally.
            if K <= len(maturities_base):
                bin_edges = np.concatenate([[0.0], maturities_base])
            else:
                # interpolate to get K+1 edges over maturities range
                base_edges = np.concatenate([[0.0], maturities_base])
                target_u = np.linspace(0, 1, K + 1)
                bin_edges = np.interp(target_u, np.linspace(0, 1, len(base_edges)), base_edges)

            # --- map each simulation time t to its corresponding bin ---
            t_flat = rb.t.flatten()
            idx = np.searchsorted(bin_edges, t_flat, side="right") - 1
            idx = np.clip(idx, 0, K - 1)

            # --- assign forward variance ---
            xi_t = xi0_knots[idx]

        

            V = rb.V(Y, xi=xi_t[np.newaxis, :], eta=eta)
            S = rb.S(V, dB, S0=cfg.S0)

            for g_id in range(cfg.G):

                if randomize_grid:
                    strikes_shifted = jitter_grid(strikes_base, grid_jitter=grid_jitter, min_spacing=0.05)
                    maturities_shifted = jitter_grid(maturities_base, grid_jitter=grid_jitter, min_spacing=0.05)
                else:
                    strikes_shifted = strikes_base.copy()
                    maturities_shifted = maturities_base.copy()

                price_surf = np.zeros((len(maturities_shifted), len(strikes_shifted)))
                iv_surf = np.zeros_like(price_surf)

                for mi, Tm in enumerate(maturities_shifted):
                    t_idx = np.searchsorted(rb.t.flatten(), Tm, side="right") - 1
                    ST = S[:, t_idx]
                    F = np.mean(ST)
                    prices = np.mean(np.maximum(ST[:, None] - strikes_shifted[None, :] * cfg.S0, 0), axis=0)
                    price_surf[mi, :] = prices

                    for ki, K in enumerate(strikes_shifted):
                        try:
                            iv_surf[mi, ki] = bsinv(prices[ki], F, K, Tm)
                        except Exception:
                            iv_surf[mi, ki] = np.nan

                    iv_surf[mi, :] = np.clip(iv_surf[mi, :], 1e-4, 5.0)
                iv_surf = repair_edges_local_directional(iv_surf, maturities_shifted, strikes_shifted, t_threshold=0.35)

                # Sanity check
                if np.isnan(iv_surf).any() or np.isinf(iv_surf).any():
                    bad_dir = os.path.join("data", "debug_nans")
                    os.makedirs(bad_dir, exist_ok=True)
                    bad_path = os.path.join(bad_dir, f"bad_surface_set{s}_fwd{j}_grid{g_id}.npz")
                    np.savez_compressed(
                        bad_path,
                        price_surface=price_surf,
                        iv_surface=iv_surf,
                        strikes=strikes_shifted,
                        maturities=maturities_shifted,
                        params=np.array([eta, rho, H], dtype=float),
                        xi0_knots=xi0_knots.astype(float),
                    )
                    print(f"NaN/Inf detected in surface (set={s}, fwd={j}, grid={g_id}) {bad_path}")
                    continue

                # Record metadata
                params_record = {
                    "eta": eta,
                    "rho": rho,
                    "H": H,
                    "xi0_knots": xi0_knots.astype(float).tolist(),
                }

                results.append({
                    "set_id": s,
                    "fwd_id": j,
                    "grid_id": g_id,
                    "params": params_record,
                    "grid": {
                        "strikes": strikes_shifted.astype(float),
                        "maturities": maturities_shifted.astype(float),
                    },
                    "price_surface": price_surf,
                    "iv_surface": iv_surf,
                })

                # --- Periodic checkpoint save ---
                if len(results) % save_every == 0:
                    data = {"cfg": cfg.__dict__, "surfaces": results}
                    tmp1 = timestamped_path + ".tmp"
                    tmp2 = progress_path + ".tmp"
                    with open(tmp1, "wb") as f:
                        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                    os.replace(tmp1, timestamped_path)
                    with open(tmp2, "wb") as f:
                        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                    os.replace(tmp2, progress_path)
                    print(f"[Progress] Saved {len(results)} surfaces  {timestamped_path}")

    # --- Final save ---
    data = {"cfg": cfg.__dict__, "surfaces": results}
    tmp1 = timestamped_path + ".tmp"
    tmp2 = progress_path + ".tmp"
    with open(tmp1, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp1, timestamped_path)
    with open(tmp2, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp2, progress_path)
    print(f"[Done] Saved {len(results)} surfaces  {timestamped_path}")

    return results


def generate_fixed_surface(
    params_fixed: Dict[str, float],
    xi0_knots: np.ndarray,
    strikes: np.ndarray,
    maturities: np.ndarray,
    cfg,
    seed: int = 42
) -> Dict:
    """
    Generate an implied-volatility surface using the original rBergomi class and utils.py.
    This replaces the rDonsker approximation with the exact hybrid-scheme implementation.
    """

    np.random.seed(seed)
    eta = float(params_fixed["eta"])
    rho = float(params_fixed["rho"])
    H   = float(params_fixed["H"])
    S0  = cfg.S0
    M   = cfg.M
    n   = cfg.n
    T_max = cfg.T_max

    # Convert H to alpha = H - 0.5
    a = H - 0.5

    # Initialise rBergomi simulator
    rb = rBergomi(n=n, N=M, T=T_max, a=a)

    # Simulate correlated Brownian motions
    dW1 = rb.dW1()                       # variance driver
    dW2 = rb.dW2()                       # orthogonal driver
    dB  = rb.dB(dW1, dW2, rho=rho)       # correlated price driver

    # Construct Volterra process and variance paths
    Y = rb.Y(dW1)

    # Piecewise-constant forward variance interpolation
    #e0 equally spaced
    # K = len(xi0_knots)
    # bin_edges = np.linspace(0.0, T_max, K + 1)  # length K+1
    # # for each t, find bin index i with bin_edges[i] <= t < bin_edges[i+1]
    # idx = np.searchsorted(bin_edges, rb.t.flatten(), side="right") - 1
    # idx = np.clip(idx, 0, K - 1)
    # xi_t = xi0_knots[idx]

    #e0 according to maturities
    T_max = maturities[-1]
    K = len(xi0_knots)
    # --- build bin edges that match the maturity grid ---
    # First bin goes from 0 to first maturity, next between consecutive maturities, etc.
    # If you have more knots than maturities, we interpolate extra bins proportionally.
    if K <= len(maturities):
        bin_edges = np.concatenate([[0.0], maturities])
    else:
        # interpolate to get K+1 edges over maturities range
        base_edges = np.concatenate([[0.0], maturities])
        target_u = np.linspace(0, 1, K + 1)
        bin_edges = np.interp(target_u, np.linspace(0, 1, len(base_edges)), base_edges)

    # --- map each simulation time t to its corresponding bin ---
    t_flat = rb.t.flatten()
    idx = np.searchsorted(bin_edges, t_flat, side="right") - 1
    idx = np.clip(idx, 0, K - 1)

    # --- assign forward variance ---
    xi_t = xi0_knots[idx]


    V = rb.V(Y, xi=xi_t[np.newaxis, :], eta=eta)

    # Simulate price paths
    S = rb.S(V, dB, S0=S0)

    # Build IV surface on provided strike/maturity grid
    iv_surface = np.zeros((len(maturities), len(strikes)))
    price_surface = np.zeros_like(iv_surface)

    for iT, T in enumerate(maturities):
        t_idx = min(int(T * n), rb.s)
        ST = S[:, t_idx]
        F = np.mean(ST)

        # Price calls for all strikes
        prices = np.mean(np.maximum(ST[:, None] - strikes[None, :] * S0, 0), axis=0)
        price_surface[iT, :] = prices

        # Implied vols via Brent root-finder (from utils.py)
        for iK, K in enumerate(strikes):
            try:
                iv_surface[iT, iK] = bsinv(prices[iK], F, K, T)
            except Exception:
                iv_surface[iT, iK] = np.nan

        # numerical clipping
        iv_surface[iT, :] = np.clip(iv_surface[iT, :], 1e-4, 5.0)
    iv_surface = repair_edges_local_directional(iv_surface, maturities, strikes, t_threshold=0.35)

    return {
        "params": {
            "eta": eta,
            "rho": rho,
            "H": H,
            "xi0_knots": xi0_knots.tolist(),
        },
        "grid": {
            "strikes": strikes.astype(float),
            "maturities": maturities.astype(float),
        },
        "price_surface": price_surface,
        "iv_surface": iv_surface,
    }
