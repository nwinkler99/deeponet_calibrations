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
    dtype: np.dtype = np.float32  # <--- global dtype control

    def __post_init__(self):
        if self.strikes is None:
            self.strikes = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
        if self.maturities is None:
            self.maturities = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1])


def sample_param_sets_lhs(num_sets: int, rng: np.random.RandomState) -> List[RBergomiParams]:
    """Sample (eta, rho, H) + random xi0_knots via Latin Hypercube, fully tied to rng."""
    sampler = qmc.LatinHypercube(d=3, seed=rng)
    sample = sampler.random(num_sets)

    lower = np.array([0.5, -0.95, 0.025])
    upper = np.array([4.0, -0.1, 0.5])
    scaled = qmc.scale(sample, lower, upper)

    param_sets: List[RBergomiParams] = []
    for i in range(num_sets):
        param_sets.append(
            RBergomiParams(
                eta=float(scaled[i, 0]),
                rho=float(scaled[i, 1]),
                H=float(scaled[i, 2]),
            )
        )
    return param_sets


def jitter_grid(base_grid, grid_jitter=0.25, min_spacing=0.1):
    """Randomly perturb a base grid while keeping minimum spacing; returns float64, cast by caller."""
    base_grid = np.sort(np.array(base_grid, dtype=float))
    n = len(base_grid)
    lo, hi = base_grid[0], base_grid[-1]
    total_range = hi - lo
    total_budget = grid_jitter * total_range

    grid_shifted = base_grid.copy()
    for _ in range(100):
        signs = np.random.choice([-1, 1], size=n)
        weights = np.random.uniform(0, 1, size=n)
        weights /= np.sum(weights) + 1e-12
        jitter = signs * weights * total_budget
        grid_shifted = np.clip(np.sort(base_grid + jitter), lo, hi)
        if np.all(np.diff(grid_shifted) > min_spacing):
            return np.round(grid_shifted, 6)
    return np.round(grid_shifted, 6)

import numpy as np
from scipy.ndimage import convolve

def repair_edges_local_directional(iv_surface: np.ndarray,
                                   maturities: np.ndarray,
                                   strikes: np.ndarray,
                                   t_threshold: float = 0.35,
                                   dtype: np.dtype = np.float32,
                                   min_floor: float = 0.05) -> np.ndarray:
    """
    Lightweight, local edge stabilization for short maturities.
    - Fills NaNs row/col-wise.
    - Lifts left/right edges of short-maturity rows toward neighbors.
    - Smoothly replaces values < min_floor with weighted 3×3 Gaussian-style local mean.
    """
    iv = np.array(iv_surface, dtype=dtype, copy=True)
    nT, nK = iv.shape

    # --- Row-wise NaN fill
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
                iv[i] = np.nan

    # --- Column-wise fill
    if np.isnan(iv).any():
        col_means = np.nanmean(iv, axis=0)
        for j in range(nK):
            nan_idx = np.isnan(iv[:, j])
            if np.any(nan_idx):
                iv[nan_idx, j] = col_means[j]
    iv = np.nan_to_num(iv, nan=np.nanmean(iv)).astype(dtype)

    #--- Directional edge correction for short maturities
    short_idx = np.where(maturities <= t_threshold)[0]
    for i in reversed(short_idx):
        row = iv[i]

        # ---- Left side: start from 3rd inner strike and move outward
        for j in range(2, -1, -1):  # indices [2, 1, 0]
            row[j] = max(row[j], row[j+1])

        # ---- Right side: start from 3rd inner strike from right and move outward
        for j in range(nK-3, nK):  # indices [nK-3, nK-2, nK-1]
            if j < nK-1:  # skip the last since j+1 would be out of bounds
                row[j+1] = max(row[j+1], row[j])

        iv[i] = row

    # --- Weighted local-mean correction for values < min_floor
    mask_rows = maturities <= 1000
    if np.any(mask_rows):
        iv_short = iv[mask_rows]

        # Gaussian-like kernel (center-heavy)
        kernel = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], dtype=np.float32)
        kernel /= kernel.sum()

        local_mean = convolve(iv_short, kernel, mode="reflect")

        mask_low = iv_short < min_floor
        iv_short[mask_low] = local_mean[mask_low]
        iv[mask_rows] = iv_short

    return iv.astype(dtype)




from .rbergomi import rBergomi
from .utils import bs, bsinv
import numpy as np, os, pickle
from datetime import datetime
from typing import List, Dict

import os
import numpy as np
from datetime import datetime
from typing import List, Dict
from scipy.stats import norm

# ---- Helpers (forward-form Black–Scholes) ------------------------------------
def bs_vega(F: float, K: float, T: float, sigma: float) -> float:
    """Black–Scholes vega (∂Price/∂Vol) with forward F, strike K, maturity T."""
    if sigma <= 0 or T <= 0 or not np.isfinite(sigma) or F <= 0 or K <= 0:
        return np.nan
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * np.sqrt(T))
    return F * norm.pdf(d1) * np.sqrt(T)

def bs_call_vec_pathwise(F_path: np.ndarray, K_abs: float, T: float, sigma_path: np.ndarray) -> np.ndarray:
    """
    Pathwise Black–Scholes call price with forward F_path (M,), strike K_abs (scalar),
    maturity T, and per-path vol sigma_path (M,). Returns (M,).
    """
    eps = 1e-12
    Fp = np.maximum(F_path, eps)
    sig = np.maximum(sigma_path, eps)
    d1 = (np.log(Fp / K_abs) + 0.5 * sig * sig * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return Fp * norm.cdf(d1) - K_abs * norm.cdf(d2)

def bs_put_vec_pathwise(F_path, K_abs, T, sigma_path):
    eps = 1e-12
    Fp  = np.maximum(F_path, eps)
    sig = np.maximum(sigma_path, eps)
    d1  = (np.log(Fp / K_abs) + 0.5 * sig*sig * T) / (sig * np.sqrt(T))
    d2  = d1 - sig * np.sqrt(T)
    # forward-measure (undiscounted) BS put
    return K_abs * norm.cdf(-d2) - Fp * norm.cdf(-d1)

# ==============================
# Main generator with CMC / martingale CV
# ==============================
def bs_call_vec_pathwise(F_path, K_abs, T, sigma_path):
    # forward-measure BS call, vectorized over paths
    eps = 1e-12
    Fp  = np.maximum(F_path, eps)
    sig = np.maximum(sigma_path, eps)
    rt  = np.sqrt(T)
    d1  = (np.log(Fp / K_abs) + 0.5 * sig*sig * T) / (sig * rt)
    d2  = d1 - sig * rt
    return Fp * norm.cdf(d1) - K_abs * norm.cdf(d2)

def generate_surfaces(
    num_sets=1,
    forward_curves_per_set=1,
    cfg=None,
    seed=42,
    randomize_grid=False,
    grid_jitter=0.25,
    save_every=200,
) -> List[Dict]:
    """
    Generate implied-volatility surfaces from the rBergomi model using Monte Carlo simulation.
    Uses rho-aware conditional Monte Carlo (martingale control variate):
      per path and maturity, use BS with F_path = S0*exp(rho*∫sqrt(V)dW - 0.5*rho^2∫Vds)
      and sigma_path^2 = (1-rho^2)*∫Vds / T.
    Computes iv_rel_error as 1.96 * SE_iv / iv.
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")

    save_dir = os.path.join("data", date_str if randomize_grid else "fixed_longrun")
    os.makedirs(save_dir, exist_ok=True)

    results: List[Dict] = []
    rng = np.random.RandomState(seed)
    n, T_max = cfg.n, cfg.T_max

    param_sets = sample_param_sets_lhs(num_sets, rng)

    for s, params in enumerate(param_sets):
        set_seed = rng.randint(0, 2**31 - 1)
        set_rng = np.random.RandomState(set_seed)

        eta, rho, H = float(params.eta), float(params.rho), float(params.H)
        a = H - 0.5

        # --- Simulator
        rb = rBergomi(n=n, N=cfg.M, T=T_max, a=a)
        t_grid = rb.t.flatten().astype(cfg.dtype)                # (n,)
        dt = np.diff(t_grid, prepend=cfg.dtype(0.0)).astype(cfg.dtype)  # (n,)
        dW1, dW2 = rb.dW1(), rb.dW2()
        dB, Y = rb.dB(dW1, dW2, rho=rho), rb.Y(dW1)

        # volatility driver increments W^(1): take correct component
        dW_vol = dW1[..., 0] if dW1.ndim == 3 else dW1          # (M, n-1)
        dW_vol = dW_vol.astype(cfg.dtype, copy=False)

        for j in range(forward_curves_per_set):
            strikes_base = cfg.strikes.astype(cfg.dtype)
            maturities_base = cfg.maturities.astype(cfg.dtype)
            xi0_knots = set_rng.uniform(0.01, 0.16, size=len(maturities_base)).astype(cfg.dtype)

            # --- map xi0_knots to sim grid
            base_edges = np.concatenate([[0.0], maturities_base.astype(float)])
            target_u = np.linspace(0, 1, len(xi0_knots) + 1)
            bin_edges = np.interp(target_u, np.linspace(0, 1, len(base_edges)), base_edges)
            idx = np.searchsorted(bin_edges, t_grid.astype(float), side="right") - 1
            idx = np.clip(idx, 0, len(xi0_knots) - 1)
            xi_t = xi0_knots[idx]

            V = rb.V(Y, xi=xi_t[np.newaxis, :], eta=eta).astype(cfg.dtype)   # (M, n)
            S = rb.S(V, dB, S0=cfg.S0).astype(cfg.dtype)                      # (M, n)

            # Precompute cumulative integrals for CMC (left-point rules)
            V_left = V[:, :-1].astype(cfg.dtype, copy=False)                  # (M, n-1)
            dt_incr = dt[1:]                                                  # (n-1,)
            sqrtV_left = np.sqrt(np.maximum(V_left, cfg.dtype(0.0)))          # (M, n-1)

            # I1 = ∫ sqrt(V) dW^(1), I2 = ∫ V ds
            I1_cum = np.cumsum(sqrtV_left * dW_vol, axis=1, dtype=cfg.dtype)  # (M, n-1)
            I2_cum = np.cumsum(V_left * dt_incr[None, :], axis=1, dtype=cfg.dtype)  # (M, n-1)

            for g_id in range(cfg.G):
                if randomize_grid:
                    strikes_shifted = np.array(
                        jitter_grid(strikes_base, grid_jitter=grid_jitter, min_spacing=0.05),
                        dtype=cfg.dtype,
                    )
                    maturities_shifted = np.array(
                        jitter_grid(maturities_base, grid_jitter=grid_jitter, min_spacing=0.05),
                        dtype=cfg.dtype,
                    )
                else:
                    strikes_shifted = strikes_base.copy()
                    maturities_shifted = maturities_base.copy()

                nT, nK = len(maturities_shifted), len(strikes_shifted)
                iv_surf    = np.zeros((nT, nK), dtype=cfg.dtype)
                iv_relerr  = np.zeros_like(iv_surf)

                S0 = cfg.dtype(cfg.S0)

                for mi, Tm in enumerate(maturities_shifted):
                    # maturity index on sim grid
                    t_idx = np.searchsorted(t_grid.astype(float), float(Tm), side="right") - 1
                    t_idx = int(np.clip(t_idx, 0, S.shape[1] - 1))
                    T_float = float(Tm)
                    if T_float <= 0:
                        iv_surf[mi, :], iv_relerr[mi, :] = np.nan, np.nan
                        continue

                    # pull integrals up to t_idx
                    if t_idx == 0:
                        I1 = np.zeros((V.shape[0],), dtype=cfg.dtype)
                        I2 = np.zeros((V.shape[0],), dtype=cfg.dtype)
                    else:
                        I1 = I1_cum[:, t_idx - 1]
                        I2 = I2_cum[:, t_idx - 1]

                    # Conditional forward and vol per path (rho-aware)
                    F_path = S0 * np.exp(cfg.dtype(rho) * I1 - cfg.dtype(0.5) * (cfg.dtype(rho) * cfg.dtype(rho)) * I2)
                    sigma_path = np.sqrt(np.maximum((cfg.dtype(1.0) - cfg.dtype(rho) * cfg.dtype(rho)) * I2 / Tm,
                                                    cfg.dtype(1e-16)))

                    for ki, K_ in enumerate(strikes_shifted):
                        K_abs = K_ * S0

                        # Choose OTM side: right wing (K > S0) -> call, left wing -> put (via parity)
                        use_call = bool(float(K_abs) > float(S0))

                        call_vals = bs_call_vec_pathwise(F_path, float(K_abs), T_float, sigma_path).astype(cfg.dtype)

                        if use_call:
                            cond_vals = call_vals
                            o_flag = 'call'
                        else:
                            # put via pathwise parity: P = C - (F_path - K)
                            cond_vals = (call_vals - (F_path - K_abs)).astype(cfg.dtype)
                            o_flag = 'put'

                        # Mean and SE across paths (of conditional expectations)
                        price_cmc = cfg.dtype(np.mean(cond_vals))
                        se_price  = cfg.dtype(np.std(cond_vals, ddof=1)) / cfg.dtype(np.sqrt(cfg.M))

                        # guard tiny prices (ill-posed inversion)
                        if not np.isfinite(price_cmc) or price_cmc < cfg.dtype(1e-8):
                            iv_surf[mi, ki] = np.nan
                            iv_relerr[mi, ki] = np.nan
                            continue

                        # Invert with correct option flag; propagate SE via vega
                        try:
                            iv_val = bsinv(float(price_cmc), float(S0), float(K_abs), T_float, o=o_flag)
                            iv_surf[mi, ki] = cfg.dtype(iv_val)
                            vega = bs_vega(float(S0), float(K_abs), T_float, float(iv_val))
                            if np.isfinite(vega) and vega > 1e-12 and iv_val > 1e-12:
                                se_iv = se_price / cfg.dtype(vega)
                                iv_relerr[mi, ki] = cfg.dtype(1.96) * se_iv / cfg.dtype(iv_val)
                            else:
                                iv_relerr[mi, ki] = np.nan
                        except Exception:
                            iv_surf[mi, ki] = np.nan
                            iv_relerr[mi, ki] = np.nan

                    iv_surf[mi, :] = np.clip(iv_surf[mi, :], cfg.dtype(1e-4), cfg.dtype(5.0))

                # optional smoothing/repair (after computing errors

                results.append(
                    {
                        "set_id": s,
                        "fwd_id": j,
                        "grid_id": g_id,
                        "params": {
                            "eta": float(eta),
                            "rho": float(rho),
                            "H": float(H),
                            "xi0_knots": xi0_knots.astype(cfg.dtype).tolist(),
                        },
                        "grid": {
                            "strikes": strikes_shifted.astype(cfg.dtype),
                            "maturities": maturities_shifted.astype(cfg.dtype),
                        },
                        "iv_surface": iv_surf,
                        "iv_rel_error": iv_relerr,
                    }
                )

    return results








# ---- Main --------------------------------------------------------------------
# ---- Main --------------------------------------------------------------------
def generate_fixed_surface(param_set: Dict,
                           xi0_knots: np.ndarray,
                           strikes: np.ndarray,
                           maturities: np.ndarray,
                           cfg: SimulationConfig,
                           seed: int = 123) -> Dict:
    """
    Generate one IV surface for fixed parameters and a fixed (K, T) grid.
    Uses conditional Monte Carlo (martingale CV with rho) and returns
    iv_surface and iv_rel_error (95% relative CI half-width).
    Fully dtype-consistent with cfg.dtype.
    """
    dtype = cfg.dtype
    eta = dtype(param_set["eta"])
    rho = dtype(param_set["rho"])
    H   = dtype(param_set["H"])
    a   = H - dtype(0.5)

    strikes   = np.array(strikes, dtype=dtype)
    maturities = np.array(maturities, dtype=dtype)
    xi0_knots = np.array(xi0_knots, dtype=dtype)

    # Simulator
    rb = rBergomi(n=cfg.n, N=cfg.M, T=float(maturities[-1]), a=float(a))
    t_grid = rb.t.flatten().astype(dtype)             # (n,)
    dt = np.diff(t_grid, prepend=dtype(0.0))          # (n,)
    dW1, dW2 = rb.dW1(), rb.dW2()                     # increments for volatility BM W and orthogonal BM
    dB, Y = rb.dB(dW1, dW2, rho=float(rho)), rb.Y(dW1)

    # Map xi0_knots to time grid using maturity bins
    Kk = len(xi0_knots)
    if Kk <= len(maturities):
        bin_edges = np.concatenate([[0.0], maturities.astype(float)])
    else:
        base_edges = np.concatenate([[0.0], maturities.astype(float)])
        target_u = np.linspace(0, 1, Kk + 1)
        bin_edges = np.interp(target_u, np.linspace(0, 1, len(base_edges)), base_edges)

    idx = np.searchsorted(bin_edges, t_grid.astype(float), side="right") - 1
    idx = np.clip(idx, 0, Kk - 1)
    xi_t = xi0_knots[idx]

    V = rb.V(Y, xi=xi_t[np.newaxis, :], eta=float(eta)).astype(dtype)   # (M, n)
    S = rb.S(V, dB, S0=dtype(cfg.S0)).astype(dtype)                     # (M, n)

    # ----- Conditional MC ingredients with rho --------------------------------
    # Left-point Ito sum for I1 = ∫ sqrt(V) dW  and rectangle rule for I2 = ∫ V ds
    dW_vol = dW1[..., 0] if dW1.ndim == 3 else dW1
    dW_vol = dW_vol.astype(dtype, copy=False)        # (M, n-1)
    V_left = V[:, :-1].astype(dtype, copy=False)     # (M, n-1)
    dt_incr = dt[1:].astype(dtype)                   # (n-1,)

    sqrtV_left = np.sqrt(np.maximum(V_left, dtype(0.0)))       # (M, n-1)
    I1_cum = np.cumsum(sqrtV_left * dW_vol, axis=1, dtype=dtype)  # (M, n-1)
    I2_cum = np.cumsum(V_left * dt_incr[None, :], axis=1, dtype=dtype)  # (M, n-1)

    # Build surfaces
    nT, nK = len(maturities), len(strikes)
    price_surf = np.zeros((nT, nK), dtype=dtype)
    iv_surf    = np.zeros_like(price_surf)
    iv_relerr  = np.zeros_like(price_surf)

    S0 = dtype(cfg.S0)

    for mi, Tm in enumerate(maturities):
        # maturity index
        t_idx = np.searchsorted(t_grid, float(Tm), side="right") - 1
        t_idx = int(np.clip(t_idx, 0, S.shape[1] - 1))
        T_float = float(Tm)

        if T_float <= 0:
            price_surf[mi, :], iv_surf[mi, :], iv_relerr[mi, :] = np.nan, np.nan, np.nan
            continue

        if t_idx == 0:
            I1 = np.zeros((V.shape[0],), dtype=dtype)
            I2 = np.zeros((V.shape[0],), dtype=dtype)
        else:
            I1 = I1_cum[:, t_idx - 1]
            I2 = I2_cum[:, t_idx - 1]

        # Conditional forward and conditional vol per path (preserves skew via rho-term)
        F_path = S0 * np.exp(rho * I1 - dtype(0.5) * (rho * rho) * I2)
        sigma_path = np.sqrt(np.maximum((dtype(1.0) - rho * rho) * I2 / Tm, dtype(1e-16)))

        for ki, K_ in enumerate(strikes):
            K_abs = K_ * S0

            # Pathwise conditional BS prices; average gives CMC estimator
            is_call = (K_abs >= float(S0))   # OTM call region
            if is_call:
                cond_vals = bs_call_vec_pathwise(F_path, K_abs, T_float, sigma_path).astype(dtype)
                o_flag = 'call'
            else:
                # either direct put helper:
                cond_vals = bs_put_vec_pathwise(F_path, K_abs, T_float, sigma_path).astype(dtype)
                # or via parity:
                # cond_vals = (bs_call_vec_pathwise(F_path, K_abs, T_float, sigma_path) - (F_path - K_abs)).astype(dtype)
                o_flag = 'put'

            price_cmc = dtype(np.mean(cond_vals))
            se_price  = dtype(np.std(cond_vals, ddof=1)) / dtype(np.sqrt(cfg.M))
            price_surf[mi, ki] = price_cmc

            try:
                iv_val = bsinv(float(price_cmc), float(S0), float(K_abs), float(Tm), o=o_flag)
                iv_surf[mi, ki] = dtype(iv_val)
                vega = bs_vega(float(S0), float(K_abs), float(Tm), float(iv_val))
                if np.isfinite(vega) and vega > 1e-12 and iv_val > 1e-12:
                    se_iv = se_price / dtype(vega)
                    iv_relerr[mi, ki] = dtype(1.96) * se_iv / dtype(iv_val)
                else:
                    iv_relerr[mi, ki] = np.nan
            except Exception:
                iv_surf[mi, ki] = np.nan
                iv_relerr[mi, ki] = np.nan

        iv_surf[mi, :] = np.clip(iv_surf[mi, :], dtype(1e-4), dtype(5.0))

    # Optional repairs (after computing errors)
    #iv_surf = repair_edges_local_directional(iv_surf, maturities, strikes,
                                             #t_threshold=0.35, dtype=dtype)

    # Final sanity
    if np.isnan(iv_surf).any() or np.isinf(iv_surf).any():
        bad_dir = os.path.join("data", "debug_nans")
        os.makedirs(bad_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(bad_dir, f"bad_fixed_surface_seed{seed}.npz"),
            price_surface=price_surf,
            iv_surface=iv_surf,
            iv_rel_error=iv_relerr,
            strikes=strikes,
            maturities=maturities,
            params=np.array([eta, rho, H], dtype=dtype),
            xi0_knots=xi0_knots,
        )
        print("NaN/Inf detected in fixed surface; saved debug npz.")

    return {
        "params": {"eta": float(eta), "rho": float(rho), "H": float(H),
                   "xi0_knots": xi0_knots.astype(dtype).tolist()},
        "grid": {"strikes": strikes.astype(dtype), "maturities": maturities.astype(dtype)},
        "price_surface": price_surf,
        "iv_surface": iv_surf,
        "iv_rel_error": iv_relerr,
    }
