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
            self.strikes = np.array([0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5])
        if self.maturities is None:
            self.maturities = np.array([0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2.0])

# -------------------------------------------------------------
# rDonsker fractional Brownian motion simulator
# -------------------------------------------------------------

def fBm_path_rDonsker(grid_points: int, M: int, H: float, T: float) -> np.ndarray:
    """
    rDonsker fractional Brownian motion paths with correct variance scaling.
    Ensures Var[X_t] ≈ t^{2H}.
    """
    dt = 1 / (grid_points - 1)
    dw = np.random.normal(0.0, (dt**H), size=(M, grid_points - 1))
    i = np.arange(1, grid_points)
    # "Optimal" Donsker kernel (from Bayer, Friz, Gatheral 2016)
    opt_k = ((i**(2*H) - (i-1)**(2*H)) / (2*H)) ** 0.5

    Y = np.zeros((M, grid_points))
    for m in range(M):
        conv = np.convolve(opt_k, dw[m, :])[:grid_points - 1]
        Y[m, 1:] = conv

    # empirically normalize to Var[X_T] = T^{2H}
    scale = T**H*np.sqrt(2*H)
    Y *= scale
    return Y

# -------------------------------------------------------------
# Piecewise constant forward variance curve
# -------------------------------------------------------------

def build_xi0_piecewise_constant(knots: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    T_max = t_grid[-1]
    knot_times = np.linspace(0.0, T_max, len(knots))
    idx = np.digitize(t_grid, knot_times) - 1
    idx = np.clip(idx, 0, len(knots) - 1)
    return knots[idx]

# -------------------------------------------------------------
# Price simulation with correlated Brownian motions
# -------------------------------------------------------------

def simulate_price_paths(S0: float, t: np.ndarray, X: np.ndarray, dw: np.ndarray, dW_perp: np.ndarray,
                         xi0_t: np.ndarray, eta: float, rho: float, H: float) -> np.ndarray:
    n = min(X.shape[1], t.shape[0], dw.shape[1] + 1, xi0_t.shape[0])
    X, t, xi0_t = X[:, :n], t[:n], xi0_t[:n]
    dw, dW_perp = dw[:, :n-1], dW_perp[:, :n-1]
    M = X.shape[0]
    dt = np.mean(np.diff(t))
    t2H = np.power(t, 2.0 * H)
    exp_etaX = np.exp(eta * X - 0.5 * eta**2 *t2H)
    v = xi0_t * exp_etaX
    dW_S = rho * dw + np.sqrt(max(1.0 - rho*rho, 0.0)) * dW_perp
    logS = np.zeros((M, n))
    logS[:, 0] = np.log(S0)
    logS[:, 1:] = logS[:, [0]] + np.cumsum(-0.5*v[:, :-1]*dt + np.sqrt(np.maximum(v[:, :-1], 0.0))*dW_S, axis=1)
    return np.exp(logS)


# -------------------------------------------------------------
# Plain OTM pricing (no control variate)
# -------------------------------------------------------------

def price_calls_plain(ST: np.ndarray, S0: float, Ks: np.ndarray) -> np.ndarray:
    return np.mean(np.maximum(ST[:, None] - Ks[None, :], 0.0), axis=0)

def price_puts_plain(ST: np.ndarray, S0: float, Ks: np.ndarray) -> np.ndarray:
    return np.mean(np.maximum(Ks[None, :] - ST[:, None], 0.0), axis=0)

def price_otm_plain(ST: np.ndarray, S0: float, Ks: np.ndarray) -> np.ndarray:
    calls = price_calls_plain(ST, S0, Ks)
    puts = price_puts_plain(ST, S0, Ks)
    return np.where(Ks >= S0, calls, puts)
    
# -------------------------------------------------------------
# Black–Scholes pricing + implied vol inversion (robust)
# -------------------------------------------------------------

def bs_call_price(S0, K, T, vol):
    if vol <= 0.0 or T <= 0.0:
        return max(S0 - K, 0.0)
    sT = np.sqrt(T)
    d1 = (np.log(S0 / K) + 0.5 * vol**2 * T) / (vol * sT)
    d2 = d1 - vol * sT
    return S0 * norm.cdf(d1) - K * norm.cdf(d2)

def bs_put_price(S0, K, T, vol):
    if vol <= 0.0 or T <= 0.0:
        return max(K - S0, 0.0)
    sT = np.sqrt(T)
    d1 = (np.log(S0 / K) + 0.5 * vol**2 * T) / (vol * sT)
    d2 = d1 - vol * sT
    return K * norm.cdf(-d2) - S0 * norm.cdf(-d1)

def implied_vol_from_price_otm(S0, K, T, price, tol=1e-7):
    if price < 1e-8:
        return np.nan
    price_func = bs_call_price if K >= S0 else bs_put_price
    a, b = 1e-8, 10
    fa = price_func(S0, K, T, a) - price
    fb = price_func(S0, K, T, b) - price
    if fa * fb > 0:
        return np.nan
    for _ in range(80):
        m = 0.5 * (a + b)
        fm = price_func(S0, K, T, m) - price
        if np.sign(fm) == np.sign(fa):
            a, fa = m, fm
        else:
            b, fb = m, fm
        if abs(b - a) < tol:
            break
    vol = max(0.0, 0.5 * (a + b))
    return np.clip(vol, 0.01, 3.0)

def surface_implied_vols_otm(S0, Ks, T, prices):
    ivs = [implied_vol_from_price_otm(S0, float(K), float(T), float(p)) for K, p in zip(Ks, prices)]
    return np.array(ivs)


import numpy as np

import numpy as np

def fill_nans_edgewise(
    arr: np.ndarray,
    strikes: np.ndarray = None,
    maturities: np.ndarray = None
) -> np.ndarray:
    """
    Fill NaNs across maturities and strikes by log-linear interpolation + trend-based extrapolation.
    Then lift artificial valleys at the edges for a smooth, realistic surface.

    All interpolation/extrapolation is done in log-vol space.
    The implementation is robust against duplicate coordinates and zero-length segments.
    """

    arr_out = arr.copy()
    orig_nan = np.isnan(arr_out)
    if not np.any(orig_nan):
        return arr_out

    n_strikes, n_mats = arr_out.shape
    mat_coords = np.asarray(maturities, dtype=float) if maturities is not None else np.arange(n_mats, dtype=float)
    strike_coords = np.asarray(strikes, dtype=float) if strikes is not None else np.arange(n_strikes, dtype=float)

    # -------------------------------------------------------------------------
    # 1D interpolation + safe extrapolation
    # -------------------------------------------------------------------------
    def _interp_extrap(x_all, x_known, y_known, eps=1e-12):
        """Safe linear interpolation + slope-based extrapolation in log-space."""
        interp = np.empty_like(x_all, dtype=float)
        interp[:] = np.nan

        # remove NaNs and duplicates in x_known
        mask = ~np.isnan(x_known)
        x_known = np.asarray(x_known[mask], float)
        y_known = np.asarray(y_known[mask], float)
        if x_known.size == 0:
            return np.full_like(x_all, np.nan)
        # if duplicates exist, keep unique with mean y
        uniq_x, idx = np.unique(x_known, return_inverse=True)
        if uniq_x.size < x_known.size:
            y_avg = np.zeros_like(uniq_x)
            counts = np.zeros_like(uniq_x)
            for i, xi in enumerate(idx):
                y_avg[xi] += y_known[i]
                counts[xi] += 1
            y_known = y_avg / np.maximum(counts, 1)
            x_known = uniq_x

        if len(x_known) == 1:
            interp[:] = y_known[0]
            return interp

        interp[:] = np.interp(x_all, x_known, y_known)

        # safe slope helper
        def safe_slope(y2, y1, x2, x1):
            dx = x2 - x1
            if abs(dx) < eps:
                return 0.0
            return (y2 - y1) / dx

        # left extrapolation
        slope_left = safe_slope(y_known[1], y_known[0], x_known[1], x_known[0])
        left_mask = x_all < x_known[0]
        interp[left_mask] = y_known[0] + slope_left * (x_all[left_mask] - x_known[0])

        # right extrapolation
        slope_right = safe_slope(y_known[-1], y_known[-2], x_known[-1], x_known[-2])
        right_mask = x_all > x_known[-1]
        interp[right_mask] = y_known[-1] + slope_right * (x_all[right_mask] - x_known[-1])

        # optional numeric sanity clipping (avoid insane extrapolations)
        interp = np.clip(interp, np.nanmin(y_known) - 5.0 * abs(np.nanstd(y_known)),
                         np.nanmax(y_known) + 5.0 * abs(np.nanstd(y_known)))
        return interp

    # -------------------------------------------------------------------------
    # Pass 1: across maturities (per strike)
    # -------------------------------------------------------------------------
    mat_fill = np.full_like(arr_out, np.nan)
    for i in range(n_strikes):
        row = arr_out[i, :]
        known = ~np.isnan(row)
        if np.count_nonzero(known) == 0:
            continue
        known_x = mat_coords[known]
        known_y = np.log(np.maximum(row[known], 1e-12))
        interp_log = _interp_extrap(mat_coords, known_x, known_y)
        mat_fill[i, :] = np.exp(interp_log)

    # -------------------------------------------------------------------------
    # Pass 2: across strikes (per maturity)
    # -------------------------------------------------------------------------
    strike_fill = np.full_like(arr_out, np.nan)
    for j in range(n_mats):
        col = arr_out[:, j]
        known = ~np.isnan(col)
        if np.count_nonzero(known) == 0:
            continue
        known_x = strike_coords[known]
        known_y = np.log(np.maximum(col[known], 1e-12))
        interp_log = _interp_extrap(strike_coords, known_x, known_y)
        strike_fill[:, j] = np.exp(interp_log)

    # -------------------------------------------------------------------------
    # Combine: choose larger estimate for originally missing entries
    # -------------------------------------------------------------------------
    combined = np.maximum(mat_fill, strike_fill)
    arr_out[orig_nan] = combined[orig_nan]

    # -------------------------------------------------------------------------
    # Edge-valley lifting pass
    # -------------------------------------------------------------------------
    def _lift_edge_valleys(surface: np.ndarray, axis: int = 1) -> np.ndarray:
        out = surface.copy()
        if axis == 1:  # across maturities
            for i in range(out.shape[0]):
                row = out[i, :]
                if len(row) < 3 or np.any(np.isnan(row)):
                    continue
                if row[0] < row[1] + (row[1] - row[2]):
                    out[i, 0] = row[1] + (row[1] - row[2])
                if row[1] < (row[0] + row[2]) / 2:
                    out[i, 1] = (row[0] + row[2]) / 2
                if row[-1] < row[-2] + (row[-2] - row[-3]):
                    out[i, -1] = row[-2] + (row[-2] - row[-3])
        else:  # across strikes
            for j in range(out.shape[1]):
                col = out[:, j]
                if len(col) < 3 or np.any(np.isnan(col)):
                    continue
                if col[0] < col[1] + (col[1] - col[2]):
                    out[0, j] = col[1] + (col[1] - col[2])
                if col[1] < (col[0] + col[2]) / 2:
                    out[1, j] = (col[0] + col[2]) / 2
                if col[-1] < col[-2] + (col[-2] - col[-3]):
                    out[-1, j] = col[-2] + (col[-2] - col[-3])
        return out

    arr_out = _lift_edge_valleys(arr_out, axis=1)
    arr_out = _lift_edge_valleys(arr_out, axis=0)

    return arr_out

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



# ---------------------------------------------------------------------
# Main surface generation
# ---------------------------------------------------------------------
def generate_surfaces(
    num_sets=1,
    forward_curves_per_set=1,
    cfg=SimulationConfig(),
    seed=42,
    randomize_grid=False,
    grid_jitter=0.5,
    save_every=200
) -> List[Dict]:
    """
    Generate implied-volatility surfaces from Rough Bergomi simulations.
    Each parameter set (η, ρ, H) defines one "model world".
    Within that world, multiple forward-variance curves (xi₀_t) are simulated
    using shared stochastic paths (dw, dW_perp, X).
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

    np.random.seed(seed)
    results = []

    # --- Base time grid ---
    n, T_max = cfg.n, cfg.T_max
    base_t = np.linspace(0.0, T_max, n)
    dt = base_t[1] - base_t[0]

    # --- Sample parameters via Latin Hypercube ---
    param_sets = sample_param_sets_lhs(num_sets)

    for s, params in enumerate(param_sets):
        np.random.seed(seed + 10_000 * s)

        # --------------------------------------------------------------
        # Shared Brownian increments & fBm volatility driver per set
        # --------------------------------------------------------------
        M = int(2 * round(cfg.M / 2))
        M_half = M // 2
        dw_half = np.random.normal(0.0, np.sqrt(dt), size=(M_half, n - 1))
        dW_perp_half = np.random.normal(0.0, np.sqrt(dt), size=(M_half, n - 1))
        dw = np.vstack([dw_half, -dw_half])
        dW_perp = np.vstack([dW_perp_half, -dW_perp_half])

        X = fBm_path_rDonsker(n, M, params.H, T_max)

        # --------------------------------------------------------------
        # Generate multiple forward curves under same model & randomness
        # --------------------------------------------------------------
        for j in range(forward_curves_per_set):
            # unique forward variance curve (market regime)
            xi0_knots = np.random.uniform(0.01, 0.16, size=8)
            xi0_t = build_xi0_piecewise_constant(xi0_knots, base_t)

            # simulate price paths under this forward curve
            S = simulate_price_paths(
                cfg.S0, base_t, X, dw, dW_perp,
                xi0_t, params.eta, params.rho, params.H
            )

            strikes_base = cfg.strikes
            maturities_base = cfg.maturities

            for g_id in range(cfg.G):

                # Optional randomized grids
                if randomize_grid:
                    dK = strikes_base[1] - strikes_base[0]
                    dT = maturities_base[-1] - maturities_base[-2]
                    strikes_shifted = np.clip(
                        strikes_base + np.random.uniform(-grid_jitter * dK, grid_jitter * dK, len(strikes_base)),
                        0.5, 1.5
                    )
                    maturities_shifted = np.clip(
                        maturities_base + np.random.uniform(-grid_jitter * dT, grid_jitter * dT, len(maturities_base)),
                        0.01, T_max
                    )
                    maturities_shifted = np.sort(maturities_shifted)
                else:
                    strikes_shifted = strikes_base.copy()
                    maturities_shifted = maturities_base.copy()

                # ------------------------------------------------------
                # Extract from fine grid: prices and implied vols
                # ------------------------------------------------------
                mat_idx = [np.argmin(np.abs(base_t - Tm)) for Tm in maturities_shifted]
                price_surf = np.zeros((len(maturities_shifted), len(strikes_shifted)))
                iv_surf = np.zeros_like(price_surf)

                for mi, idx in enumerate(mat_idx):
                    ST = S[:, idx]
                    prices = price_otm_plain(ST, cfg.S0, strikes_shifted)
                    price_surf[mi, :] = prices
                    iv_surf[mi, :] = surface_implied_vols_otm(cfg.S0, strikes_shifted, base_t[idx], prices)

                # Fill missing vols across strikes/maturities
                iv_surf = fill_nans_edgewise(iv_surf.T, strikes=strikes_shifted, maturities=maturities_shifted).T

                results.append({
                    "set_id": s,
                    "fwd_id": j,
                    "grid_id": g_id,
                    "params": vars(params),
                    "grid": {
                        "strikes": strikes_shifted.astype(float),
                        "maturities": maturities_shifted.astype(float),
                    },
                    "price_surface": price_surf,
                    "iv_surface": iv_surf,
                })

                # Periodic save
                if len(results) % save_every == 0:
                    data = {"cfg": cfg.__dict__, "surfaces": results}
                    with open(timestamped_path, "wb") as f:
                        pickle.dump(data, f)
                    with open(progress_path, "wb") as f:
                        pickle.dump(data, f)
                    print(f"[Progress] Saved {len(results)} surfaces → {timestamped_path}")



