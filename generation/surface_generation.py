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
            self.strikes = np.array([0.5,0.6,0.7,0.8,0.9, 0.95, 1.0, 1.05 ,1.1,1.2,1.3,1.4,1.5])
        if self.maturities is None:
            self.maturities = np.array([0.02,0.035,0.05,0.1,0.2,0.4,0.6,0.8,1,1.2,1.4,1.7,2.0])

# -------------------------------------------------------------
# rDonsker fractional Brownian motion simulator
# -------------------------------------------------------------

def fBm_path_rDonsker_from_increments(dW_perp: np.ndarray, H: float, T: float) -> np.ndarray:
    """
    Construct fractional Brownian motion paths X_t = ∫_0^t K(t-s) dW_perp(s)
    using the rDonsker approximation.

    Parameters
    ----------
    dW_perp : np.ndarray
        Increments of the Brownian motion driving volatility (M x (n-1)).
        These already contain any correlation structure (via rho).
    H : float
        Hurst exponent.
    T : float
        Total time horizon.

    Returns
    -------
    np.ndarray
        Fractional Brownian motion paths X of shape (M, n).
    """
    M, n_minus_1 = dW_perp.shape
    n = n_minus_1 + 1
    i = np.arange(1, n)
    opt_k = ((i**(2*H) - (i-1)**(2*H)) / (2*H)) ** 0.5

    Y = np.zeros((M, n))
    for m in range(M):
        conv = np.convolve(opt_k, dW_perp[m, :])[:n - 1]
        Y[m, 1:] = conv

    scale = T**H * np.sqrt(2*H)
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

def simulate_price_paths(S0: float, t: np.ndarray, X: np.ndarray,
                         dw: np.ndarray, dW_perp: np.ndarray,
                         xi0_t: np.ndarray, eta: float, rho: float, H: float) -> np.ndarray:
    n = min(X.shape[1], t.shape[0], dw.shape[1] + 1, xi0_t.shape[0])
    X, t, xi0_t = X[:, :n], t[:n], xi0_t[:n]
    dw, dW_perp = dw[:, :n-1], dW_perp[:, :n-1]
    M = X.shape[0]
    dt = np.mean(np.diff(t))

    # Fractional volatility process
    t2H = np.power(t, 2.0 * H)
    v = xi0_t * np.exp(eta * X - 0.5 * eta**2 * t2H)

    # Use dw directly (already correlated with dW_perp)
    logS = np.zeros((M, n))
    logS[:, 0] = np.log(S0)
    logS[:, 1:] = logS[:, [0]] + np.cumsum(
        -0.5 * v[:, :-1] * dt + np.sqrt(np.maximum(v[:, :-1], 0.0)) * dw,
        axis=1
    )
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

def fill_nans_edgewise(
    arr: np.ndarray,
    strikes: np.ndarray = None,
    maturities: np.ndarray = None,
    apply_scar_removal: bool = True
) -> np.ndarray:
    """
    Fill NaNs across maturities and strikes by log-linear interpolation + trend-based extrapolation.
    Then lift artificial valleys at the edges for a smooth, realistic surface.
    Optionally applies a post-processing step to remove local scars along the strike dimension.

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

        def safe_slope(y2, y1, x2, x1):
            dx = x2 - x1
            return (y2 - y1) / dx if abs(dx) > eps else 0.0

        # left extrapolation
        slope_left = safe_slope(y_known[1], y_known[0], x_known[1], x_known[0])
        left_mask = x_all < x_known[0]
        interp[left_mask] = y_known[0] + slope_left * (x_all[left_mask] - x_known[0])

        # right extrapolation
        slope_right = safe_slope(y_known[-1], y_known[-2], x_known[-1], x_known[-2])
        right_mask = x_all > x_known[-1]
        interp[right_mask] = y_known[-1] + slope_right * (x_all[right_mask] - x_known[-1])

        interp = np.clip(interp,
                         np.nanmin(y_known) - 5.0 * abs(np.nanstd(y_known)),
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

    # Combine both directions
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

    # -------------------------------------------------------------------------
    # Optional: apply scar removal (strike-wise smoothing)
    # -------------------------------------------------------------------------
    if apply_scar_removal:
        arr_out = fix_strike_outliers(arr_out)

    return arr_out


# --- scar removal helper ---
def fix_strike_outliers(surface, threshold_low=0.98, threshold_high=1.02, 
                        lift_strength=0.9, damp_strength=0.6, window=2):
    """
    Entfernt unphysikalische Dellen (Narben) und Spikes entlang der Strike-Achse.
    Arbeitet maturity-wise (Zeile für Zeile), respektiert lokale Smile-Struktur.
    """
    surface = np.array(surface, dtype=float)
    out = surface.copy()
    n_T, n_K = surface.shape

    for t in range(n_T):
        row = surface[t, :]
        corrected = row.copy()

        for i in range(n_K):
            left = max(0, i - window)
            right = min(n_K, i + window + 1)
            local = row[left:right]
            local_med = np.median(local)

            # Delle → anheben
            if row[i] < threshold_low * local_med:
                corrected[i] = row[i] + lift_strength * (local_med - row[i])
            # Spike → absenken
            elif row[i] > threshold_high * local_med:
                corrected[i] = row[i] - damp_strength * (row[i] - local_med)

        out[t, :] = corrected

    return out



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




# ---------------------------------------------------------------------
# Main surface generation
# ---------------------------------------------------------------------
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
    Generate implied-volatility surfaces from Rough Bergomi simulations.
    Each parameter set (η, ρ, H) defines one 'model world'.
    Within that world, multiple forward-variance curves (xi₀_t) are simulated
    using shared stochastic paths (dw, dW_perp, X), then reshuffled per curve.
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
    base_t = np.linspace(0.0, T_max, n)
    dt = base_t[1] - base_t[0]

    # --- Sample parameters via Latin Hypercube ---
    param_sets = sample_param_sets_lhs(num_sets)

    for s, params in enumerate(param_sets):
        # deterministically advance seed per set
        set_rng = np.random.RandomState(seed + 10_000 * s)

        # --------------------------------------------------------------
        # Build correlated Brownian increments (with antithetics)
        # --------------------------------------------------------------
        M = int(2 * round(cfg.M / 2))
        M_half = M // 2

        z1_half = set_rng.normal(0.0, 1.0, size=(M_half, n - 1))
        z2_half = set_rng.normal(0.0, 1.0, size=(M_half, n - 1))

        # core Brownian for vol driver
        dW_perp_half = np.sqrt(dt) * z1_half
        # correlated spot Brownian: dw = ρ z1 + sqrt(1-ρ²) z2
        rho = float(params.rho)
        dw_half = np.sqrt(dt) * (rho * z1_half + np.sqrt(max(1.0 - rho * rho, 0.0)) * z2_half)

        # antithetic pairing (keeps correlation structure intact)
        dW_perp = np.vstack([dW_perp_half, -dW_perp_half])
        dw = np.vstack([dw_half, -dw_half])

        # fBm from the SAME dW_perp
        X = fBm_path_rDonsker_from_increments(dW_perp, float(params.H), T_max)

        # --------------------------------------------------------------
        # Generate multiple forward curves under same model world
        # --------------------------------------------------------------
        for j in range(forward_curves_per_set):
            # unique forward variance curve
            xi0_knots = set_rng.uniform(0.01, 0.16, size=8)
            xi0_t = build_xi0_piecewise_constant(xi0_knots, base_t)


            # simulate price paths
            S = simulate_price_paths(
                cfg.S0, base_t, X, dw, dW_perp,
                xi0_t, float(params.eta), rho, float(params.H)
            )

            strikes_base = cfg.strikes
            maturities_base = cfg.maturities

            for g_id in range(cfg.G):

                # Optional randomized grids (use your jitter helper)
                if randomize_grid:
                    strikes_shifted = jitter_grid(strikes_base, grid_jitter=0.25, min_spacing=0.05)
                    maturities_shifted = jitter_grid(maturities_base, grid_jitter=grid_jitter, min_spacing=0.02)
                else:
                    strikes_shifted = strikes_base.copy()
                    maturities_shifted = maturities_base.copy()

                # ------------------------------------------------------
                # Extract: prices and implied vols on the (possibly jittered) grid
                # ------------------------------------------------------
                mat_idx = [int(np.argmin(np.abs(base_t - Tm))) for Tm in maturities_shifted]
                price_surf = np.zeros((len(maturities_shifted), len(strikes_shifted)))
                iv_surf = np.zeros_like(price_surf)

                for mi, idx in enumerate(mat_idx):
                    ST = S[:, idx]
                    prices = price_otm_plain(ST, cfg.S0, strikes_shifted)
                    price_surf[mi, :] = prices
                    iv_surf[mi, :] = surface_implied_vols_otm(cfg.S0, strikes_shifted, base_t[idx], prices)

                # Fill/sanitize IV surface
                iv_surf = fill_nans_edgewise(
                    iv_surf.T, strikes=strikes_shifted, maturities=maturities_shifted
                ).T

                # ======================================================
                # NaN / Inf guard with debug dump
                # ======================================================
                if (
                    np.isnan(iv_surf).any()
                    or np.isnan(price_surf).any()
                    or np.isinf(iv_surf).any()
                    or np.isinf(price_surf).any()
                ):
                    bad_dir = os.path.join("data", "debug_nans")
                    os.makedirs(bad_dir, exist_ok=True)
                    bad_path = os.path.join(bad_dir, f"bad_surface_set{s}_fwd{j}_grid{g_id}.npz")
                    np.savez_compressed(
                        bad_path,
                        price_surface=price_surf,
                        iv_surface=iv_surf,
                        strikes=strikes_shifted,
                        maturities=maturities_shifted,
                        params=np.array([float(params.eta), rho, float(params.H)], dtype=float),
                        xi0_knots=xi0_knots.astype(float),
                    )
                    print(f"NaN/Inf detected in surface (set={s}, fwd={j}, grid={g_id}) → {bad_path}")
                    continue

                # Record (note: params include THIS surface's xi0_knots)
                params_record = {
                    "eta": float(params.eta),
                    "rho": rho,
                    "H": float(params.H),
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

                # Periodic safe save (atomic replace)
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

    # --- Final save (atomic) ---
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

