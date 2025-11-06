# Rough Bergomi IV surfaces via rDonsker fBm — Stable IV Extraction + OTM Options
# --------------------------------------------------------------------------------------------------------
# Features:
# - Constant time grid (consistent discretization)
# - Randomized Maturities (±15%)
# - OTM pricing (Calls for K>=S0, Puts for K<S0)
# - Stable implied vol inversion (clipping + NaN handling)
# - Piecewise constant xi0, Antithetic variates, batch seeding
# --------------------------------------------------------------------------------------------------------

import os
import pickle
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
from scipy.stats import norm, qmc

from .rbergomi import rBergomi
from .utils import (
    bs,
    bsinv,
    bs_vega,
    bs_call_vec_pathwise,
    bs_put_vec_pathwise,
    sample_param_sets_lhs,
    jitter_grid,
)


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
            self.strikes = np.array(
                [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
            )
        if self.maturities is None:
            self.maturities = np.array(
                [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1]
            )


# --------------------------------------------------------------------------------------------------------
# Surface generation (batch)
# --------------------------------------------------------------------------------------------------------

def generate_surfaces(
    num_sets: int = 1,
    forward_curves_per_set: int = 1,
    cfg: SimulationConfig = None,
    seed: int = 42,
    randomize_grid: bool = False,
    grid_jitter: float = 0.5
) -> List[Dict]:
    """
    Generate implied-volatility surfaces from the rBergomi model using Monte Carlo simulation.
    Uses rho-aware conditional Monte Carlo (martingale control variate):
      per path and maturity, use BS with
        F_path = S0*exp(rho*∫sqrt(V)dW - 0.5*rho^2∫Vds)
        sigma_path^2 = (1 - rho^2) * ∫Vds / T.
    Computes iv_rel_error as 1.96 * SE_iv / iv.
    """
    #save_dir = os.path.join("data", date_str if randomize_grid else "fixed_longrun")
    #os.makedirs(save_dir, exist_ok=True)

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
        t_grid = rb.t.flatten().astype(cfg.dtype)
        dt = np.diff(t_grid, prepend=cfg.dtype(0.0)).astype(cfg.dtype)
        dW1, dW2 = rb.dW1(), rb.dW2()
        dB, Y = rb.dB(dW1, dW2, rho=rho), rb.Y(dW1)

        # volatility driver increments W^(1)
        dW_vol = dW1[..., 0] if dW1.ndim == 3 else dW1
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

            V = rb.V(Y, xi=xi_t[np.newaxis, :], eta=eta).astype(cfg.dtype)
            S = rb.S(V, dB, S0=cfg.S0).astype(cfg.dtype)

            # Precompute cumulative integrals for CMC
            V_left = V[:, :-1].astype(cfg.dtype, copy=False)
            dt_incr = dt[1:]
            sqrtV_left = np.sqrt(np.maximum(V_left, cfg.dtype(0.0)))

            I1_cum = np.cumsum(sqrtV_left * dW_vol, axis=1, dtype=cfg.dtype)
            I2_cum = np.cumsum(V_left * dt_incr[None, :], axis=1, dtype=cfg.dtype)

            for g_id in range(cfg.G):
                if randomize_grid and g_id > 0:
                    strikes_shifted = np.array(
                        jitter_grid(strikes_base, grid_jitter=grid_jitter, min_spacing=0.02),
                        dtype=cfg.dtype,
                    )
                    maturities_shifted = np.array(
                        jitter_grid(maturities_base, grid_jitter=grid_jitter, min_spacing=0.02),
                        dtype=cfg.dtype,
                    )
                else:
                    strikes_shifted = strikes_base.copy()
                    maturities_shifted = maturities_base.copy()

                nT, nK = len(maturities_shifted), len(strikes_shifted)
                iv_surf = np.zeros((nT, nK), dtype=cfg.dtype)
                iv_relerr = np.zeros_like(iv_surf)
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
                    F_path = S0 * np.exp(
                        cfg.dtype(rho) * I1
                        - cfg.dtype(0.5) * (cfg.dtype(rho) * cfg.dtype(rho)) * I2
                    )
                    sigma_path = np.sqrt(
                        np.maximum(
                            (cfg.dtype(1.0) - cfg.dtype(rho) * cfg.dtype(rho)) * I2 / Tm,
                            cfg.dtype(1e-16),
                        )
                    )

                    for ki, K_ in enumerate(strikes_shifted):
                        K_abs = K_ * S0

                        # Choose OTM side: right wing (K > S0) -> call, left wing -> put (via parity)
                        use_call = bool(float(K_abs) > float(S0))
                        call_vals = bs_call_vec_pathwise(
                            F_path, float(K_abs), T_float, sigma_path
                        ).astype(cfg.dtype)

                        if use_call:
                            cond_vals = call_vals
                            o_flag = "call"
                        else:
                            cond_vals = (call_vals - (F_path - K_abs)).astype(cfg.dtype)
                            o_flag = "put"

                        # Mean and SE across paths (of conditional expectations)
                        price_cmc = cfg.dtype(np.mean(cond_vals))
                        se_price = cfg.dtype(np.std(cond_vals, ddof=1)) / cfg.dtype(
                            np.sqrt(cfg.M)
                        )

                        # guard tiny prices (ill-posed inversion)
                        if not np.isfinite(price_cmc) or price_cmc < cfg.dtype(1e-8):
                            iv_surf[mi, ki] = np.nan
                            iv_relerr[mi, ki] = np.nan
                            continue

                        # Invert with correct option flag; propagate SE via vega
                        try:
                            iv_val = bsinv(
                                float(price_cmc), float(S0), float(K_abs), T_float, o=o_flag
                            )
                            iv_surf[mi, ki] = cfg.dtype(iv_val)
                            vega = bs_vega(float(S0), float(K_abs), T_float, float(iv_val))
                            if (
                                np.isfinite(vega)
                                and vega > 1e-12
                                and iv_val > 1e-12
                            ):
                                se_iv = se_price / cfg.dtype(vega)
                                iv_relerr[mi, ki] = (
                                    cfg.dtype(1.96) * se_iv / cfg.dtype(iv_val)
                                )
                            else:
                                iv_relerr[mi, ki] = np.nan
                        except Exception:
                            iv_surf[mi, ki] = np.nan
                            iv_relerr[mi, ki] = np.nan

                    iv_surf[mi, :] = np.clip(iv_surf[mi, :], cfg.dtype(1e-4), cfg.dtype(5.0))

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


# --------------------------------------------------------------------------------------------------------
# Single fixed-surface generator
# --------------------------------------------------------------------------------------------------------

import time

def generate_fixed_surface(
    param_set: Dict,
    xi0_knots: np.ndarray,
    strikes: np.ndarray,
    maturities: np.ndarray,
    cfg: SimulationConfig,
    seed: int = 123,
) -> Dict:
    """
    Generate one IV surface for fixed parameters and a fixed (K, T) grid.
    Includes detailed runtime diagnostics for profiling.
    """
    dtype = cfg.dtype
    eta = dtype(param_set["eta"])
    rho = dtype(param_set["rho"])
    H = dtype(param_set["H"])
    a = H - dtype(0.5)

    t0_global = time.perf_counter()

    # ----------------- 1. Simulator setup -----------------
    t0 = time.perf_counter()
    rb = rBergomi(n=cfg.n, N=cfg.M, T=float(maturities[-1]), a=float(a))
    t_grid = rb.t.flatten().astype(dtype)
    dt = np.diff(t_grid, prepend=dtype(0.0))
    dW1, dW2 = rb.dW1(), rb.dW2()
    dB, Y = rb.dB(dW1, dW2, rho=float(rho)), rb.Y(dW1)
    sim_time = time.perf_counter() - t0

    # ----------------- 2. Forward variance mapping -----------------
    t0 = time.perf_counter()
    Kk = len(xi0_knots)
    if Kk <= len(maturities):
        bin_edges = np.concatenate([[0.0], maturities.astype(float)])
    else:
        base_edges = np.concatenate([[0.0], maturities.astype(float)])
        target_u = np.linspace(0, 1, Kk + 1)
        bin_edges = np.interp(target_u, np.linspace(0, 1, len(base_edges)), base_edges)

    idx = np.searchsorted(bin_edges, t_grid.astype(float), side="right") - 1
    idx = np.clip(idx, 0, Kk - 1)
    xi_t = np.array(xi0_knots, dtype=dtype)[idx]
    map_time = time.perf_counter() - t0

    # ----------------- 3. Path simulation (V, S) -----------------
    t0 = time.perf_counter()
    V = rb.V(Y, xi=xi_t[np.newaxis, :], eta=float(eta)).astype(dtype)
    S = rb.S(V, dB, S0=dtype(cfg.S0)).astype(dtype)
    sim_paths_time = time.perf_counter() - t0

    # ----------------- 4. Conditional MC setup -----------------
    t0 = time.perf_counter()
    dW_vol = dW1[..., 0] if dW1.ndim == 3 else dW1
    dW_vol = dW_vol.astype(dtype, copy=False)
    V_left = V[:, :-1].astype(dtype, copy=False)
    dt_incr = dt[1:].astype(dtype)
    sqrtV_left = np.sqrt(np.maximum(V_left, dtype(0.0)))
    I1_cum = np.cumsum(sqrtV_left * dW_vol, axis=1, dtype=dtype)
    I2_cum = np.cumsum(V_left * dt_incr[None, :], axis=1, dtype=dtype)
    cmc_setup_time = time.perf_counter() - t0

    # ----------------- 5. IV extraction -----------------
    t0 = time.perf_counter()
    nT, nK = len(maturities), len(strikes)
    price_surf = np.zeros((nT, nK), dtype=dtype)
    iv_surf = np.zeros_like(price_surf)
    iv_relerr = np.zeros_like(price_surf)
    S0 = dtype(cfg.S0)

    inv_time_total = 0.0
    for mi, Tm in enumerate(maturities):
        t_m_start = time.perf_counter()

        t_idx = np.searchsorted(t_grid, float(Tm), side="right") - 1
        t_idx = int(np.clip(t_idx, 0, S.shape[1] - 1))
        T_float = float(Tm)
        if T_float <= 0:
            continue

        I1 = np.zeros(V.shape[0], dtype=dtype) if t_idx == 0 else I1_cum[:, t_idx - 1]
        I2 = np.zeros(V.shape[0], dtype=dtype) if t_idx == 0 else I2_cum[:, t_idx - 1]

        F_path = S0 * np.exp(rho * I1 - dtype(0.5) * (rho * rho) * I2)
        sigma_path = np.sqrt(np.maximum((dtype(1.0) - rho * rho) * I2 / Tm, dtype(1e-16)))

        for ki, K_ in enumerate(strikes):
            K_abs = K_ * S0
            is_call = K_abs >= float(S0)
            if is_call:
                cond_vals = bs_call_vec_pathwise(F_path, K_abs, T_float, sigma_path).astype(dtype)
                o_flag = "call"
            else:
                cond_vals = bs_put_vec_pathwise(F_path, K_abs, T_float, sigma_path).astype(dtype)
                o_flag = "put"

            price_cmc = dtype(np.mean(cond_vals))
            se_price = dtype(np.std(cond_vals, ddof=1)) / dtype(np.sqrt(cfg.M))
            price_surf[mi, ki] = price_cmc

            inv_t0 = time.perf_counter()
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
            inv_time_total += time.perf_counter() - inv_t0

        # Clip once per maturity
        iv_surf[mi, :] = np.clip(iv_surf[mi, :], dtype(1e-4), dtype(5.0))

    iv_extraction_time = time.perf_counter() - t0
    total_time = time.perf_counter() - t0_global

    # ----------------- Diagnostics summary -----------------
    print(
        f"\n=== Diagnostics for seed {seed} ===\n"
        f"Simulator setup:     {sim_time:7.3f} s\n"
        f"Xi0 mapping:         {map_time:7.3f} s\n"
        f"Path generation:     {sim_paths_time:7.3f} s\n"
        f"CMC integrals:       {cmc_setup_time:7.3f} s\n"
        f"IV extraction total: {iv_extraction_time:7.3f} s "
        f"(of which Brent inversions ≈ {inv_time_total:7.3f} s)\n"
        f"TOTAL runtime:       {total_time:7.3f} s"
    )

    # ----------------- Optional postprocessing -----------------
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

