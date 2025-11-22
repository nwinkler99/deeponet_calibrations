# Rough Bergomi IV surfaces via rDonsker fBm — Stable IV Extraction + OTM Options
# --------------------------------------------------------------------------------------------------------
# Features:
# - Constant time grid (consistent discretization)
# - Randomized Maturities (±15%)
# - OTM pricing (Calls for K>=S0, Puts for K<S0)
# - Stable implied vol inversion (clipping + NaN handling)
# - Piecewise constant xi0, Antithetic variates, batch seeding
# --------------------------------------------------------------------------------------------------------
import gc
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
    lhs_grid,
    heston_call_price
)


# -------------------------------------------------------------
# Parameter structures
# -------------------------------------------------------------

@dataclass
class SimulationConfig:
    # -----------------------------
    # global simulation settings
    # -----------------------------
    M: int = 20000
    n: int = 1200
    T_min: float = 25/365
    T_max: float = 2.0
    S0: float = 1.0
    batch_size: int = 5000
    G: int = 10
    dtype: np.dtype = np.float32

    # -----------------------------
    # RBergomi parameter ranges
    # -----------------------------
    min_xi0 : float = 0.01
    max_xi0 : float = 0.25
    min_eta : float = 0.5
    max_eta : float = 4.0
    min_rho : float = -1.0
    max_rho : float = -0.1
    min_H   : float = 0.025
    max_H   : float = 0.5

    # -----------------------------
    # Heston parameter ranges
    # -----------------------------
    heston_min_kappa = 0.5
    heston_max_kappa = 3.0

    heston_min_theta = 0.01
    heston_max_theta = 0.25

    heston_min_v0 = 0.01
    heston_max_v0 = 0.25

    heston_min_sigma = 0.5
    heston_max_sigma = 3

    heston_min_rho = -1
    heston_max_rho = -0.1

    # -----------------------------
    # grids
    # -----------------------------
    strikes: np.ndarray = None
    maturities: np.ndarray = None

    def __post_init__(self):
        if self.strikes is None:
            self.logstrikes = np.linspace(-0.4, 0.4, 15)
            self.strikes = np.exp(self.logstrikes) * self.S0

        if self.maturities is None:
            self.logmaturities = np.linspace(np.log(self.T_min),
                                             np.log(self.T_max), 15)
            self.maturities = np.exp(self.logmaturities)




# --------------------------------------------------------------------------------------------------------
# Surface generation (batch)
# --------------------------------------------------------------------------------------------------------

import gc
import numpy as np
from numpy.random import SeedSequence, default_rng
from typing import List, Dict

# ============================================================
#  Main generator: fully reproducible
# ============================================================
def generate_surfaces(
    num_sets: int = 1,
    forward_curves_per_set: int = 1,
    cfg=None,
    seed: int = 42,
    randomize_grid: bool = False,
    model: str = "rbergomi"
):
    model = model.lower()

    if model == "rbergomi":
        # --- dein bestehender Generator ---
        return generate_surfaces_rbergomi(
            num_sets=num_sets,
            forward_curves_per_set=forward_curves_per_set,
            cfg=cfg,
            seed=seed,
            randomize_grid=randomize_grid
        )

    elif model == "heston":
        return generate_heston_surfaces(
            num_sets=num_sets,
            cfg=cfg,
            seed=seed,
            randomize_grid=randomize_grid
        )

    else:
        raise ValueError(f"Unknown model '{model}'. Use 'rbergomi' or 'heston'.")



def generate_surfaces_rbergomi(
    num_sets: int = 1,
    forward_curves_per_set: int = 1,
    cfg=None,
    seed: int = 42,
    randomize_grid: bool = False
) -> List[Dict]:
    """
    Deterministic generation of implied-volatility surfaces from rBergomi model.

    Every stochastic element (param LHS, xi0, rBergomi paths, jitter grids)
    is driven by explicitly controlled RNGs derived from one SeedSequence.
    """
    from generation.rbergomi import rBergomi, bs_call_vec_pathwise, bsinv, bs_vega  # adjust paths if needed

    results: List[Dict] = []

    # --- 1️⃣ Central deterministic seed hierarchy ---
    root_seq = SeedSequence(seed)
    rng_params = default_rng(root_seq.spawn(1)[0])
    param_sets = sample_param_sets_lhs(
        num_sets,
        rng_params,
        lower=np.array([cfg.min_eta, cfg.min_rho, cfg.min_H]),
        upper=np.array([cfg.max_eta, cfg.max_rho, cfg.max_H]),
    )

    set_seqs = root_seq.spawn(num_sets)

    n, T_max = cfg.n, cfg.T_max

    # --- 2️⃣ Generate per param set ---
    for s, (params, set_seq) in enumerate(zip(param_sets, set_seqs)):
        eta, rho, H = params   # <— Vektor unpacken
        eta = float(eta)
        rho = float(rho)
        H   = float(H)
        subseqs = set_seq.spawn(3)
        rng_rb = default_rng(subseqs[0])     # for Brownian increments (rBergomi)
        rng_xi = default_rng(subseqs[1])     # for xi0 knots
        rng_jit = default_rng(subseqs[2])    # for jitter grids

        eta, rho, H = float(params.eta), float(params.rho), float(params.H)
        a = H - 0.5

        # --- 3️⃣ rBergomi simulation (deterministic via rng_rb) ---
        np.random.seed(rng_rb.integers(0, 2**31 - 1))  # if rBergomi uses np.random internally
        rb = rBergomi(n=n, N=cfg.M, T=T_max, a=a)
        t_grid = rb.t.flatten().astype(cfg.dtype)
        dt = np.diff(t_grid, prepend=cfg.dtype(0.0)).astype(cfg.dtype)

        dW1, dW2 = rb.dW1(), rb.dW2()
        dB, Y = rb.dB(dW1, dW2, rho=rho), rb.Y(dW1)
        dW_vol = dW1[..., 0] if dW1.ndim == 3 else dW1
        dW_vol = dW_vol.astype(cfg.dtype, copy=False)
        del dW1, dW2
        gc.collect()

        # --- 4️⃣ Forward variance curve generation ---
        for j in range(forward_curves_per_set):

            # --------------------------------------------------------------
            # Forward variance knots derived from log-maturity grid
            # Guaranteed to include T_max
            # --------------------------------------------------------------

            # 1) jedes zweite log-T nehmen
            log_forward_points = cfg.logmaturities[::3].copy()

            # 2) sicherstellen, dass T_max (als log(T_max)) enthalten ist
            log_T_max = np.log(cfg.T_max)

            if not np.isclose(log_forward_points[-1], log_T_max):
                log_forward_points[-1] = log_T_max

            # 3) Aufräumen: sortieren & Duplikate entfernen
            log_forward_points = np.unique(np.sort(log_forward_points))

            # 4) zurück zu physischer Zeit
            forward_points = np.exp(log_forward_points)

            # 5) xi0-Knoten generieren
            xi0_knots = rng_xi.uniform(cfg.min_xi0, cfg.max_xi0, size=len(forward_points)).astype(cfg.dtype)


            # map xi0_knots to sim grid
            base_edges = np.concatenate([[0.0], forward_points.astype(float)])
            target_u = np.linspace(0, 1, len(xi0_knots) + 1)
            bin_edges = np.interp(target_u, np.linspace(0, 1, len(base_edges)), base_edges)
            idx = np.searchsorted(bin_edges, t_grid.astype(float), side="right") - 1
            idx = np.clip(idx, 0, len(xi0_knots) - 1)
            xi_t = xi0_knots[idx]

            # simulate volatility & price paths
            V = rb.V(Y, xi=xi_t[np.newaxis, :], eta=eta).astype(cfg.dtype)
            S = rb.S(V, dB, S0=cfg.S0).astype(cfg.dtype)

            # precompute cumulative integrals for conditional MC
            V_left = V[:, :-1].astype(cfg.dtype, copy=False)
            dt_incr = dt[1:]
            sqrtV_left = np.sqrt(np.maximum(V_left, cfg.dtype(0.0)))
            I1_cum = np.cumsum(sqrtV_left * dW_vol, axis=1, dtype=cfg.dtype)
            I2_cum = np.cumsum(V_left * dt_incr[None, :], axis=1, dtype=cfg.dtype)

            # --- 5️⃣ Grid generation & IV computation ---
            for g_id in range(cfg.G):

                # --- Base grids (linear + log) ---
                strikes_base        = cfg.strikes.astype(cfg.dtype)
                logstrikes_base     = cfg.logstrikes.astype(cfg.dtype)

                maturities_base     = cfg.maturities.astype(cfg.dtype)
                logmaturities_base  = cfg.logmaturities.astype(cfg.dtype)

                # ----------------------------------------------------------
                # Apply jitter (log-space) or use base grid
                # ----------------------------------------------------------
                if randomize_grid and g_id > 0:
                    # --- log-strikes jitter ---
                    logstrikes_shifted = np.array(
                        lhs_grid(
                            start=logstrikes_base.min(),
                            end=logstrikes_base.max(),
                            n=len(logstrikes_base),
                            rng=rng_jit
                        ),
                        dtype=cfg.dtype,
                    )
                    strikes_shifted = np.exp(logstrikes_shifted) * cfg.S0

                    # --- log-maturities jitter ---
                    logmaturities_shifted = np.array(
                        lhs_grid(
                             start=logmaturities_base.min(),
                            end=logmaturities_base.max(),    
                            n=len(logmaturities_base),                 
                            rng=rng_jit
                        ),
                        dtype=cfg.dtype,
                    )
                    maturities_shifted = np.exp(logmaturities_shifted).astype(cfg.dtype)

                else:
                    # No jitter → use base grids
                    logstrikes_shifted     = logstrikes_base.copy()
                    strikes_shifted        = strikes_base.copy()

                    logmaturities_shifted  = logmaturities_base.copy()
                    maturities_shifted     = maturities_base.copy()
                

                nT, nK = len(maturities_shifted), len(strikes_shifted)
                iv_surf = np.zeros((nT, nK), dtype=cfg.dtype)
                iv_relerr = np.zeros_like(iv_surf)
                S0 = cfg.dtype(cfg.S0)

                for mi, Tm in enumerate(maturities_shifted):
                    t_idx = np.searchsorted(t_grid.astype(float), float(Tm), side="right") - 1
                    t_idx = int(np.clip(t_idx, 0, S.shape[1] - 1))
                    T_float = float(Tm)
                    if T_float <= 0:
                        iv_surf[mi, :], iv_relerr[mi, :] = np.nan, np.nan
                        continue

                    # pull integrals
                    if t_idx == 0:
                        I1 = np.zeros((V.shape[0],), dtype=cfg.dtype)
                        I2 = np.zeros((V.shape[0],), dtype=cfg.dtype)
                    else:
                        I1 = I1_cum[:, t_idx - 1]
                        I2 = I2_cum[:, t_idx - 1]

                    F_path = S0 * np.exp(cfg.dtype(rho) * I1 - cfg.dtype(0.5) * (cfg.dtype(rho) ** 2) * I2)
                    sigma_path = np.sqrt(
                        np.maximum((cfg.dtype(1.0) - cfg.dtype(rho) ** 2) * I2 / Tm, cfg.dtype(1e-16))
                    )

                    for ki, K_ in enumerate(strikes_shifted):
                        K_abs = K_ * S0
                        is_call = K_abs >= float(S0)
                        if is_call:
                            cond_vals = bs_call_vec_pathwise(F_path, K_abs, T_float, sigma_path).astype( cfg.dtype)
                            o_flag = "call"
                        else:
                            cond_vals = bs_put_vec_pathwise(F_path, K_abs, T_float, sigma_path).astype( cfg.dtype)
                            o_flag = "put"

                        price_cmc =  cfg.dtype(np.mean(cond_vals))
                        se_price =  cfg.dtype(np.std(cond_vals, ddof=1)) /  cfg.dtype(np.sqrt(cfg.M))

                        inv_t0 = time.perf_counter()
                        try:
                            iv_val = bsinv(float(price_cmc), float(S0), float(K_abs), float(Tm), o=o_flag)
                            iv_surf[mi, ki] =  cfg.dtype(iv_val)
                            vega = bs_vega(float(S0), float(K_abs), float(Tm), float(iv_val))
                            if np.isfinite(vega) and vega > 1e-12 and iv_val > 1e-12:
                                se_iv = se_price /  cfg.dtype(vega)
                                iv_relerr[mi, ki] =  cfg.dtype(1.96) * se_iv /  cfg.dtype(iv_val)
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
                            "strikes": logstrikes_shifted.astype(cfg.dtype),
                            "maturities": logmaturities_shifted.astype(cfg.dtype),
                        },
                        "iv_surface": iv_surf,
                        "iv_rel_error": iv_relerr,
                    }
                )

    return results


def generate_heston_surfaces(
    num_sets: int = 1,
    cfg=None,
    seed: int = 42,
    randomize_grid: bool = False,
):
    """
    Heston IV-surface generator:
        - LHS parameter sampling based on cfg.heston_min_* and cfg.heston_max_*
        - deterministic SeedSequence hierarchy
        - log-grid jitter (same style as rBergomi)
        - identical output structure as rBergomi surfaces
    """

    root_seq = SeedSequence(seed)
    rng_params = default_rng(root_seq.spawn(1)[0])

    # LHS sampling
    param_sets = sample_param_sets_lhs(
        num_sets,
        rng_params,
        lower=np.array([
            cfg.heston_min_kappa,
            cfg.heston_min_theta,
            cfg.heston_min_v0,
            cfg.heston_min_sigma,
            cfg.heston_min_rho,
        ]),
        upper=np.array([
            cfg.heston_max_kappa,
            cfg.heston_max_theta,
            cfg.heston_max_v0,
            cfg.heston_max_sigma,
            cfg.heston_max_rho,
        ]),
    )

    # independent jitter RNGs
    set_seqs = root_seq.spawn(num_sets)
    results = []

    for s, (params, set_seq) in enumerate(zip(param_sets, set_seqs)):
        kappa, theta, v0, sigma, rho = params
        kappa = float(kappa)
        theta = float(theta)
        v0    = float(v0)
        sigma = float(sigma)
        rho   = float(rho)

        rng_jit = default_rng(set_seq.spawn(1)[0])

        base_logstrikes = cfg.logstrikes.astype(cfg.dtype)
        base_logmaturities = cfg.logmaturities.astype(cfg.dtype)
        base_strikes = cfg.strikes.astype(cfg.dtype)
        base_maturities = cfg.maturities.astype(cfg.dtype)

        for g_id in range(cfg.G):

            if randomize_grid and g_id > 0:
                logstrikes = np.array(
                    lhs_grid(
                        float(base_logstrikes.min()),
                        float(base_logstrikes.max()),
                        len(base_logstrikes),
                        rng=rng_jit
                    ),
                    dtype=cfg.dtype
                )
                strikes = np.exp(logstrikes) * cfg.S0

                logmaturities = np.array(
                    lhs_grid(
                        float(base_logmaturities.min()),
                        float(base_logmaturities.max()),
                        len(base_logmaturities),
                        rng=rng_jit
                    ),
                    dtype=cfg.dtype
                )
                maturities = np.exp(logmaturities).astype(cfg.dtype)

            else:
                logstrikes = base_logstrikes.copy()
                strikes = base_strikes.copy()

                logmaturities = base_logmaturities.copy()
                maturities = base_maturities.copy()

            # Create IV surface
            nT, nK = len(maturities), len(strikes)
            iv_surf = np.zeros((nT, nK), dtype=cfg.dtype)

            for ti, T in enumerate(maturities):
                T_float = float(T)
                for ki, K in enumerate(strikes):
                    K_float = float(K)

                    try:
                        # Decide call or put based on moneyness:
                        if K_float >= cfg.S0:
                            price = heston_call_price(cfg.S0, K_float, T_float, params)
                            iv = bsinv(price, cfg.S0, K_float, T_float, o="call")
                        else:
                            # Use put via put-call parity
                            call_price = heston_call_price(cfg.S0, K_float, T_float, params)
                            put_price  = call_price - cfg.S0 + K_float * np.exp(-0*T_float)
                            iv = bsinv(put_price, cfg.S0, K_float, T_float, o="put")

                    except:
                        iv = np.nan

                    iv_surf[ti, ki] = cfg.dtype(iv)

            results.append(
                {
                    "set_id": s,
                    "fwd_id": 0,
                    "grid_id": g_id,
                    "params": {
                        "kappa": kappa,
                        "theta": theta,
                        "v0": v0,
                        "sigma": sigma,
                        "rho": rho,
                    },
                    "grid": {
                        "strikes": logstrikes.astype(cfg.dtype),
                        "maturities": logmaturities.astype(cfg.dtype),
                    },
                    "iv_surface": iv_surf,
                    "iv_rel_error": np.zeros_like(iv_surf),
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
    t0_total = time.perf_counter()

    # --- 1. rBergomi object creation ---
    t0 = time.perf_counter()
    rb = rBergomi(n=cfg.n, N=cfg.M, T=float(maturities[-1]), a=float(a))
    t_obj = time.perf_counter() - t0

    # --- 2. Time grid construction ---
    t0 = time.perf_counter()
    t_grid = rb.t.flatten().astype(dtype)
    dt = np.diff(t_grid, prepend=dtype(0.0))
    t_grid_time = time.perf_counter() - t0

    # --- 3. Draw Brownian motions ---
    t0 = time.perf_counter()
    dW1 = rb.dW1()
    dW2 = rb.dW2()
    t_brown_time = time.perf_counter() - t0

    # --- 4. Correlated driver dB ---
    t0 = time.perf_counter()
    dB = rb.dB(dW1, dW2, rho=float(rho))
    t_dB_time = time.perf_counter() - t0

    # --- 5. fBM kernel convolution / Y computation ---
    t0 = time.perf_counter()
    Y = rb.Y(dW1)
    t_Y_time = time.perf_counter() - t0

    t_total = time.perf_counter() - t0_total

    print(f"\n=== Simulator setup breakdown ===")
    print(f"rBergomi object init : {t_obj:7.3f} s")
    print(f"t-grid + dt setup    : {t_grid_time:7.3f} s")
    print(f"Brownian draws (W1,W2): {t_brown_time:7.3f} s")
    print(f"dB computation        : {t_dB_time:7.3f} s")
    print(f"Y computation (fBM)   : {t_Y_time:7.3f} s")
    print(f"TOTAL setup           : {t_total:7.3f} s")
    sim_time = time.perf_counter() - t0_global

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
        "iv_surface": iv_surf,
        "iv_rel_error": iv_relerr,
    }

