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
from typing import List, Dict
from scipy.stats import norm

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
            self.maturities = np.array([0.1,0.3,0.6,0.9,1.2,1.5,1.8,2.0])

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
    t2H = np.power(t, 2.0*H)
    exp_term = np.clip(eta * X - 0.5 * eta**2 * t2H, -1000, 1000)
    v = xi0_t * np.exp(exp_term)
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
    a, b = 1e-8, 5.0
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
    return np.nan_to_num(np.array(ivs), nan=0.05)

# -------------------------------------------------------------
# Parameter sampling
# -------------------------------------------------------------

def sample_param_set() -> RBergomiParams:
    xi0_knots = np.array([0.05,  0.1 ,  0.1 ,  0.1   ,  0.1,
        0.1,  0.1,  0.1])
    eta = 3
    rho = -0.5
    H = 0.4
    return RBergomiParams(eta=eta, rho=rho, H=H, xi0_knots=xi0_knots)

# -------------------------------------------------------------
# Main workflow
# -------------------------------------------------------------

# -------------------------------------------------------------
# Main workflow with optional randomized grids
# -------------------------------------------------------------

def generate_surfaces(
    num_sets=1,
    forward_curves_per_set=1,
    cfg=SimulationConfig(),
    seed=42,
    randomize_grid=False,
    grid_jitter=0.5
) -> List[Dict]:
    """
    Generate implied volatility surfaces from a Rough Bergomi toy simulator.
    Optionally randomizes K and T grids for each surface (± grid_jitter × step).
    """

    np.random.seed(seed)
    results = []
    n, T_max = cfg.n, cfg.T_max
    base_t = np.linspace(0.0, T_max, n)
    dt = base_t[1] - base_t[0]

    for s in range(num_sets):
        params = sample_param_set()
        np.random.seed(seed + s)

        M_half = cfg.M // 2
        dw_half = np.random.normal(0.0, np.sqrt(dt), size=(M_half, n - 1))
        dW_perp_half = np.random.normal(0.0, np.sqrt(dt), size=(M_half, n - 1))
        dw = np.vstack([dw_half, -dw_half])
        dW_perp = np.vstack([dW_perp_half, -dW_perp_half])

        X = fBm_path_rDonsker(n, cfg.M, params.H, T_max)
        grids = [base_t.copy()] * cfg.G

        for j in range(forward_curves_per_set):
            knots = params.xi0_knots

            # --- Base grid definitions ---
            strikes_base = cfg.strikes
            maturities_base = cfg.maturities

            # --- Optional randomized grids ---
            if randomize_grid:
                # random jitter proportional to spacing
                ΔK = strikes_base[1] - strikes_base[0]
                ΔT = maturities_base[1] - maturities_base[0]

                strikes_shifted = strikes_base + np.random.uniform(
                    -grid_jitter * ΔK, grid_jitter * ΔK, size=len(strikes_base)
                )
                maturities_shifted = maturities_base + np.random.uniform(
                    -grid_jitter * ΔT, grid_jitter * ΔT, size=len(maturities_base)
                )

                # clip to valid ranges
                strikes_shifted = np.clip(strikes_shifted, 0.5, 1.5)
                maturities_shifted = np.clip(maturities_shifted, 0.01, T_max)
            else:
                strikes_shifted = strikes_base.copy()
                maturities_shifted = maturities_base.copy()

            # --- Build xi₀ and simulate paths ---
            for g_id, t in enumerate(grids):
                xi0_t = build_xi0_piecewise_constant(knots, t)
                S = simulate_price_paths(cfg.S0, t, X, dw, dW_perp,
                                         xi0_t, params.eta, params.rho, params.H)

                # --- Extract surfaces ---
                mat_idx = [np.argmin(np.abs(t - Tm)) for Tm in maturities_shifted]
                price_surf = np.zeros((len(maturities_shifted), len(strikes_shifted)))
                iv_surf = np.zeros_like(price_surf)

                for mi, idx in enumerate(mat_idx):
                    ST = S[:, idx]
                    prices = price_otm_plain(ST, cfg.S0, strikes_shifted)
                    price_surf[mi, :] = prices
                    iv_surf[mi, :] = surface_implied_vols_otm(cfg.S0, strikes_shifted, t[idx], prices)

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

    return results
