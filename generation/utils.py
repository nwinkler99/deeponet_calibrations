# --------------------------------------------------------------------------------------------------------
# Utility functions for Rough Bergomi model simulation and visualization
# --------------------------------------------------------------------------------------------------------
# Includes:
# - TBSS kernel and discretization helpers
# - Black–Scholes pricing and implied volatility inversion
# - Vectorized pathwise pricing for Monte Carlo
# - Latin Hypercube parameter sampling and randomized grids
# - Plotting helper for IV surfaces
# --------------------------------------------------------------------------------------------------------

import numpy as np
from scipy.stats import norm, qmc
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.pyplot as plt
from numpy.random import SeedSequence, default_rng
import pandas as pd

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "lines.linewidth": 1.8,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "axes.titleweight": "semibold",
    "figure.dpi": 150,
})

def _tight_suptitle(fig, title, fontsize=17):
    """Add a large, bold suptitle closer to subplots."""
    fig.suptitle(title, fontsize=fontsize, weight="bold", y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
# --------------------------------------------------------------------------------------------------------
# TBSS / rBergomi helper functions
# --------------------------------------------------------------------------------------------------------

def g(x: float, a: float) -> float:
    """TBSS kernel applicable to the rBergomi variance process."""
    return x ** a


def b(k: int, a: float) -> float:
    """Optimal discretisation of TBSS process for minimising hybrid scheme error."""
    return ((k ** (a + 1) - (k - 1) ** (a + 1)) / (a + 1)) ** (1 / a)


def cov(a: float, n: int) -> np.ndarray:
    """
    Covariance matrix for given alpha and n, assuming kappa = 1 for tractability.
    """
    C = np.zeros((2, 2))
    C[0, 0] = 1.0 / n
    C[0, 1] = 1.0 / ((a + 1.0) * n ** (a + 1))
    C[1, 1] = 1.0 / ((2.0 * a + 1) * n ** (2 * a + 1))
    C[1, 0] = C[0, 1]
    return C


# --------------------------------------------------------------------------------------------------------
# Black–Scholes formulas
# --------------------------------------------------------------------------------------------------------

def bs(F: float, K: float, V: float, o: str = "call") -> float:
    """
    Returns the Black call/put/otm price for given forward F, strike K, and integrated variance V.
    """
    # Set appropriate weight for option token o
    w = 1
    if o == "put":
        w = -1
    elif o == "otm":
        w = 2 * (K > 1.0) - 1

    sv = np.sqrt(V)
    d1 = np.log(F / K) / sv + 0.5 * sv
    d2 = d1 - sv
    return w * F * norm.cdf(w * d1) - w * K * norm.cdf(w * d2)


def bsinv(P: float, F: float, K: float, t: float, o: str = "call") -> float:
    """
    Robust implied Black volatility from price P, forward F, strike K, maturity t.
    Handles call/put/otm options; safe for MC-generated prices.
    """
    if t <= 1e-10:
        return 1e-8  # degenerate maturity

    w = 1.0
    if o == "put":
        w = -1.0
    elif o == "otm":
        w = 2.0 * (K > F) - 1.0  # consistent OTM switch

    intrinsic = max(w * (F - K), 0.0)
    P = max(P, intrinsic + 1e-12)

    def error(sigma):
        return bs(F, K, sigma ** 2 * t, o) - P

    try:
        return brentq(error, 1e-8, 5.0, xtol=1e-8, maxiter=100)
    except ValueError:
        # Root not bracketed (e.g. price outside BS range)
        return np.nan


def bs_vega(F: float, K: float, T: float, sigma: float) -> float:
    """Black–Scholes vega (∂Price/∂Vol) with forward F, strike K, maturity T."""
    if sigma <= 0 or T <= 0 or not np.isfinite(sigma) or F <= 0 or K <= 0:
        return np.nan
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * np.sqrt(T))
    return F * norm.pdf(d1) * np.sqrt(T)


# --------------------------------------------------------------------------------------------------------
# Pathwise Black–Scholes pricing for Monte Carlo
# --------------------------------------------------------------------------------------------------------

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


def bs_put_vec_pathwise(F_path: np.ndarray, K_abs: float, T: float, sigma_path: np.ndarray) -> np.ndarray:
    """Pathwise Black–Scholes put price under the forward measure."""
    eps = 1e-12
    Fp = np.maximum(F_path, eps)
    sig = np.maximum(sigma_path, eps)
    d1 = (np.log(Fp / K_abs) + 0.5 * sig * sig * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return K_abs * norm.cdf(-d2) - Fp * norm.cdf(-d1)


# --------------------------------------------------------------------------------------------------------
# Sampling utilities
# --------------------------------------------------------------------------------------------------------

@dataclass
class RBergomiParams:
    eta: float
    rho: float
    H: float


def sample_param_sets_lhs(num_sets, rng, lower, upper):
    assert lower.shape == upper.shape
    d = len(lower)

    sampler_seed = int(rng.integers(0, 2**31 - 1))
    sampler = qmc.LatinHypercube(d=d, seed=sampler_seed)

    sample = sampler.random(num_sets)
    scaled = qmc.scale(sample, lower, upper)
    return scaled


# ============================================================
# HESTON PRICING (semi-close form, Gauss-Laguerre)
# ============================================================

# ----------------------------------------------------------
# 1) Characteristic function (same as yours, but vectorized)
# ----------------------------------------------------------
# def heston_cf(u, tau, kappa, theta, sigma, rho, v0, r=0.0):
#     """
#     Vektorisierte CF: u kann array sein!
#     """
#     iu = 1j * u

#     alpha = kappa - rho * sigma * iu
#     beta  = sigma * sigma
#     d = np.sqrt(alpha * alpha + beta * (u * u + iu))
#     d = np.where(np.real(d) < 0, -d, d)
#     g = (alpha - d) / (alpha + d)
#     g = g / (1 + 1e-12)

#     exp_dt = np.exp(-d * tau)
#     one_minus_gexp = 1 - g * exp_dt

#     C = (kappa * theta / beta) * ((alpha - d) * tau - 2 * np.log(one_minus_gexp / (1 - g)))
#     D = ((alpha - d) / beta) * ((1 - exp_dt) / one_minus_gexp)

#     return np.exp(C + D * v0 + iu * 0)   # Forward measure drift 0

# LAGUERRE_X, LAGUERRE_W = np.polynomial.laguerre.laggauss(64)

# def _fourier_probabilities(S0, K, tau, kappa, theta, v0, sigma, rho):
#     """
#     Vektorisierte Berechnung von P1 und P2.
#     """
#     x = LAGUERRE_X
#     w = LAGUERRE_W

#     phi = x          # Gauss-Laguerre-Knoten
#     exp_x = np.exp(x)

#     logS0 = np.log(S0)
#     logK  = np.log(K)

#     # -------- P1: u = phi - i ----------
#     u1 = phi - 1j
#     cf1 = heston_cf(u1, tau, kappa, theta, sigma, rho, v0)

#     integrand1 = np.real(
#         np.exp(1j * phi * (logS0 - logK)) * cf1 / (1j * phi)
#     )
#     P1 = 0.5 + (1/np.pi) * np.sum(w * exp_x * integrand1)

#     # -------- P2: u = phi ----------
#     u2 = phi
#     cf2 = heston_cf(u2, tau, kappa, theta, sigma, rho, v0)

#     integrand2 = np.real(
#         np.exp(1j * phi * (logS0 - logK)) * cf2 / (1j * phi)
#     )
#     P2 = 0.5 + (1/np.pi) * np.sum(w * exp_x * integrand2)
#     P1 = np.clip(P1, 0.0, 1.0)
#     P2 = np.clip(P2, 0.0, 1.0)
#     return P1, P2

# def heston_call_price(S0, K, tau, params, r=0.0):
#     kappa, theta, v0, sigma, rho = params

#     if tau <= 0:
#         return max(S0 - K, 0.0)

#     P1, P2 = _fourier_probabilities(S0, K, tau, kappa, theta, v0, sigma, rho)

#     call = S0 * P1 - K * np.exp(-r * tau) * P2

#     # Preislogik
#     call = np.clip(call, max(S0-K,0), S0)
#     return call


# def heston_put_price(S0, K, tau, params, r=0.0):
#     kappa, theta, v0, sigma, rho = params

#     if tau <= 0:
#         return max(K - S0, 0.0)

#     P1, P2 = _fourier_probabilities(S0, K, tau, kappa, theta, v0, sigma, rho)

#     # Put-Formel (klassisch)
#     put = K * np.exp(-r * tau) * (1 - P2) - S0 * (1 - P1)

#     put = np.clip(put, max(K-S0, 0), K)
#     return put


import numpy as np
import math

def heston_cf(u, tau, kappa, theta, sigma, rho, v0, r=0.0):
    """
    Standard-Heston-Charfunktion unter Forward/RN-Maß (drift = r - q, hier r).
    u: array oder scalar (Reell, aber komplexe Auswertung)
    """
    u = np.asarray(u, dtype=complex)

    alpha = kappa - rho * sigma * 1j * u
    beta  = sigma**2

    d = np.sqrt(alpha**2 + beta * (u**2 + 1j*u))
    g = (alpha - d) / (alpha + d)

    exp_dt = np.exp(-d * tau)
    num = 1 - g * exp_dt
    den = 1 - g

    # Numerik-Safeguards
    num = np.where(num == 0, 1e-16 + 0j, num)
    den = np.where(den == 0, 1e-16 + 0j, den)

    C = (kappa * theta / sigma**2) * ((alpha - d) * tau - 2 * np.log(num / den))
    D = (alpha - d) / sigma**2 * ((1 - exp_dt) / num)

    return np.exp(C + D * v0 + 1j * u * 0.0)  # forward measure: drift ~ 0

# Globale Laguerre-Knoten/Gewichte (64 reicht idR.)
LAG_N = 64
LAG_X, LAG_W = np.polynomial.laguerre.laggauss(LAG_N)

def heston_call_price(S0, K, tau, params, r=0.0,
                          x=LAG_X, w=LAG_W):
    """
    Vektorisierter Heston-Callpreis für EIN fixes tau, beliebig viele K.

    S0: float
    K:  float oder 1D-Array
    tau: float (Maturity)
    params: (kappa, theta, v0, sigma, rho)
    """
    kappa, theta, v0, sigma, rho = params

    K = np.asarray(K, dtype=float)  # shape (m,)
    tau = float(tau)

    logS0 = math.log(S0)
    logK  = np.log(K)               # (m,)

    u = x                           # (n,)
    weight = w * np.exp(x)          # (n,), Gauss–Laguerre-Korrektur

    # ---------- P1 ----------
    u1 = u - 1j
    cf1 = heston_cf(u1, tau, kappa, theta, sigma, rho, v0, r)  # (n,)

    # broadcast: (m,1) × (1,n)
    u_col     = u[None, :]         # (1,n)
    logK_col  = logK[:, None]      # (m,1)

    f1 = np.real(
        np.exp(1j * u_col * (logS0 - logK_col)) *
        cf1[None, :] / (1j * u_col)
    )                               # (m,n)

    P1 = 0.5 + (weight * f1).sum(axis=1) / math.pi  # (m,)

    # ---------- P2 ----------
    u2  = u
    cf2 = heston_cf(u2, tau, kappa, theta, sigma, rho, v0, r)  # (n,)

    f2 = np.real(
        np.exp(1j * u_col * (logS0 - logK_col)) *
        cf2[None, :] / (1j * u_col)
    )                               # (m,n)

    P2 = 0.5 + (weight * f2).sum(axis=1) / math.pi  # (m,)

    # ---------- Callpreis ----------
    disc = math.exp(-r * tau)
    call = S0 * P1 - K * disc * P2

    # Numerische Safeguards
    call = np.clip(call, 0.0, S0)

    return call

def heston_put_price(S0, K, tau, params, r=0.0,
                         x=LAG_X, w=LAG_W):
    """
    Getrennte Put-Funktion, aber basierend auf Call via Put-Call-Parität.
    """
    K = np.asarray(K, dtype=float)
    tau = float(tau)

    call = heston_call_price(S0, K, tau, params, r, x, w)
    disc = np.exp(-r * tau)

    put = call - S0 + K * disc
    put = np.clip(put, 0.0, K)   # Safeguard

    return put






from scipy.stats import qmc
import numpy as np

def lhs_grid(start, end, n, rng, min_spacing=0.02):
    """
    Generate a sorted 1D grid in [start, end] with:
    - first point fixed at start
    - last point fixed at end
    - (n-2) internal points sampled via LHS
    - minimal spacing between adjacent points enforced

    Works in log-space (recommended) and is monotone + deterministic.
    """
    start = float(start)
    end = float(end)

    if n < 2:
        return np.array([start], dtype=float)
    if n == 2:
        return np.array([start, end], dtype=float)

    # ---- 1) LHS internal points ----
    sampler = qmc.LatinHypercube(d=1, seed=rng.integers(1_000_000))
    internal = sampler.random(n - 2).flatten()   # uniform in [0,1]
    internal = start + internal * (end - start)
    internal.sort()

    # ---- 2) Combine with anchors ----
    grid = np.concatenate(([start], internal, [end]))

    # ---- 3) Enforce minimal spacing ----
    if min_spacing > 0:
        for i in range(1, n):
            if grid[i] - grid[i - 1] < min_spacing:
                grid[i] = grid[i - 1] + min_spacing

        # Ensure we do not push beyond the end anchor
        if grid[-1] > end:
            grid[-1] = end*0.95
        # Re-sort to maintain monotonicity
        grid.sort()

    return grid


def preprocess_and_filter_otm(df):
    # --- Datum parsen
    df["date"] = pd.to_datetime(df["date"])
    df["expiration"] = pd.to_datetime(df["expiration"])

    # --- Maturity in Jahren
    df["maturity"] = (df["expiration"] - df["date"]).dt.days / 365.0

    # --- Spot, der für die IV verwendet wurde
    df["S0"] = df["stock price for iv"].astype(float)

    # --- Strike als float
    df["strike"] = df["strike"].astype(float)

    # --- IV als float
    df["iv"] = df["iv"].astype(float)

    # --------------------------------------------------
    # --- OTM FILTER
    # --------------------------------------------------
    # OTM Call: strike > S0  AND call/put == "C"
    otm_calls = (df["call/put"] == "C") & (df["strike"] > df["S0"])

    # OTM Put: strike < S0  AND call/put == "P"
    otm_puts = (df["call/put"] == "P") & (df["strike"] < df["S0"])

    df = df[otm_calls | otm_puts].copy()

    # --------------------------------------------------
    # --- Log-Moneyness
    # --------------------------------------------------
    df["log_moneyness"] = np.log(df["strike"] / df["S0"])
    df["log_maturity"] = np.log(df["maturity"] )
    # --------------------------------------------------
    # --- Ungültige Werte entfernen
    # --------------------------------------------------
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["iv", "maturity", "log_moneyness"])
    df = df[(df["volume"]>100) & (df["maturity"]>25/365) & (df["maturity"]<2)&(df["log_moneyness"]>-0.4)&(df["log_moneyness"]<0.4)]
    # --------------------------------------------------
    # --- Output
    # --------------------------------------------------
    return df[["symbol","exchange","date","expiration","iv", "maturity","log_maturity", "log_moneyness", "volume", "strike", "S0"]]


def build_market_surfaces(df):
    """
    Create one IV surface per (symbol, expiration) pair.

    Each output element has:
    {
        "symbol": str,
        "expiration": pd.Timestamp,
        "grid": {
            "strikes": np.array(... log-moneyness ...)
            "maturities": np.array(... log-maturities ...)
        },
        "iv_surface": 2D array (nT x nK),
        "volume": 2D array (nT x nK),
        "weights": 2D array (nT x nK)
    }
    """

    output = []

    # -----------------------------------------
    # GROUP DATA BY (symbol, expiration)
    # -----------------------------------------
    grouped = df.groupby(["symbol", "date"])

    for (symbol, date), g in grouped:

        # sort maturities + log-moneyness
        maturities = np.sort(g["maturity"].unique())
        strikes = np.sort(g["log_moneyness"].unique())

        nT = len(maturities)
        nK = len(strikes)

        iv_surface = np.full((nT, nK), np.nan, dtype=np.float32)
        vol_surface = np.full((nT, nK), np.nan, dtype=np.float32)

        # fill surface
        for i, T in enumerate(maturities):
            sub = g[g["maturity"] == T]
            for _, row in sub.iterrows():
                K = row["log_moneyness"]
                iv = row["iv"]
                vol = row["volume"]
                j = np.searchsorted(strikes, K)
                if 0 <= j < nK:
                    iv_surface[i, j] = iv
                    vol_surface[i, j] = vol

        # log-domain grid (consistent with synthetic data)
        log_maturities = np.log(maturities).astype(np.float32)
        log_strikes = strikes.astype(np.float32)

        # volume weights (NaNs preserved!)
        weights = np.sqrt(np.log(vol_surface+1)).astype(np.float32)

        output.append({
            "symbol": symbol,
            "date": date,
            "grid": {
                "strikes": log_strikes,
                "maturities": log_maturities
            },
            "iv_surface": iv_surface,
            "volume": vol_surface,
            "weights": weights
        })

    return output




# --------------------------------------------------------------------------------------------------------
# Plotting utilities
# --------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def plot_iv_surface_scatter(
    iv_surface: np.ndarray,
    strikes: np.ndarray,
    maturities: np.ndarray,
    cmap: str = "plasma",
    figsize: Tuple[int, int] = (10, 6),
    title: str = "IV Surface (Scatter Only)",
    log_maturity: bool = True,
):
    """
    Scatter-Plot einer IV-Surface.

    Erwartete Konvention:
    - iv_surface.shape == (len(maturities), len(strikes))
    - Erste Zeile: kürzeste Maturity
    - Spalten: von kleinem Strike (log-moneyness) zu großem
    """
    iv_surface = np.asarray(iv_surface, float)
    strikes = np.asarray(strikes, float)
    maturities = np.asarray(maturities, float)

    assert iv_surface.shape == (len(maturities), len(strikes)), (
        f"Shape mismatch: iv_surface {iv_surface.shape} vs "
        f"(nT={len(maturities)}, nK={len(strikes)})"
    )

    # Grid der Punkte bauen (gleiche Indizierung wie iv_surface!)
    Kgrid, Tgrid = np.meshgrid(strikes, maturities, indexing="xy")

    # Nur valide Punkte plotten (keine NaNs)
    mask = np.isfinite(iv_surface)
    K_plot = Kgrid[mask]
    T_plot = Tgrid[mask]
    Z_plot = iv_surface[mask]

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(K_plot, T_plot, c=Z_plot, cmap=cmap, s=35)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Implied Volatility")

    ax.set_xlabel("Strike (log-moneyness)")
    ax.set_ylabel("Maturity (Years)")
    ax.set_title(title)

    if log_maturity:
        ax.set_yscale("log")

    plt.tight_layout()
    plt.show()



def plot_iv_surface(
    iv_surface: np.ndarray,
    strikes: np.ndarray,
    maturities: np.ndarray,
    xi0_knots: np.ndarray | None = None,
    xi0_bin_edges: np.ndarray | None = None,
    kind: str = "contour",
    cmap: str = "plasma",
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Implied Volatility Surface",
    log_maturity: bool = True,
):
    """
    Plot an implied volatility surface (contour or heatmap),
    optionally overlaying the piecewise-constant forward variance ξ₀(t).
    """
    maturities = np.asarray(maturities, float)
    strikes = np.asarray(strikes, float)
    Kgrid, Tgrid = np.meshgrid(strikes, maturities)

    fig, ax = plt.subplots(figsize=figsize)
    if kind == "heatmap":
        im = ax.imshow(
            iv_surface,
            extent=[strikes.min(), strikes.max(), maturities.min(), maturities.max()],
            origin="lower",
            aspect="auto",
            cmap=cmap,
        )
        fig.colorbar(im, ax=ax, label="Implied Volatility")
    elif kind == "contour":
        cs = ax.contourf(Kgrid, Tgrid, iv_surface, levels=20, cmap=cmap)
        fig.colorbar(cs, ax=ax, label="Implied Volatility")
    else:
        raise ValueError(f"Unknown kind '{kind}'")

    ax.set_xlabel("Strike")
    ax.set_ylabel("Maturity (Years)")
    ax.set_title(title)
    if log_maturity:
        ax.set_yscale("log")

    # Overlay ξ₀(t)
    if xi0_knots is not None and xi0_bin_edges is not None:
        xi0_knots = np.asarray(xi0_knots, float)
        edges = np.asarray(xi0_bin_edges, float)
        assert len(edges) == len(xi0_knots) + 1, "xi0_bin_edges must have length K+1."

        # Skip first bin [0, maturities[0]) → not visible
        first_vis_i = np.searchsorted(edges, maturities[0], side="left")
        first_vis_i = max(1, min(first_vis_i, len(xi0_knots) - 1))

        T_step, X_step = [], []
        for i in range(first_vis_i, len(xi0_knots)):
            t0, t1 = edges[i], edges[i + 1]
            v = xi0_knots[i]
            T_step += [t0, t1]
            X_step += [v, v]

        T_step = np.asarray(T_step)
        X_step = np.asarray(X_step)

        # Normalize overlay to appear near strike ≈ 1
        xnorm = (X_step - X_step.min()) / (X_step.max() - X_step.min() + 1e-12)
        xnorm = xnorm * (strikes.max() - strikes.min()) * 0.2
        xplot = 1.0 + xnorm

        ax.plot(
            xplot,
            T_step,
            color="white",
            lw=2.0,
            label=r"$\xi_0(t)$ (forward variance)",
        )
        ax.legend(loc="upper right", frameon=False)

    plt.tight_layout()
    plt.show()


import numpy as np
from scipy.ndimage import convolve
def repair_edges_local_directional(
        iv_surface: np.ndarray,
        maturities: np.ndarray,
        strikes: np.ndarray,
        t_threshold: float = 0.25,
        dtype=np.float32,
        min_floor: float = 0.015,
        protect_wings: int = 2
    ) -> np.ndarray:
    """
    Robust, wing-safe edge correction for short maturities.
    """
    iv = np.array(iv_surface, dtype=dtype, copy=True)
    nT, nK = iv.shape

    # ---------- 1) Row-wise fill ----------
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

    # ---------- 2) Column-wise fill ----------
    if np.isnan(iv).any():
        col_means = np.nanmean(iv, axis=0)
        for j in range(nK):
            nan_idx = np.isnan(iv[:, j])
            if np.any(nan_idx):
                iv[nan_idx, j] = col_means[j]
    iv = np.nan_to_num(iv, nan=np.nanmean(iv)).astype(dtype)

    # ---------- Convert maturities from log(T) if needed ----------
    T_phys = np.exp(maturities) if np.max(maturities) < 1 else maturities

    # Which rows are short-dated?
    short_mask = T_phys <= t_threshold
    if not np.any(short_mask):
        return iv

    short_iv = iv[short_mask]

    # ---------- 3) local smoothing kernel ----------
    kernel = np.array([[1, 2, 1],
                       [2, 4, 2],
                       [1, 2, 1]], dtype=np.float32)
    kernel /= kernel.sum()

    # Full convolution
    blurred = convolve(short_iv, kernel, mode="reflect")

    # ---------- 4) protect wings ----------

    low_vals = short_iv < min_floor
    short_iv[low_vals] = blurred[low_vals]

    # wings untouched
    iv[short_mask] = short_iv
    return iv.astype(dtype)



import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares, differential_evolution
import os, re, numpy as np, matplotlib.pyplot as plt

import os
import re
import numpy as np
import matplotlib.pyplot as plt


def _extract_param_names_consistent(results):
    """
    Determine a globally consistent parameter order across results.
    Priority:
      1) Use explicit order if provided in first result's dict.
      2) Otherwise infer from all dicts and apply natural sorting.
    """
    # Try to get order from first result (keeps mapping to array columns)
    for key in ["per_param_abs_errors", "per_param_rel_errors"]:
        if key in results[0]:
            return list(results[0][key].keys())

    # Otherwise infer from all and natural-sort
    all_keys = set()
    for res in results:
        for k in ["per_param_abs_errors", "per_param_rel_errors"]:
            if k in res:
                all_keys.update(res[k].keys())
    return sorted(
        list(all_keys),
        key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r"([0-9]+)", s)],
    )


def plot_param_true_vs_est(results, labels, out_dir="calibration_plots", alpha=0.6):
    """Compare true vs estimated parameters across multiple models."""
    os.makedirs(out_dir, exist_ok=True)
    colors = plt.cm.tab10.colors

    # unified parameter order across all plots
    all_param_names = _extract_param_names_consistent(results)

    n_params = len(all_param_names)
    ncols = min(4, n_params)
    nrows = int(np.ceil(n_params / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

    for i, param in enumerate(all_param_names):
        ax = axes[i // ncols, i % ncols]
        for j, (res, label) in enumerate(zip(results, labels)):
            if i >= res["true_params"].shape[1]:
                continue
            true_vals = res["true_params"][:, i]
            est_vals = res["est_params"][:, i]
            ax.scatter(true_vals, est_vals, alpha=alpha, s=12,
                       color=colors[j % len(colors)], label=label, edgecolors="none")

        all_true = np.concatenate(
            [r["true_params"][:, i] for r in results if i < r["true_params"].shape[1]]
        )
        lo, hi = np.nanmin(all_true), np.nanmax(all_true)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(param, fontsize=14)
        ax.set_xlabel("True")
        ax.set_ylabel("Estimated")
        if i == 0:
            ax.legend(frameon=True, loc="upper left")

    # deactivate empty subplots
    for j in range(i + 1, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    _tight_suptitle(fig, "True vs Estimated Parameters (Calibration Comparison)")
    path = os.path.join(out_dir, "param_true_vs_est.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved scatter comparison to {path}")


def plot_param_error_ecdfs(
    results,
    labels,
    out_dir="calibration_plots",
    kind="relative",
    cut_quantile=None,
):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    os.makedirs(out_dir, exist_ok=True)

    if cut_quantile is not None:
        assert 0 < cut_quantile <= 1

    n_models = len(results)
    add_diff = (n_models == 2)

    # -------- collect data --------
    param_sets, rmses_sets = [], []
    for r in results:
        rmses_sets.append(r.get("rmses", []))

        if kind == "absolute":
            param_sets.append(r["per_param_abs_errors"])
        else:
            if "per_param_rel_errors" in r:
                param_sets.append(r["per_param_rel_errors"])
            else:
                param_sets.append(r["per_param_errors"])

    all_param_names = _extract_param_names_consistent(results)
    num_params = len(all_param_names)

    # -------- layout: parameters + RMSE panel --------
    total_plots = num_params + 1
    ncols = min(4, num_params)
    nrows = int(np.ceil(total_plots / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), squeeze=False)

    def ecdf_cut(x):
        xs = np.sort(np.asarray(x))
        ys = np.linspace(0, 1, len(xs))
        if cut_quantile is not None:
            idx = int(cut_quantile * len(xs))
            xs = xs[:idx]
            ys = ys[:idx]
        return xs, ys

    # --------------------------
    #  PARAMETER PANELS
    # --------------------------
    for i, param in enumerate(all_param_names):
        ax = axes[i // ncols, i % ncols]

        curves = []  # store (ys, xs_scaled)
        for data, label in zip(param_sets, labels):
            if param not in data:
                curves.append(None)
                continue

            xs, ys = ecdf_cut(data[param])
            xs_scaled = xs * (100 if kind == "relative" else 1)

            curves.append((ys, xs_scaled))
            ax.plot(ys, xs_scaled, label=label)

        # ----- optional: difference curve -----
        if add_diff:
            ref = curves[0]
            other = curves[1]

            if ref is not None and other is not None:
                ys_ref, xs_ref = ref
                ys_i, xs_i = other

                # Interpolate other onto ref-ys grid
                xs_i_interp = np.interp(ys_ref, ys_i, xs_i)
                diff = xs_i_interp - xs_ref

                ax.plot(
                    ys_ref,
                    diff,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.8,
                    label=f"{labels[1]} – {labels[0]} (Δ)"
                )

        ax.set_title(param, fontsize=13)
        ax.set_xlabel(
            "Quantiles" + (f" (cut @ {cut_quantile:.2%})" if cut_quantile else "")
        )
        ax.set_ylabel(
            f"{kind.title()} Error" + (" [%]" if kind == "relative" else "")
        )

        if i == 0:
            ax.legend(frameon=True, loc="upper left")

    # --------------------------
    #  RMSE PANEL
    # --------------------------
    ax_rmse = axes[num_params // ncols, num_params % ncols]

    rmse_curves = []
    for rmses, label in zip(rmses_sets, labels):
        if len(rmses) == 0:
            rmse_curves.append(None)
            continue

        xs, ys = ecdf_cut(rmses)
        rmse_curves.append((ys, xs))
        ax_rmse.plot(ys, xs, label=label)

    # optional difference curve
    if add_diff:
        ref = rmse_curves[0]
        other = rmse_curves[1]
        if ref is not None and other is not None:
            ys_ref, xs_ref = ref
            ys_i, xs_i = other
            xs_i_interp = np.interp(ys_ref, ys_i, xs_i)
            diff = xs_i_interp - xs_ref

            ax_rmse.plot(
                ys_ref,
                diff,
                linestyle="--",
                linewidth=1.0,
                alpha=0.8,
                label=f"{labels[1]} – {labels[0]} (Δ)"
            )

    ax_rmse.set_title("RMSE", fontsize=13)
    ax_rmse.set_xlabel(
        "Quantiles" + (f" (cut @ {cut_quantile:.2%})" if cut_quantile else "")
    )
    ax_rmse.set_ylabel("RMSE")

    # Turn off unused subplots
    for idx in range(total_plots, nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")

    # Title
    _tight_suptitle(fig, f"Parameter {kind.title()} Errors + RMSE (ECDFs)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(out_dir, f"param_and_rmse_ecdfs_{kind}.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved combined ECDF comparison to {out_path}")







import os, numpy as np, matplotlib.pyplot as plt, copy
import numpy as np
import matplotlib.pyplot as plt
import os

import os
import numpy as np
import matplotlib.pyplot as plt

def flatten_params(params, prefix=""):
    """
    Recursively flattens a nested dict/list structure into
    {flat_key: scalar_value}.
    Example:
        {"a":1, "b":[1,2], "c":{"x":3}}
    ⇒ {
        "a":1,
        "b_0":1,
        "b_1":2,
        "c_x":3
    }
    """
    flat = {}

    if isinstance(params, dict):
        for k, v in params.items():
            new_prefix = f"{prefix}{k}" if prefix == "" else f"{prefix}_{k}"
            flat.update(flatten_params(v, new_prefix))

    elif isinstance(params, (list, tuple)):
        for i, v in enumerate(params):
            new_prefix = f"{prefix}_{i}"
            flat.update(flatten_params(v, new_prefix))

    else:
        # assume scalar numeric value
        flat[prefix] = params

    return flat



def plot_param_histograms(surface_samples, out_dir="param_histograms"):
    os.makedirs(out_dir, exist_ok=True)

    # ---- Collect all flattened params ----
    collected = {}

    for s in surface_samples:
        flat = flatten_params(s["params"])
        for k, v in flat.items():
            collected.setdefault(k, []).append(v)

    # Convert to numpy arrays
    for k in collected:
        collected[k] = np.array(collected[k])

    # ---- Plot each histogram individually ----
    for name, data in collected.items():
        plt.figure(figsize=(6, 4))
        plt.hist(data, bins=40, alpha=0.8)
        plt.title(f"Histogram of {name}")
        plt.xlabel(name)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"hist_{name}.png"), dpi=200)
        plt.close()

    # ---- Multi-grid summary plot ----
    all_params = list(collected.items())
    cols = 3
    rows = int(np.ceil(len(all_params) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for ax, (name, data) in zip(axes, all_params):
        ax.hist(data, bins=40, alpha=0.8)
        ax.set_title(name)
        ax.set_xlabel(name)
        ax.set_ylabel("Count")

    # turn off empty axes
    for j in range(len(all_params), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "all_parameters_hist.png"), dpi=200)
    plt.close(fig)

    print(f"Saved {len(all_params)} histograms to: {out_dir}")




import numpy as np

def count_local_maxima_1d(arr):
    """
    Count strict local minima in a 1D array.
    A local minimum is an index i such that arr[i] < arr[i-1] and arr[i] < arr[i+1].
    """
    if len(arr) < 3:
        return 0

    return int(np.sum((arr[1:-1] > arr[:-2]) & (arr[1:-1] > arr[2:])))

def surface_has_too_many_maxima(surface, max_minima=3):
    """
    Returns True if ANY maturity has more than `max_minima` local minima.
    surface["iv_surface"] is assumed to be shape (nT, nK).
    """
    iv = surface["iv_surface"]

    for row in iv:
        minima_count = count_local_maxima_1d(row)
        if minima_count > max_minima:
            return True

    return False

import numpy as np
import random
import numpy as np
import random

def sample_uniform_param(surface_list, param_name="v0", n_samples=40000, n_bins=50):
    param_vals = np.array([s["params"][param_name] for s in surface_list])

    # ---- Safe bins: include max properly ----
    p_min, p_max = param_vals.min(), param_vals.max()
    eps = (p_max - p_min) * 1e-9
    bins = np.linspace(p_min, p_max + eps, n_bins + 1)

    # ---- Digitize ----
    bin_indices = np.digitize(param_vals, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # ---- Build mapping ----
    bin_to_indices = {b: [] for b in range(n_bins)}
    for idx, b in enumerate(bin_indices):
        bin_to_indices[b].append(idx)

    # ---- Target per bin ----
    samples_per_bin = n_samples // n_bins
    selected = []

    for b in range(n_bins):
        indices = bin_to_indices[b]
        if len(indices) <= samples_per_bin:
            selected.extend(indices)
        else:
            selected.extend(random.sample(indices, samples_per_bin))

    # Fill leftover
    if len(selected) < n_samples:
        remaining = n_samples - len(selected)
        leftovers = list(set(range(len(surface_list))) - set(selected))
        selected.extend(random.sample(leftovers, remaining))
    random.shuffle(selected)
    return [surface_list[i] for i in selected]


# Proper names for your params
PARAM_NAMES = ["eta", "rho", "H"] + [f"xi0_{i}" for i in range(5)]


def prepare_df(res_list, name, PARAM_NAMES):
    df = pd.DataFrame([{
        "date": r["date"],
        "rmse": r["rmse"],
        "runtime_ms": r["runtime_ms"],
        **{PARAM_NAMES[i]: r["theta_hat"][i] for i in range(len(r["theta_hat"]))}
    } for r in res_list])

    df["model"] = name
    df["date"] = pd.to_datetime(df["date"])
    return df


def plot_param_grid(results, labels, PARAM_NAMES, out_dir=None):
    """
    Plot time-series of parameter estimates, RMSE, and runtime
    for ANY number of models.

    results: list of result dicts (each like output from evaluate_calibrate)
    labels:  list of model names, same length as results
    PARAM_NAMES: list of param names in correct order
    """

    # ---------- Build DataFrame ----------
    dfs = []
    for res, name in zip(results, labels):
        df_sub = prepare_df(res, name, PARAM_NAMES)
        dfs.append(df_sub)

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("date")

    # ---------- Model list ----------
    model_names = df["model"].unique().tolist()

    # ---------- Layout ----------
    param_cols = PARAM_NAMES
    n_params = len(param_cols)

    # Grid: 2 columns for parameters
    cols = 2
    rows = math.ceil(n_params / cols) + 1  # +1 for RMSE & Runtime panel

    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 3), squeeze=False)
    axes = axes.flatten()

    # Colors per model
    color_cycle = plt.cm.tab10.colors
    model_to_color = {m: color_cycle[i % len(color_cycle)] for i, m in enumerate(model_names)}

    # ---------- Parameter plots ----------
    for idx, p in enumerate(param_cols):
        ax = axes[idx]

        for model in model_names:
            sub = df[df["model"] == model]
            if p in sub.columns:
                ax.plot(
                    sub["date"], sub[p],
                    marker="o",
                    label=model if idx == 0 else None,
                    color=model_to_color[model]
                )

        ax.set_title(p)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.grid(True)

    # Add legend only once
    axes[0].legend(frameon=True)

    # ---------- RMSE ----------
    rmse_ax = axes[n_params]
    for model in model_names:
        sub = df[df["model"] == model]
        rmse_ax.plot(
            sub["date"], sub["rmse"],
            marker="o",
            color=model_to_color[model],
            label=model
        )

    rmse_ax.set_title("Weighted RMSE over Time")
    rmse_ax.set_xlabel("Date")
    rmse_ax.set_ylabel("RMSE")
    rmse_ax.grid(True)

    # ---------- Runtime ----------
    runtime_ax = axes[n_params + 1]
    for model in model_names:
        sub = df[df["model"] == model]
        runtime_ax.plot(
            sub["date"], sub["runtime_ms"],
            marker="o",
            color=model_to_color[model],
            label=model
        )

    runtime_ax.set_title("Calibration Runtime (ms)")
    runtime_ax.set_xlabel("Date")
    runtime_ax.set_ylabel("Runtime [ms]")
    runtime_ax.set_yscale("log")
    runtime_ax.grid(True)

    # ---------- Turn off unused axes ----------
    for j in range(n_params + 2, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "param_grid_plot.png")
        plt.savefig(path, dpi=200)
        print(f"Saved plot to {path}")
    else:
        plt.show()

    # ---------- Print summaries ----------
    print("\nAverage Calibration Runtime (ms):")
    print(df.groupby("model")["runtime_ms"].mean())

    print("\nAverage RMSE:")
    print(df.groupby("model")["rmse"].mean())
