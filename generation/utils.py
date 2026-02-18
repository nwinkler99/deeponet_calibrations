# ====
# Utility functions for Rough Bergomi model and IV surface operations
# ====

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
    """Add a large suptitle closer to subplots."""
    fig.suptitle(title, fontsize=fontsize, weight="bold", y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

# ====
# TBSS and rBergomi helpers
# ====

def g(x: float, a: float) -> float:
    """TBSS kernel for rBergomi variance process."""
    return x ** a


def b(k: int, a: float) -> float:
    """Optimal discretization of TBSS process."""
    return ((k ** (a + 1) - (k - 1) ** (a + 1)) / (a + 1)) ** (1 / a)


def cov(a: float, n: int) -> np.ndarray:
    """Covariance matrix for TBSS with alpha=a and n discretization steps."""
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
    Black option price in money-market-deflated (discounted) units.

    Parameters
    ----------
    F : float
        Deflated underlying value 𝑆̃ (often normalized, e.g. S̃₀ = 1).
    K : float
        Deflated strike 𝑲̃.
    V : float
        Integrated variance ∫₀ᵀ σ_t² dt.
    o : str, optional
        Option type: "call", "put", or "otm".

    Notes
    -----
    - All quantities are interpreted in deflated (discounted) units.
    - No interest rate or discount factor appears in this pricing formula.
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
    """Robust implied Black volatility inversion in deflated units.
    
    Parameters
    ----------
    P : float
        Deflated option price.
    F : float
        Deflated underlying value.
    K : float
        Deflated strike.
    t : float
        Maturity in years.
    o : str
        Option type: 'call', 'put', or 'otm'.
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
    """Black vega (price sensitivity to volatility) in deflated units.
    
    Parameters
    ----------
    F : float
        Deflated underlying value.
    K : float
        Deflated strike.
    T : float
        Maturity in years.
    sigma : float
        Volatility level.
    """
    if sigma <= 0 or T <= 0 or not np.isfinite(sigma) or F <= 0 or K <= 0:
        return np.nan
    d1 = (np.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * np.sqrt(T))
    return F * norm.pdf(d1) * np.sqrt(T)


# ====
# Pathwise Black pricing for Monte Carlo
# ====

def bs_call_vec_pathwise(F_path: np.ndarray, K_abs: float, T: float, sigma_path: np.ndarray) -> np.ndarray:
    """Pathwise Black call prices in deflated units.
    
    Parameters
    ----------
    F_path : np.ndarray
        Deflated terminal underlying values along paths.
    K_abs : float
        Deflated strike.
    T : float
        Maturity in years.
    sigma_path : np.ndarray
        Pathwise conditional volatility.
    
    Returns
    -------
    np.ndarray
        Call prices along each Monte Carlo path.
    """
    eps = 1e-12
    Fp = np.maximum(F_path, eps)
    sig = np.maximum(sigma_path, eps)
    d1 = (np.log(Fp / K_abs) + 0.5 * sig * sig * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return Fp * norm.cdf(d1) - K_abs * norm.cdf(d2)


def bs_put_vec_pathwise(F_path: np.ndarray, K_abs: float, T: float, sigma_path: np.ndarray) -> np.ndarray:
    """Pathwise Black put prices in deflated units.
    
    Parameters
    ----------
    F_path : np.ndarray
        Deflated terminal underlying values along paths.
    K_abs : float
        Deflated strike.
    T : float
        Maturity in years.
    sigma_path : np.ndarray
        Pathwise conditional volatility.
    
    Returns
    -------
    np.ndarray
        Put prices along each Monte Carlo path.
    """
    eps = 1e-12
    Fp = np.maximum(F_path, eps)
    sig = np.maximum(sigma_path, eps)
    d1 = (np.log(Fp / K_abs) + 0.5 * sig * sig * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return K_abs * norm.cdf(-d2) - Fp * norm.cdf(-d1)


# ====
# Sampling utilities
# ====

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





import numpy as np
import math

from scipy.stats import qmc
import numpy as np

def lhs_grid(start, end, n, rng, min_spacing=0.02):
    """Generate sorted 1D grid with LHS sampling and minimal spacing.
    
    Parameters
    ----------
    start : float
        Grid start point.
    end : float
        Grid end point.
    n : int
        Number of grid points.
    rng : np.random.Generator
        Random number generator.
    min_spacing : float
        Minimal spacing between adjacent points.
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


import numpy as np
import pandas as pd


def load_effr_rates_csv(path: str) -> pd.DataFrame:
    """Load EFFR rates from CSV with observation_date and EFFR columns.
    
    Converts EFFR from percent to decimal and forward-fills holidays.
    """
    rates = pd.read_csv(path)
    rates.columns = rates.columns.str.strip()

    rates["date"] = pd.to_datetime(rates["observation_date"])
    rates["effr"] = pd.to_numeric(rates["EFFR"], errors="coerce") / 100.0  # percent -> decimal
    rates = rates.sort_values("date")

    # forward fill missing rates (e.g., holidays) using last available business day
    rates["effr"] = rates["effr"].ffill()

    return rates[["date", "effr"]]


def preprocess_and_filter_otm(df: pd.DataFrame,
                                       rates: pd.DataFrame,
                                       q: float = 0.016) -> pd.DataFrame:
    """Preprocess market data and filter for OTM options.
    
    Uses EFFR as interest rate and constant dividend yield to compute
    forward prices. Filters for OTM strikes and valid maturity/moneyness ranges.
    """

    df = df.copy()
    df.columns = df.columns.str.strip()

    # Parse dates
    df["date"] = pd.to_datetime(df["date"])
    df["expiration"] = pd.to_datetime(df["expiration"])

    # Maturity in years
    df["maturity"] = (df["expiration"] - df["date"]).dt.days / 365.0

    # Numerics
    df["S0"] = pd.to_numeric(df["stock price for iv"], errors="coerce")
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["iv"] = pd.to_numeric(df["iv"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

    # Clean early
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["S0", "strike", "iv", "maturity"])
    df = df[(df["iv"] > 0) & (df["maturity"] > 0)]  # your file uses iv=-1 for missing

    # Join rates (last available <= date)
    df = df.sort_values("date")
    rates = rates.sort_values("date")
    df = pd.merge_asof(df, rates, on="date", direction="backward")

    # If some early dates precede first rate row, you can backfill once
    df["effr"] = df["effr"].bfill()

    # Forward approximation
    df["F"] = df["S0"] * np.exp((df["effr"] - q) * df["maturity"])

    # OTM relative to forward
    otm_calls = (df["call/put"] == "C") & (df["strike"] > df["F"])
    otm_puts  = (df["call/put"] == "P") & (df["strike"] < df["F"])
    df = df[otm_calls | otm_puts].copy()

    # Log coordinates
    df["log_moneyness"] = np.log(df["strike"] / df["F"])
    df["log_maturity"] = np.log(df["maturity"])

    # Final clean + your filters
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["iv", "maturity", "log_moneyness", "log_maturity"])
    df = df[
    # --- Maturity window ---
    (df["maturity"] > 25 / 365)
    & (df["maturity"] < 2.0)

    # --- Moneyness window ---
    & (df["log_moneyness"] > -0.4)
    & (df["log_moneyness"] < 0.4)

    # --- Liquidity ---
    & ((df["volume"] > 100) | (df["open interest"] > 1000))

    # --- Valid quotes ---
    & (df["bid"] > 0)
    & (df["ask"] > 0)
    & (df["ask"] >= df["bid"])

    # --- Spread control ---
    & ((df["ask"] - df["bid"]) / df["price"] < 0.05)

    # --- Price sanity ---

    # --- IV sanity ---
    & (df["iv"] > 0.01)
    & (df["iv"] < 3.0)
    ]

    return df[[
        "symbol", "exchange", "date", "expiration", "iv",
        "maturity", "log_maturity", "log_moneyness",
        "volume", "strike", "S0", "F", "effr"
    ]]



def build_market_surfaces(df):
    """Build IV surfaces from market data grouped by symbol and date.
    
    Returns list of dicts with keys: symbol, date, grid, iv_surface, volume, weights.
    """

    output = []

    # Group data by (symbol, date)
    grouped = df.groupby(["symbol", "date"])

    for (symbol, date), g in grouped:

        # Sort maturities and moneyness
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




# ====
# Plotting utilities
# ====
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
    """Scatter plot of IV surface.
    
    Parameters
    ----------
    iv_surface : np.ndarray
        2D array of shape (nT, nK).
    strikes : np.ndarray
        Strike grid (log-moneyness).
    maturities : np.ndarray
        Maturity grid.
    cmap : str
        Colormap name.
    figsize : tuple
        Figure size.
    title : str
        Plot title.
    log_maturity : bool
        Whether to use log scale for maturity axis.
    """
    iv_surface = np.asarray(iv_surface, float)
    strikes = np.asarray(strikes, float)
    maturities = np.asarray(maturities, float)

    assert iv_surface.shape == (len(maturities), len(strikes)), (
        f"Shape mismatch: iv_surface {iv_surface.shape} vs "
        f"(nT={len(maturities)}, nK={len(strikes)})"
    )

    # Create grid
    Kgrid, Tgrid = np.meshgrid(strikes, maturities, indexing="xy")

    # Plot only finite points
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
    """Plot IV surface as contour or heatmap with optional xi0 overlay.
    
    Parameters
    ----------
    iv_surface : np.ndarray
        2D array of IV values.
    strikes : np.ndarray
        Strike grid.
    maturities : np.ndarray
        Maturity grid.
    xi0_knots : np.ndarray, optional
        Forward variance values for overlay.
    xi0_bin_edges : np.ndarray, optional
        Bin edges for xi0 piecewise constant.
    kind : str
        'contour' or 'heatmap'.
    cmap : str
        Colormap name.
    figsize : tuple
        Figure size.
    title : str
        Plot title.
    log_maturity : bool
        Use log scale for maturity axis.
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

        # Skip first bin [0, maturities[0]) - not visible
        first_vis_i = np.searchsorted(edges, maturities[0], side="left")
        first_vis_i = max(1, min(first_vis_i, len(xi0_knots) - 1))

        # Build piecewise constant overlay
        T_step, X_step = [], []
        for i in range(first_vis_i, len(xi0_knots)):
            t0, t1 = edges[i], edges[i + 1]
            v = xi0_knots[i]
            T_step += [t0, t1]
            X_step += [v, v]

        T_step = np.asarray(T_step)
        X_step = np.asarray(X_step)

        # Normalize overlay to strike scale
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
    #_tight_suptitle(fig, f"Parameter {kind.title()} Errors + RMSE (ECDFs)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = os.path.join(out_dir, f"param_and_rmse_ecdfs_{kind}.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved combined ECDF comparison to {out_path}")


def plot_grid_median_comparison(
    cal_evals,
    labels,
    out_dir,
    strikes,
    maturities,
    field="pred_rel",
    stat="median",
    fname=None,
    unit="%",
):
    """
    Compare two models' grid statistics (typically median errors) in a 1x3 plot:
      col1: model 1
      col2: model 2
      col3: difference (model1 - model2)

    Parameters
    ----------
    cal_evals : list
        List with exactly two evaluation dicts, e.g. [deeponet_cal_eval, mlp_cal_eval].
        Each dict must contain cal_eval[field][stat] as a 2D array of shape (nT, nK).
    labels : list
        Two labels for the models, e.g. ["DeepONet", "MLP (bil. interp.)"].
    out_dir : str
        Output directory where the figure is saved.
    strikes : array-like
        Strike grid (x-axis), e.g. deeponet_loaded.strikes.
    maturities : array-like
        Maturity grid (y-axis), e.g. deeponet_loaded.maturities.
    field : str
        Key under which the binned stats are stored (default: "pred_rel").
    stat : str
        Statistic name to plot (default: "median").
    fname : str or None
        Output filename. If None, a default is created from field/stat.
    unit : str
        Colorbar label unit (default: "%").
    """
    if not isinstance(cal_evals, (list, tuple)) or len(cal_evals) != 2:
        raise ValueError("cal_evals must be a list/tuple of length 2.")
    if not isinstance(labels, (list, tuple)) or len(labels) != 2:
        raise ValueError("labels must be a list/tuple of length 2.")

    os.makedirs(out_dir, exist_ok=True)

    # Extract grids
    e1, e2 = cal_evals
    try:
        A = np.asarray(e1[field][stat], dtype=np.float32)
        B = np.asarray(e2[field][stat], dtype=np.float32)
    except Exception as exc:
        raise KeyError(
            f"Expected eval dict structure eval[field][stat], got field='{field}', stat='{stat}'."
        ) from exc

    if A.shape != B.shape:
        raise ValueError(f"Grid shapes differ: {A.shape} vs {B.shape}")

    strikes = np.asarray(strikes, dtype=np.float32)
    maturities = np.asarray(maturities, dtype=np.float32)

    if A.shape != (len(maturities), len(strikes)):
        raise ValueError(
            f"Grid shape {A.shape} does not match (len(maturities), len(strikes))="
            f"({len(maturities)}, {len(strikes)})"
        )

    D = A - B  # difference of medians

    # Mesh
    Ks_mesh, Ts_mesh = np.meshgrid(strikes, maturities, indexing="xy")

    # Use shared limits for fair comparison in col1/col2
    vmin = float(np.nanmin([np.nanmin(A), np.nanmin(B)]))
    vmax = float(np.nanmax([np.nanmax(A), np.nanmax(B)]))

    # Symmetric limits for diff
    dmax = float(np.nanmax(np.abs(D)))
    if not np.isfinite(dmax) or dmax == 0.0:
        dmax = 1.0
    dvmin, dvmax = -dmax, dmax

    if fname is None:
        fname = f"{field}_{stat}_comparison.png"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Col 1: Model 1
    im0 = axes[0].pcolormesh(Ks_mesh, Ts_mesh, A, shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
    axes[0].set_title(labels[0])
    axes[0].set_xlabel("Strike (K)")
    axes[0].set_ylabel("Maturity (T)")
    axes[0].invert_yaxis()
    fig.colorbar(im0, ax=axes[0], label=unit)

    # Col 2: Model 2
    im1 = axes[1].pcolormesh(Ks_mesh, Ts_mesh, B, shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
    axes[1].set_title(labels[1])
    axes[1].set_xlabel("Strike (K)")
    axes[1].set_ylabel("Maturity (T)")
    axes[1].invert_yaxis()
    fig.colorbar(im1, ax=axes[1], label=unit)

    # Col 3: Difference
    im2 = axes[2].pcolormesh(Ks_mesh, Ts_mesh, D, shading="auto", cmap="RdBu_r", vmin=dvmin, vmax=dvmax)
    axes[2].set_title(f"Difference ({labels[0]} - {labels[1]})")
    axes[2].set_xlabel("Strike (K)")
    axes[2].set_ylabel("Maturity (T)")
    axes[2].invert_yaxis()
    fig.colorbar(im2, ax=axes[2], label=unit)

    title = (
    f"Median Relative IV Error across Strike–Maturity Grid\n"
    f"{labels[0]} vs. {labels[1]} (and Difference)"
    )
    #fig.suptitle(title, fontsize=14, y=1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(out_dir, fname), dpi=200)
    plt.close(fig)





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
