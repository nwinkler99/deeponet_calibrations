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
        return brentq(error, 1e-8, 5.0, xtol=1e-10, maxiter=100)
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


def sample_param_sets_lhs(num_sets: int, rng: np.random.RandomState) -> List[RBergomiParams]:
    """Sample (eta, rho, H) via Latin Hypercube, fully tied to rng."""
    sampler_seed = rng.randint(0, 2**31 - 1)
    sampler = qmc.LatinHypercube(d=3, seed=sampler_seed)
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


def jitter_grid(base_grid: np.ndarray, grid_jitter: float = 0.5, min_spacing: float = 0.05) -> np.ndarray:
    """
    Randomly perturb a base grid while keeping minimum spacing.
    Returns float64; caller can cast to cfg.dtype.
    """
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


# --------------------------------------------------------------------------------------------------------
# Plotting utilities
# --------------------------------------------------------------------------------------------------------

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


import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares, differential_evolution


import re

def _natural_key(name: str):
    """
    Sort helper: 'xi0_10' -> ('xi0_', 10), 'eta' -> ('eta', -1)
    Ensures xi0_2 < xi0_10.
    """
    m = re.match(r"^([A-Za-z_]+)(\d+)$", name)
    if m:
        return (m.group(1), int(m.group(2)))
    return (name, -1)


def plot_param_error_ecdfs(error_dicts, labels, out_dir="calibration_plots", kind="relative"):
    """
    Compare ECDFs of per-parameter errors across multiple models (natural order).
    """
    os.makedirs(out_dir, exist_ok=True)

    # Natural / numeric order
    all_params = sorted({k for d in error_dicts for k in d.keys()}, key=_natural_key)

    ncols = min(4, len(all_params))
    nrows = int(np.ceil(len(all_params) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    def ecdf(x):
        xs = np.sort(x)
        ys = np.linspace(0, 1, len(x))
        return xs, ys

    for i, param in enumerate(all_params):
        ax = axes[i // ncols, i % ncols]
        for errs, label in zip(error_dicts, labels):
            if param in errs:
                xs, ys = ecdf(errs[param])
                ax.plot(ys, xs * (100 if kind == "relative" else 1), label=label)
        ax.set_title(param)
        ax.set_xlabel("Quantiles")
        ax.set_ylabel(f"{kind.title()} Error" + (" [%]" if kind == "relative" else ""))
        ax.grid(True, ls=":", lw=0.5)
        ax.legend(fontsize=8)

    for j in range(i + 1, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    plt.suptitle(f"Parameter {kind.title()} Error CDFs")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(out_dir, f"param_error_cdfs_{kind}.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {kind} error ECDF comparison to {path}")


def plot_param_true_vs_est(
    results,
    labels,
    out_dir="calibration_plots",
    alpha=0.6,
):
    """
    Compare true vs estimated parameters across multiple models.

    Parameters
    ----------
    results : list[dict]
        Each dict must contain:
            - 'true_params': (N, d) array
            - 'est_params': (N, d) array
            - optionally 'per_param_abs_errors' or 'per_param_rel_errors' (for param names)
    labels : list[str]
        Model names corresponding to each result dict.
    out_dir : str
        Output directory for saved figures.
    alpha : float
        Scatter transparency for overlapping points.
    """
    import os, numpy as np, matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # Determine parameter names (prefer consistent naming if present)
    all_param_names = None
    for res in results:
        if "per_param_abs_errors" in res:
            all_param_names = list(res["per_param_abs_errors"].keys())
            break
        elif "per_param_rel_errors" in res:
            all_param_names = list(res["per_param_rel_errors"].keys())
            break

    if all_param_names is None:
        # fallback: infer names as generic indices
        max_dim = max(r["true_params"].shape[1] for r in results)
        all_param_names = [f"param_{i}" for i in range(max_dim)]

    n_params = len(all_param_names)
    ncols = min(4, n_params)
    nrows = int(np.ceil(n_params / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

    colors = plt.cm.tab10.colors  # distinct color palette

    for i, param in enumerate(all_param_names):
        ax = axes[i // ncols, i % ncols]

        for j, (res, label) in enumerate(zip(results, labels)):
            if i >= res["true_params"].shape[1]:
                continue  # skip if model has fewer parameters

            true_vals = res["true_params"][:, i]
            est_vals = res["est_params"][:, i]

            ax.scatter(
                true_vals,
                est_vals,
                alpha=alpha,
                s=12,
                color=colors[j % len(colors)],
                label=label,
                edgecolors="none",
            )

        # draw perfect-fit line
        all_true = np.concatenate(
            [r["true_params"][:, i] for r in results if i < r["true_params"].shape[1]]
        )
        lo, hi = np.nanmin(all_true), np.nanmax(all_true)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="Perfect fit")

        ax.set_title(param)
        ax.set_xlabel("True")
        ax.set_ylabel("Estimated")
        ax.grid(True, ls=":", lw=0.5)
        ax.legend(fontsize=8)

    # turn off unused subplots
    for j in range(i + 1, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    plt.suptitle("True vs Estimated Parameters (Calibration Comparison)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, "param_true_vs_est.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved scatter comparison to {path}")

