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



def jitter_grid(x, grid_jitter=0.5, min_spacing=0.02, rng=None):
    """
    Apply small random jitter to grid points while preserving spacing.
    Deterministic if rng is provided.
    """
    rng = rng or default_rng()
    x = np.array(x, dtype=float, copy=True)
    n = len(x)
    if n < 2:
        return x

    step = np.median(np.diff(x))
    jitter = rng.normal(0.0, grid_jitter * step, size=n)
    y = np.clip(x + jitter, x.min(), x.max())
    y.sort()

    # enforce minimal spacing
    for i in range(1, n):
        if y[i] - y[i - 1] < min_spacing:
            y[i] = y[i - 1] + min_spacing
    return y


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

import os, re, numpy as np, matplotlib.pyplot as plt
def _natural_key(name: str):
    """Sort helper: ensures xi0_2 < xi0_10, while keeping eta, rho, H first."""
    m = re.match(r"^([A-Za-z_]+)(\d+)$", name)
    if m:
        return (m.group(1), int(m.group(2)))
    return (name, -1)


def plot_param_true_vs_est(results, labels, out_dir="calibration_plots", alpha=0.6):
    """Compare true vs estimated parameters across multiple models."""
    os.makedirs(out_dir, exist_ok=True)
    colors = plt.cm.tab10.colors

    all_param_names = None
    for res in results:
        for key in ["per_param_abs_errors", "per_param_rel_errors"]:
            if key in res:
                all_param_names = list(res[key].keys())
                break
        if all_param_names:
            break

    if all_param_names is None:
        max_dim = max(r["true_params"].shape[1] for r in results)
        all_param_names = [f"param_{i}" for i in range(max_dim)]

    all_param_names = sorted(
        all_param_names,
        key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split("([0-9]+)", s)],
    )

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

    for j in range(i + 1, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    _tight_suptitle(fig, "True vs Estimated Parameters (Calibration Comparison)")
    path = os.path.join(out_dir, "param_true_vs_est.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved scatter comparison to {path}")



def plot_param_error_ecdfs(results, labels, out_dir="calibration_plots", kind="relative"):
    """Compare ECDFs of per-parameter errors and RMSEs across multiple models."""
    os.makedirs(out_dir, exist_ok=True)

    param_sets, rmses_sets = [], []
    for r in results:
        rmses_sets.append(r.get("rmses", []))
        if kind == "absolute":
            param_sets.append(r["per_param_abs_errors"])
        else:
            for key in ["per_param_rel_errors", "per_param_errors"]:
                if key in r:
                    param_sets.append(r[key])
                    break

    first_dict = param_sets[0]
    if not first_dict or isinstance(first_dict, dict) and not hasattr(first_dict, "__iter__"):
        all_param_names = sorted(
            {k for d in param_sets for k in d.keys()},
            key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split("([0-9]+)", s)],
        )
    else:
        all_param_names = sorted(
            list(first_dict.keys()),
            key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split("([0-9]+)", s)],
        )

    ncols = min(4, len(all_param_names))
    nrows = int(np.ceil(len(all_param_names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), squeeze=False)

    def ecdf(x):
        xs = np.sort(np.asarray(x))
        ys = np.linspace(0, 1, len(xs))
        return xs, ys

    for i, param in enumerate(all_param_names):
        ax = axes[i // ncols, i % ncols]
        for data, label in zip(param_sets, labels):
            xs, ys = ecdf(data[param])
            ax.plot(ys, xs * (100 if kind == "relative" else 1), label=label)
        ax.set_title(param, fontsize=14)
        ax.set_xlabel("Quantiles")
        ax.set_ylabel(f"{kind.title()} Error" + (" [%]" if kind == "relative" else ""))
        if i == 0:
            ax.legend(frameon=True, loc="upper left")

    for j in range(i + 1, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    _tight_suptitle(fig, f"Parameter {kind.title()} Error CDFs")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path_params = os.path.join(out_dir, f"param_error_cdfs_{kind}.png")
    plt.savefig(path_params, dpi=200)
    plt.close(fig)
    print(f"Saved {kind} error ECDF comparison to {path_params}")

    # RMSE ECDF
    fig, ax = plt.subplots(figsize=(5, 4))
    for rmses, label in zip(rmses_sets, labels):
        xs, ys = ecdf(rmses)
        ax.plot(ys, xs, label=label)
    ax.set_xlabel("Quantiles")
    ax.set_ylabel("RMSE")
    ax.legend(frameon=True, loc="upper left")
    _tight_suptitle(fig, "Calibration RMSE ECDFs")
    path_rmse = os.path.join(out_dir, "rmse_ecdf.png")
    plt.savefig(path_rmse, dpi=200)
    plt.close(fig)
    print(f"Saved RMSE ECDF comparison to {path_rmse}")


import os, numpy as np, matplotlib.pyplot as plt, copy

def apply_structured_noise(iv_surface, sigma=0.01, mode="both"):
    """Apply structured multiplicative noise (row/col/both) to an IV surface."""
    noisy = iv_surface.copy()
    nT, nK = iv_surface.shape
    if mode in ["row", "both"]:
        row_factors = 1 + np.random.normal(0, sigma, size=nT)
        noisy *= row_factors[:, None]
    if mode in ["col", "both"]:
        col_factors = 1 + np.random.normal(0, sigma, size=nK)
        noisy *= col_factors[None, :]
    return noisy


def compare_model_robustness(
    models,
    model_labels,
    surfaces,
    noise_levels=(0.001, 0.002, 0.005, 0.01, 0.02),
    n_real=10,
    out_dir="robustness_comparison",
    mode="both",
):
    """Compare robustness of multiple models under structured IV perturbations."""
    os.makedirs(out_dir, exist_ok=True)
    eps = 1e-8
    colors = plt.cm.tab10.colors

    n_knots = len(surfaces[0]["params"]["xi0_knots"])
    param_names = ["eta", "rho", "H"] + [f"xi0_{i+1}" for i in range(n_knots)]

    # sort using same rule as ECDFs
    param_names = sorted(
        param_names,
        key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split("([0-9]+)", s)],
    )

    stats = []
    for model, label in zip(models, model_labels):
        mean_err = {p: [] for p in param_names}
        low_q = {p: [] for p in param_names}
        high_q = {p: [] for p in param_names}

        for sigma in noise_levels:
            rel_errs_lvl = {p: [] for p in param_names}
            for s in surfaces:
                base = model.calibrate(s, optimiser="levenberg-marquardt")["theta_hat"]
                for _ in range(n_real):
                    s_noisy = copy.deepcopy(s)
                    if sigma > 0:
                        s_noisy["iv_surface"] = apply_structured_noise(s["iv_surface"], sigma, mode)
                    noisy = model.calibrate(s_noisy, optimiser="levenberg-marquardt")["theta_hat"]
                    for i, p in enumerate(param_names):
                        rel = abs(noisy[i] - base[i]) / (abs(base[i]) + eps)
                        rel_errs_lvl[p].append(rel)

            for p in param_names:
                errs = np.array(rel_errs_lvl[p])
                mean_err[p].append(np.mean(errs))
                low_q[p].append(np.quantile(errs, 0.05))
                high_q[p].append(np.quantile(errs, 0.95))

        stats.append({"label": label, "mean": mean_err, "low": low_q, "high": high_q})

    # Plot layout
    n_params = len(param_names)
    ncols = min(4, n_params)
    nrows = int(np.ceil(n_params / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False)

    for i, pname in enumerate(param_names):
        ax = axes[i // ncols, i % ncols]
        for mi, s in enumerate(stats):
            mean = np.array(s["mean"][pname])
            lo = np.array(s["low"][pname])
            hi = np.array(s["high"][pname])
            c = colors[mi % len(colors)]
            ax.plot(noise_levels, mean, color=c, lw=2, label=s["label"])
            ax.plot(noise_levels, lo, color=c, ls="--", lw=1)
            ax.plot(noise_levels, hi, color=c, ls="--", lw=1)
        #ax.set_xscale("symlog", linthresh=1e-4)
        ax.set_xlabel("Noise Std")
        ax.set_ylabel(f"Rel |Δ{pname}|")
        ax.set_title(pname)
        if i == 0:
            ax.legend(frameon=True, loc="upper left")

    for j in range(i + 1, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    # X-Ticks styling
    for ax in axes.flat:
        ax.set_xticks(noise_levels)
        ax.set_xticklabels([f"{x:.3g}" for x in noise_levels], rotation=45, ha="right")

    _tight_suptitle(fig, "Model Robustness Comparison under IV Perturbations")
    path = os.path.join(out_dir, "robustness_comparison.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved comparison plot to {path}")

    return stats
