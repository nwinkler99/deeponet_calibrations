import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def g(x, a):
    """
    TBSS kernel applicable to the rBergomi variance process.
    """
    return x**a

def b(k, a):
    """
    Optimal discretisation of TBSS process for minimising hybrid scheme error.
    """
    return ((k**(a+1)-(k-1)**(a+1))/(a+1))**(1/a)

def cov(a, n):
    """
    Covariance matrix for given alpha and n, assuming kappa = 1 for
    tractability.
    """
    cov = np.array([[0.,0.],[0.,0.]])
    cov[0,0] = 1./n
    cov[0,1] = 1./((1.*a+1) * n**(1.*a+1))
    cov[1,1] = 1./((2.*a+1) * n**(2.*a+1))
    cov[1,0] = cov[0,1]
    return cov

def bs(F, K, V, o = 'call'):
    """
    Returns the Black call price for given forward, strike and integrated
    variance.
    """
    # Set appropriate weight for option token o
    w = 1
    if o == 'put':
        w = -1
    elif o == 'otm':
        w = 2 * (K > 1.0) - 1

    sv = np.sqrt(V)
    d1 = np.log(F/K) / sv + 0.5 * sv
    d2 = d1 - sv
    P = w * F * norm.cdf(w * d1) - w * K * norm.cdf(w * d2)
    return P

def bsinv(P, F, K, t, o='call'):
    """
    Robust implied Black volatility from price P, forward F, strike K, maturity t.
    Handles call/put/otm options; safe for MC-generated prices.
    """
    if t <= 1e-10:
        return 1e-8  # degenerate maturity

    w = 1.0
    if o == 'put':
        w = -1.0
    elif o == 'otm':
        w = 2.0 * (K > F) - 1.0  # more consistent OTM switch

    intrinsic = max(w * (F - K), 0.0)
    P = max(P, intrinsic + 1e-12)

    def error(sigma):
        return bs(F, K, sigma**2 * t, o) - P

    try:
        return brentq(error, 1e-8, 5.0, xtol=1e-10, maxiter=100)
    except ValueError:
        # Root not bracketed (e.g. price outside BS range)
        return np.nan


import numpy as np
import matplotlib.pyplot as plt

def plot_iv_surface(
    iv_surface: np.ndarray,
    strikes: np.ndarray,
    maturities: np.ndarray,
    xi0_knots: np.ndarray | None = None,        # length K
    xi0_bin_edges: np.ndarray | None = None,    # length K+1  (e.g., [0, 0.1, 0.2, 0.4, ... , T_max])
    kind: str = "contour",
    cmap: str = "plasma",
    figsize=(10, 6),
    title: str = "Implied Volatility Surface",
    log_maturity: bool = True,
):
    maturities = np.asarray(maturities, float)
    strikes = np.asarray(strikes, float)
    Kgrid, Tgrid = np.meshgrid(strikes, maturities)

    fig, ax = plt.subplots(figsize=figsize)
    if kind == "heatmap":
        im = ax.imshow(
            iv_surface,
            extent=[strikes.min(), strikes.max(), maturities.min(), maturities.max()],
            origin="lower", aspect="auto", cmap=cmap,
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

    # ---------- ξ0 overlay as TRUE piecewise-constant steps ----------
    if xi0_knots is not None and xi0_bin_edges is not None:
        xi0_knots = np.asarray(xi0_knots, float)         # length K
        edges = np.asarray(xi0_bin_edges, float)         # length K+1

        assert len(edges) == len(xi0_knots) + 1, "xi0_bin_edges must have length K+1."

        # Skip the first bin [0, maturities[0]) → not visible on plot
        first_vis_i = np.searchsorted(edges, maturities[0], side="left")
        first_vis_i = max(1, min(first_vis_i, len(xi0_knots)-1))

        # Build explicit step polyline: for each i, draw (edge[i], edge[i+1]) at value xi0_knots[i]
        T_step = []
        X_step = []
        for i in range(first_vis_i, len(xi0_knots)):     # i = 1..K-1 visible
            t0, t1 = edges[i], edges[i+1]
            v = xi0_knots[i]
            T_step += [t0, t1]
            X_step += [v,  v]

        T_step = np.asarray(T_step)
        X_step = np.asarray(X_step)

        # Normalize overlay to live near strike ≈ 1
        xnorm = (X_step - X_step.min()) / (X_step.max() - X_step.min() + 1e-12)
        xnorm = xnorm * (strikes.max() - strikes.min()) * 0.2
        xplot = 1.0 + xnorm

        ax.plot(xplot, T_step, color="white", lw=2.0, label=r"$\xi_0(t)$ (forward variance)")
        ax.legend(loc="upper right", frameon=False)

    plt.tight_layout()
    plt.show()

