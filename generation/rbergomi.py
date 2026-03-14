# ====
# Utility functions for Rough Bergomi model and IV surface operations from https://github.com/ryanmccrickerd/rough_bergomi


import numpy as np
from .utils import *
import numpy as np
from numpy.fft import rfft, irfft

class rBergomi(object):
    """
    Class for generating paths of the rBergomi model.
    """
    def __init__(self, n = 100, N = 1000, T = 1.00, a = -0.4):
        """
        Constructor for class.
        """
        # Basic assignments
        self.T = T # Maturity
        self.n = n # Granularity (steps per year)
        self.dt = 1.0/self.n # Step size
        self.s = int(self.n * self.T) # Steps
        self.t = np.linspace(0, self.T, 1 + self.s)[np.newaxis,:] # Time grid
        self.a = a # Alpha
        self.N = N # Paths

        # Construct hybrid scheme correlation structure for kappa = 1
        self.e = np.array([0,0])
        self.c = cov(self.a, self.n)

    def dW1(self):
        """
        Produces random numbers for variance process with required
        covariance structure.
        """
        rng = np.random.multivariate_normal
        return rng(self.e, self.c, (self.N, self.s))

    def Y(self, dW):
        """
        Constructs Volterra process from appropriately correlated
        2D Brownian increments (FFT-accelerated version).
        Equivalent to the original np.convolve() approach,
        but O(N * s log s) instead of O(N * s^2).
        """


        # --- shapes & constants ---
        N = self.N          # number of Monte Carlo paths
        s = self.s          # number of time steps (minus 1)
        a = self.a
        n = self.n

        # --- Y1 term (exact integral) ---
        Y1 = np.zeros((N, 1 + s))
        Y1[:, 1:] = dW[:, :, 1]

        # --- build Gamma kernel G (same as before) ---
        G = np.zeros(1 + s)
        for k in range(2, 1 + s):
            G[k] = g(b(k, a) / n, a)

        # --- extract Xi process (first Brownian component) ---
        X = dW[:, :, 0]     # shape (N, s)

        # --- FFT-based convolution for all paths simultaneously ---
        L = len(G) + X.shape[1] - 1   # full linear conv length
        G_fft = rfft(G, n=L)
        X_fft = rfft(X, n=L, axis=1)
        GX = irfft(G_fft[None, :] * X_fft, n=L, axis=1)
        Y2 = GX[:, :1 + s]             # truncate to same shape as before

        # --- final combination ---
        Y = np.sqrt(2 * a + 1) * (Y1 + Y2)
        return Y


    def dW2(self):
        """
        Obtain orthogonal increments.
        """
        return np.random.randn(self.N, self.s) * np.sqrt(self.dt)

    def dB(self, dW1, dW2, rho = 0.0):
        """
        Constructs correlated price Brownian increments, dB.
        """
        self.rho = rho
        dB = rho * dW1[:,:,0] + np.sqrt(1 - rho**2) * dW2
        return dB

    def V(self, Y, xi = 1.0, eta = 1.0):
        """
        rBergomi variance process.
        """
        self.xi = xi
        self.eta = eta
        a = self.a
        t = self.t
        V = xi * np.exp(eta * Y - 0.5 * eta**2 * t**(2 * a + 1))
        return V

    def S(self, V, dB, S0=1.0):
        """
        rBergomi price process (optimized for memory locality).
        """
        self.S0 = S0
        dt = self.dt
        rho = self.rho

        # Compute increments
        sqrtV = np.sqrt(V[:, :-1], dtype=V.dtype)
        increments = sqrtV * dB - 0.5 * V[:, :-1] * dt

        # Cumulative sum along time
        np.cumsum(increments, axis=1, out=increments)

        # Build price paths
        S = np.empty_like(V)
        S[:, 0] = S0
        S[:, 1:] = S0 * np.exp(increments)
        return S

    def S1(self, V, dW1, rho, S0=1.0):
        """
        rBergomi parallel price process (optimized for memory locality).
        """
        dt = self.dt
        sqrtV = np.sqrt(V[:, :-1], dtype=V.dtype)
        increments = rho * sqrtV * dW1[:, :, 0] - 0.5 * rho**2 * V[:, :-1] * dt
        np.cumsum(increments, axis=1, out=increments)
        S = np.empty_like(V)
        S[:, 0] = S0
        S[:, 1:] = S0 * np.exp(increments)
        return S
