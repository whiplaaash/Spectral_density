#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 23:04:49 2026

@author: czajkaa
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple
import spectra as sd


# ============================================================
# 1. Euclidean time grid
# ============================================================

def make_tau_grid(Ntau: int, exclude_endpoints: bool = True):
    """
    Construct dimensionless Euclidean time grid:
        u = tau * T in (0, 1)

    Parameters
    ----------
    Ntau : int
        Number of lattice sites in Euclidean time.
    exclude_endpoints : bool
        If True, omit u=0 and u=1 to avoid contact-term issues.

    Returns
    -------
    u : np.ndarray
        Euclidean time points in units of tau*T
    """
    if exclude_endpoints:
        return np.arange(1, Ntau) / Ntau
    else:
        return np.arange(0, Ntau + 1) / Ntau


# ============================================================
# 2. Thermal kernel
# ============================================================

def thermal_kernel(u: np.ndarray, w: np.ndarray):
    """
    Stable evaluation of the Euclidean thermal kernel:
        K(u,w) = cosh[w(u-1/2)] / sinh(w/2)

    We use the numerically stable equivalent form:
        K(u,w) = (exp(-w*u) + exp(-w*(1-u))) / (1 - exp(-w))

    Parameters
    ----------
    u : np.ndarray, shape (Nu,)
        Euclidean times in units of tau*T
    w : np.ndarray, shape (Nw,)
        Frequency grid in units of omega/T

    Returns
    -------
    K : np.ndarray, shape (Nu, Nw)
        Kernel matrix
    """
    u = np.atleast_1d(u)
    w = np.atleast_1d(w)

    U = u[:, None]
    W = w[None, :]

    # Avoid division by zero at w ~ 0 using series expansion
    small = W < 1e-8
    K = np.empty((len(u), len(w)), dtype=np.float64)

    # Stable generic formula
    denom = 1.0 - np.exp(-W)
    K_generic = (np.exp(-W * U) + np.exp(-W * (1.0 - U))) / denom

    # Small-w expansion:
    # K(u,w) ~ 2/w + w * [(u-1/2)^2 - 1/12] + ...
    # The leading term is enough for numerical stability at tiny w.
    K_small = 2.0 / np.maximum(W, 1e-300)

    K[:] = np.where(small, K_small, K_generic)
    return K


# ============================================================
# 3. Numerical integration weights
# ============================================================

def trapezoidal_weights(x: np.ndarray):
    """
    Trapezoidal-rule weights for nonuniform grid.
    """
    x = np.asarray(x)
    dx = np.diff(x)

    w = np.zeros_like(x)
    w[1:-1] = 0.5 * (dx[:-1] + dx[1:])
    w[0] = 0.5 * dx[0]
    w[-1] = 0.5 * dx[-1]
    return w


# ============================================================
# 4. Correlator generation
# ============================================================

def euclidean_correlator(
    w: np.ndarray,
    rho: np.ndarray,
    u: np.ndarray,
    prefactor: float = 1.0 / (2.0 * np.pi)
):
    """
    Compute the Euclidean correlator:
        G_E(u) = ∫_0^∞ dω/(2π) rho(ω) K(u,ω)

    On the dimensionless grid x = omega/T:
        G(u) = prefactor * ∫ dx rho(x) K(u,x)

    Parameters
    ----------
    w : np.ndarray
        Frequency grid (dimensionless omega/T)
    rho : np.ndarray
        Spectral function on the same grid
    u : np.ndarray
        Euclidean time grid (dimensionless tau*T)
    prefactor : float
        Usually 1/(2π), depending on your conventions

    Returns
    -------
    G : np.ndarray
        Euclidean correlator values on u-grid
    """
    w = np.asarray(w)
    rho = np.asarray(rho)
    u = np.asarray(u)

    weights = trapezoidal_weights(w)
    K = thermal_kernel(u, w)

    # matrix-vector multiply for the integral
    G = prefactor * (K @ (rho * weights))
    return G


def build_kernel_matrix(
    u: np.ndarray,
    w: np.ndarray,
    prefactor: float = 1.0 / (2.0 * np.pi)
):
    """
    Precompute matrix M such that:
        G = M @ rho
    including integration weights.

    Useful for ML pipelines.
    """
    weights = trapezoidal_weights(w)
    K = thermal_kernel(u, w)
    M = prefactor * K * weights[None, :]
    return M


def euclidean_correlator_from_matrix(M: np.ndarray, rho: np.ndarray):
    """
    Fast correlator evaluation from precomputed kernel matrix.
    """
    return M @ rho


# ============================================================
# 5. Noise models
# ============================================================

@dataclass
class NoiseModel:
    rel_base: float = 0.002
    rel_mid: float = 0.01
    mid_width: float = 0.18
    corr_length: float = 2.0
    correlated: bool = True


def build_relative_error_profile(u: np.ndarray, noise_model: NoiseModel):
    """
    Relative error profile, usually somewhat larger near u=1/2.
    """
    return (
        noise_model.rel_base
        + noise_model.rel_mid * np.exp(-0.5 * ((u - 0.5) / noise_model.mid_width) ** 2)
    )


def build_covariance_matrix(
    G: np.ndarray,
    u: np.ndarray,
    noise_model: NoiseModel
):
    """
    Build a simple covariance matrix:
        C_ij = sigma_i sigma_j exp(-|i-j|/ell)
    """
    rel_err = build_relative_error_profile(u, noise_model)
    sigma = rel_err * np.maximum(np.abs(G), 1e-14)

    if not noise_model.correlated:
        return np.diag(sigma**2)

    idx = np.arange(len(u))
    dist = np.abs(idx[:, None] - idx[None, :])
    corr = np.exp(-dist / noise_model.corr_length)

    C = corr * np.outer(sigma, sigma)
    return C


def add_gaussian_noise(
    G: np.ndarray,
    C: np.ndarray,
    rng: Optional[np.random.Generator] = None
):
    """
    Sample noisy correlator:
        G_noisy ~ N(G, C)
    """
    if rng is None:
        rng = np.random.default_rng()

    # small jitter for numerical stability
    jitter = 1e-14 * np.eye(len(G))
    L = np.linalg.cholesky(C + jitter)
    z = rng.normal(size=len(G))
    G_noisy = G + L @ z
    return G_noisy


# ============================================================
# 6. Diagnostic plotting
# ============================================================

def plot_correlator(u, G, G_noisy=None, label=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(u, G, '-o', lw=2, ms=4, label=f'{label} (clean)' if label else 'clean')

    if G_noisy is not None:
        ax.plot(u, G_noisy, 's', ms=4, alpha=0.7, label=f'{label} (noisy)' if label else 'noisy')

    ax.set_xlabel(r'$\tau T$')
    ax.set_ylabel(r'$G_E(\tau)$')
    ax.set_title('Euclidean correlator')
    ax.legend()
    return ax


# ============================================================
# 7. Example usage
# ============================================================

if __name__ == "__main__":
    # --------------------------------------------------------
    # Example spectral functions (simple placeholders)
    # Replace these with your actual rho_shear / rho_bulk
    # from the spectral-function module.
    # --------------------------------------------------------
    # def make_frequency_grid(
    #     wmin=1e-4, wsplit=1.0, wmax=40.0, nlog=200, nlin=400
    # ):
    #     w_log = np.logspace(np.log10(wmin), np.log10(wsplit), nlog, endpoint=False)
    #     w_lin = np.linspace(wsplit, wmax, nlin)
    #     return np.unique(np.concatenate([w_log, w_lin]))

    # def smooth_switch(w, Lambda=3.0, sharpness=4.0):
    #     x = (w / Lambda) ** sharpness
    #     return x / (1.0 + x)

    # def lorentzian_transport_peak(w, A, Gamma):
    #     return A * w / (1.0 + (w / Gamma) ** 2)

    def mock_shear_rho(w, A=0.12, Gamma=1.2, B=8e-4, Lambda=3.5):
        return sd.lorentzian_transport_peak(w, A, Gamma) + B * (w ** 4) * sd.smooth_switch(w, Lambda)

    # --------------------------------------------------------
    # Build grids
    # --------------------------------------------------------
    w = sd.make_frequency_grid()
    u = make_tau_grid(Ntau=16, exclude_endpoints=True)

    # --------------------------------------------------------
    # Make correlator
    # --------------------------------------------------------
    rho = mock_shear_rho(w)
    G = euclidean_correlator(w, rho, u)

    # --------------------------------------------------------
    # Add noise
    # --------------------------------------------------------
    noise_model = NoiseModel(
        rel_base=0.002,
        rel_mid=0.01,
        mid_width=0.18,
        corr_length=2.0,
        correlated=True
    )

    C = build_covariance_matrix(G, u, noise_model)
    rng = np.random.default_rng(12345)
    G_noisy = add_gaussian_noise(G, C, rng=rng)

    # --------------------------------------------------------
    # Plot correlator
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_correlator(u, G, G_noisy=G_noisy, label='shear example', ax=ax)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # Plot covariance matrix
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(C, origin='lower', cmap='viridis')
    ax.set_title('Covariance matrix')
    ax.set_xlabel('i')
    ax.set_ylabel('j')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()