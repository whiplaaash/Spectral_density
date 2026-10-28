

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple


# ============================================================
# 1. Utilities
# ============================================================

def make_frequency_grid(
    wmin=1e-4,
    wsplit=1.0,
    wmax=40.0,
    nlog=200,
    nlin=400
):
    """
    Build a mixed log-linear frequency grid in units of T:
        x = omega / T

    Returns
    -------
    w : np.ndarray
        Frequency grid
    """
    w_log = np.logspace(np.log10(wmin), np.log10(wsplit), nlog, endpoint=False)
    w_lin = np.linspace(wsplit, wmax, nlin)
    w = np.unique(np.concatenate([w_log, w_lin]))
    return w


def smooth_switch(w, Lambda=3.0, sharpness=4.0):
    """
    Smoothly turn on the UV contribution around Lambda.

    f(w) ~ 0 for w << Lambda
    f(w) ~ 1 for w >> Lambda
    """
    x = (w / Lambda) ** sharpness
    return x / (1.0 + x)


def gaussian_bump(w, amp, mu, sigma, power=1):
    """
    A flexible bump term:
        amp * w^power * exp(-(w-mu)^2 / (2 sigma^2))
    """
    return amp * (w ** power) * np.exp(-0.5 * ((w - mu) / sigma) ** 2)


def lorentzian_transport_peak(w, T, A, Gamma):
    """
    IR transport peak:
        rho_IR(w) = A * w / (1 + (w/Gamma)^2)

    Low-frequency slope:
        rho(w)/w -> A as w -> 0
    So viscosity is proportional to A.
    """
    return A * w * T * (Gamma**2)/(Gamma**2 + w**2)

# ============================================================
# 2. Parameter containers
# ============================================================

@dataclass
class ShearParams:
    A: float               # IR slope coefficient
    Gamma: float           # transport width
    B: float               # UV amplitude
    Lambda_uv: float       # UV onset scale
    sharpness_uv: float    # UV switch sharpness
    n_bumps: int = 0       # number of mid-frequency bumps
    bump_amps: Optional[np.ndarray] = None
    bump_mus: Optional[np.ndarray] = None
    bump_sigmas: Optional[np.ndarray] = None
    T: float = 1.0


@dataclass
class BulkParams:
    A: float
    Gamma: float
    D: float
    Lambda_uv: float
    sharpness_uv: float
    n_bumps: int = 0
    bump_amps: Optional[np.ndarray] = None
    bump_mus: Optional[np.ndarray] = None
    bump_sigmas: Optional[np.ndarray] = None
    Tc: float = 1.0
    T: float = 1.0


# ============================================================
# 3. Spectral function models
# ============================================================

def rho_shear(w, params: ShearParams):
    """
    Mock shear spectral function in dimensionless units x = omega/T.

    Structure:
        rho = IR transport + UV tail + optional mid-frequency bumps

    Here we use a simple UV tail ~ w^4 * switch(w).
    """
    # IR transport
    rho_ir = lorentzian_transport_peak(w, params.T, params.A, params.Gamma)

    # UV tail
    uv_switch = smooth_switch(w, Lambda=params.Lambda_uv, sharpness=params.sharpness_uv)
    rho_uv = params.B * ((w * params.T) ** 4) * uv_switch

    # Mid-frequency structure
    rho_mid = np.zeros_like(w)
    if params.n_bumps > 0:
        for amp, mu, sigma in zip(params.bump_amps, params.bump_mus, params.bump_sigmas):
            rho_mid += gaussian_bump(w, amp=amp, mu=mu, sigma=sigma, power=3)

    #rho = rho_ir + rho_mid + rho_uv
    rho = rho_ir + rho_uv

    # Safety
    rho = np.maximum(rho, 0.0)
    return rho


def bulk_peak_enhancement(T, Tc=1.0, height=1.0, width=0.12):
    """
    Temperature-dependent enhancement factor for bulk near Tc.
    """
    return 1.0 + height * np.exp(-0.5 * ((T - Tc) / width) ** 2)



def rho_bulk(w, params: BulkParams, peak_height=2.0, peak_width=0.12):
    """
    Mock bulk spectral function.

    Similar to shear, but the IR coefficient A is enhanced near T ~ Tc.
    """
    enh = bulk_peak_enhancement(
        T=params.T, Tc=params.Tc, height=peak_height, width=peak_width
    )

    A_eff = params.A * enh

    rho_ir = lorentzian_transport_peak(w, params.T, A_eff, params.Gamma)

    uv_switch = smooth_switch(w, Lambda=params.Lambda_uv, sharpness=params.sharpness_uv)
    rho_uv = params.D * (w ** 4) * uv_switch

    rho_mid = np.zeros_like(w)
    if params.n_bumps > 0:
        for amp, mu, sigma in zip(params.bump_amps, params.bump_mus, params.bump_sigmas):
            # bulk bumps can be softer than shear
            rho_mid += gaussian_bump(w, amp=amp, mu=mu, sigma=sigma, power=1)

    rho = rho_ir + rho_mid + rho_uv
    rho = np.maximum(rho, 0.0)
    return rho
