"""CPMG forward model for NV-13C hyperfine detection.

Port of Eq. (1)-(3) of Jung et al., npj Quantum Information 7, 41 (2021)
(refer/Deep_Learning_CPMG_Analysis) to a modern numpy stack.

Conventions
-----------
- A, B : hyperfine parallel / perpendicular components, in Hz (linear freq).
- tau  : half of the inter-pi-pulse delay, in seconds (paper convention).
- B-field in Gauss. All internal angular frequencies in rad/s.
"""

from __future__ import annotations

import numpy as np

# 13C gyromagnetic ratio, gamma/2pi in Hz/G (same value as reference repo:
# GYRO_MAGNETIC_RATIO = 1.0705 kHz/G)
GAMMA_13C_HZ_PER_G = 1070.5

# Experimental condition for this project (NV_CPMG.pptx)
B_FIELD_G = 440.1


def larmor_omega(b_field_g: float = B_FIELD_G) -> float:
    """Nuclear Larmor angular frequency (rad/s)."""
    return 2.0 * np.pi * GAMMA_13C_HZ_PER_G * b_field_g


def cpmg_M(
    tau: np.ndarray,
    ab_list: np.ndarray,
    n_pulse: int,
    b_field_g: float = B_FIELD_G,
) -> np.ndarray:
    """Coherence factor M(tau) = prod_k M_k(tau) for a set of 13C spins.

    Parameters
    ----------
    tau : (T,) times in seconds (half inter-pulse delay).
    ab_list : (K, 2) rows of (A, B) in Hz.
    n_pulse : number of pi pulses N.

    Returns
    -------
    (T,) array of M values in [-1, 1]. Px = (1 + M) / 2.
    """
    ab = np.atleast_2d(np.asarray(ab_list, dtype=np.float64))
    if ab.size == 0:
        return np.ones_like(tau)
    wL = larmor_omega(b_field_g)
    A = 2.0 * np.pi * ab[:, 0:1]  # (K,1) rad/s
    B = 2.0 * np.pi * ab[:, 1:2]

    w_tilde = np.sqrt((A + wL) ** 2 + B**2)  # (K,1)
    mz = (A + wL) / w_tilde
    mx = B / w_tilde

    t = np.asarray(tau, dtype=np.float64).reshape(1, -1)  # (1,T)
    alpha = w_tilde * t  # (K,T)
    beta = wL * t  # (1,T)

    cos_phi = np.cos(alpha) * np.cos(beta) - mz * np.sin(alpha) * np.sin(beta)
    cos_phi = np.clip(cos_phi, -1.0, 1.0)
    phi = np.arccos(cos_phi)

    K1 = (1.0 - np.cos(alpha)) * (1.0 - np.cos(beta))
    K2 = 1.0 + cos_phi
    with np.errstate(divide="ignore", invalid="ignore"):
        K = mx**2 * (K1 / K2)
    K = np.nan_to_num(K, nan=0.0, posinf=0.0)

    M_k = 1.0 - K * np.sin(n_pulse * phi / 2.0) ** 2
    return np.prod(M_k, axis=0)


def target_period(a_hz: float, b_hz: float, b_field_g: float = B_FIELD_G) -> float:
    """Local dip period TP = 2*pi / (w_tilde + wL) in seconds (paper Eq. 4)."""
    wL = larmor_omega(b_field_g)
    w_tilde = np.sqrt((2 * np.pi * a_hz + wL) ** 2 + (2 * np.pi * b_hz) ** 2)
    return 2.0 * np.pi / (w_tilde + wL)


def stretched_exp(tau: np.ndarray, T2: float, n: float) -> np.ndarray:
    """Decoherence envelope exp(-(tau/T2)^n) (paper Eq. 5)."""
    return np.exp(-((np.asarray(tau) / T2) ** n))
