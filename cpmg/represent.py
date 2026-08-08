"""Slice-stack 2D representation of CPMG signals (Jung et al. Fig. 2b).

The 1D signal P(tau) is cut into consecutive segments of length TP (a
candidate local period) and stacked vertically. A nuclear spin whose local
period matches TP produces a near-vertical line of coherence dips at the
same in-period position across all slices.
"""

from __future__ import annotations

import numpy as np

from .physics import larmor_omega, stretched_exp


def slice_stack(
    tau: np.ndarray,
    signal: np.ndarray,
    period: float,
    width: int = 53,
    tau_start: float = 0.0,
) -> np.ndarray:
    """Build the 2D slice-stack image.

    Parameters
    ----------
    tau : (T,) times in seconds, roughly uniform.
    signal : (T,) values to stack (raw Px or envelope-normalized M).
    period : candidate local period TP in seconds.
    width : pixels per period (columns). Rows are interpolated from data.
    tau_start : where to start slicing.

    Returns
    -------
    (n_slices, width) image. Cells outside the data range are NaN.
    """
    tau = np.asarray(tau, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    tau_end = tau[-1]
    n_slices = int(np.floor((tau_end - tau_start) / period))
    if n_slices < 1:
        raise ValueError("period longer than the data range")

    cols = (np.arange(width) + 0.5) / width  # in-period fractional position
    rows = np.arange(n_slices)
    grid = tau_start + (rows[:, None] + cols[None, :]) * period  # (S,W)

    img = np.interp(grid.ravel(), tau, signal, left=np.nan, right=np.nan)
    return img.reshape(n_slices, width)


def envelope_normalize(
    tau: np.ndarray,
    px: np.ndarray,
    fit_quantile: float = 0.9,
    n_windows: int = 20,
):
    """Recover M(tau) from raw Px using Eq. (5): Px = (1 + M * env) / 2.

    The stretched-exponential envelope exp(-(tau/T2)^n) is fitted to the
    upper quantile of |2*Px - 1| in time windows (the dips only ever reduce
    coherence, so the upper envelope tracks the decoherence decay).

    Returns
    -------
    m_rec : (T,) envelope-normalized M estimate, clipped to [-1.5, 1.5]
    (T2, n) : fitted envelope parameters (seconds, unitless)
    """
    from scipy.optimize import curve_fit

    tau = np.asarray(tau, dtype=np.float64)
    m_raw = 2.0 * np.asarray(px, dtype=np.float64) - 1.0

    edges = np.linspace(tau[0], tau[-1], n_windows + 1)
    t_env, v_env = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (tau >= lo) & (tau < hi)
        if mask.sum() < 5:
            continue
        t_env.append(0.5 * (lo + hi))
        v_env.append(np.quantile(np.abs(m_raw[mask]), fit_quantile))
    t_env, v_env = np.array(t_env), np.array(v_env)

    def model(t, amp, T2, n):
        return amp * stretched_exp(t, T2, n)

    p0 = (max(v_env.max(), 0.1), tau[-1], 1.0)
    bounds = ([0.05, tau[1], 0.3], [1.5, tau[-1] * 100, 3.0])
    (amp, T2, n), _ = curve_fit(model, t_env, v_env, p0=p0, bounds=bounds, maxfev=20000)

    env = amp * stretched_exp(tau, T2, n)
    m_rec = np.clip(m_raw / np.maximum(env, 1e-3), -1.5, 1.5)
    return m_rec, (T2, n)


def period_scan_contrast(
    tau: np.ndarray,
    m_rec: np.ndarray,
    a_grid_hz: np.ndarray,
    b_hz: float,
    b_field_g: float,
    width: int = 53,
) -> np.ndarray:
    """Poor-man's HPC: vertical-line contrast versus candidate A.

    For each candidate A (with a representative B), slice-stack the signal at
    the corresponding TP and measure how deep the column-averaged dip is.
    A real spin at that period aligns dips vertically -> deep column mean;
    a period mismatch smears dips across columns -> shallow column mean.

    Returns (len(a_grid),) contrast values (bigger = more spin-like).
    """
    from .physics import target_period

    out = np.empty(len(a_grid_hz))
    for i, a in enumerate(a_grid_hz):
        tp = target_period(a, b_hz, b_field_g)
        img = slice_stack(tau, m_rec, tp, width=width)
        col_mean = np.nanmean(img, axis=0)
        out[i] = np.nanmean(col_mean) - np.nanmin(col_mean)
    return out
