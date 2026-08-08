"""Sequential-greedy differential-evolution spin fitting.

Reimplementation of the old-cdetect approach (refer/old/cdetect
PROGRESS.md sections 12/18.4) with this project's forward model:

  - fit spins one at a time: at step k, a 2-parameter (A, B) DE search for
    the new spin while previously found spins stay fixed,
  - after each addition, a local L-BFGS polish of ALL spin parameters,
  - model selection by BIC over the number of spins.

Operates on envelope-normalized M(tau) jointly over all CPMG channels.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .physics import cpmg_M


def _model_m(tau, ab_flat, n_pulses, b_field_g):
    ab = np.asarray(ab_flat, dtype=np.float64).reshape(-1, 2)
    return np.array([cpmg_M(tau, ab, n, b_field_g) for n in n_pulses])


def _rss(tau, m_data, ab_flat, n_pulses, b_field_g):
    pred = _model_m(tau, ab_flat, n_pulses, b_field_g)
    return float(np.sum((pred - m_data) ** 2))


def greedy_de_fit(
    tau: np.ndarray,
    m_data: np.ndarray,
    n_pulses: tuple,
    b_field_g: float,
    max_spins: int = 8,
    a_bounds=(-60e3, 60e3),
    b_bounds=(8e3, 60e3),
    de_maxiter: int = 80,
    de_popsize: int = 30,
    seed: int = 0,
    verbose: bool = True,
):
    """Returns dict with the BIC-optimal spin list and the full trace.

    m_data : (C, T) envelope-normalized M for each CPMG channel.
    """
    n_obs = m_data.size
    spins: list = []
    trace = []

    rss0 = float(np.sum((m_data - 1.0) ** 2))  # 0-spin model: M = 1
    bic0 = n_obs * np.log(rss0 / n_obs)
    trace.append(dict(k=0, rss=rss0, bic=bic0, spins=[]))
    best = dict(k=0, bic=bic0, spins=[], rss=rss0)

    for k in range(1, max_spins + 1):
        fixed = np.array(spins, dtype=np.float64).reshape(-1, 2)

        def objective(x):
            ab = np.vstack([fixed, np.atleast_2d(x)])
            return _rss(tau, m_data, ab, n_pulses, b_field_g)

        result = differential_evolution(
            objective,
            bounds=[a_bounds, b_bounds],
            maxiter=de_maxiter,
            popsize=de_popsize,
            seed=seed + k,
            polish=True,
        )
        spins.append(list(result.x))

        # local polish of all parameters together
        x0 = np.array(spins, dtype=np.float64).ravel()
        lb = [a_bounds[0], b_bounds[0]] * len(spins)
        ub = [a_bounds[1], b_bounds[1]] * len(spins)
        res2 = minimize(
            lambda x: _rss(tau, m_data, x, n_pulses, b_field_g),
            x0,
            method="L-BFGS-B",
            bounds=list(zip(lb, ub)),
        )
        spins = res2.x.reshape(-1, 2).tolist()

        rss = float(res2.fun)
        n_par = 2 * len(spins)
        bic = n_obs * np.log(rss / n_obs) + n_par * np.log(n_obs)
        trace.append(dict(k=k, rss=rss, bic=bic,
                          spins=[list(s) for s in spins]))
        if verbose:
            print(f"  DE k={k}: rss={rss:.2f} bic={bic:.1f} "
                  f"new=({result.x[0]/1e3:+.1f},{result.x[1]/1e3:.1f}) kHz", flush=True)
        if bic < best["bic"]:
            best = dict(k=k, bic=bic, spins=[list(s) for s in spins], rss=rss)

    return dict(best=best, trace=trace)
