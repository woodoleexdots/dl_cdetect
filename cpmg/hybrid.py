"""PF->DE hybrid spin detection.

Stage 1 (PeriodFormer): the P(spin) curve over the A grid is thresholded
into candidate REGIONS (contiguous intervals + margin). PF's zero-false-
positive behaviour makes these regions a trustworthy restriction of the
search space, and its mismatch robustness carries over.

Stage 2 (region-constrained greedy DE): spins are added one at a time; at
each step a small 2-parameter DE runs inside EVERY candidate region, the
best new spin (largest RSS drop) wins, then all parameters are polished
jointly (each spin's A stays inside its region). BIC selects the count.
A region can host multiple spins -> the cluster-enumeration ability that
plain PF peak-picking lacks.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .defit import _rss


def pf_candidate_regions(p_curve, centers_hz, thresh=0.35, margin_hz=3e3):
    """Threshold the PF curve into merged candidate intervals [(lo, hi), ...]."""
    mask = np.asarray(p_curve) >= thresh
    regions = []
    i = 0
    K = len(mask)
    while i < K:
        if mask[i]:
            j = i
            while j + 1 < K and mask[j + 1]:
                j += 1
            regions.append([centers_hz[i] - margin_hz, centers_hz[j] + margin_hz])
            i = j + 1
        else:
            i += 1
    # merge overlaps
    merged = []
    for lo, hi in regions:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged]


def hybrid_greedy_fit(
    tau,
    m_data,
    n_pulses,
    b_field_g,
    regions,
    max_spins=15,
    b_bounds=(3e3, 65e3),
    de_maxiter=40,
    de_popsize=15,
    seed=0,
    patience=3,
    verbose=False,
):
    """Region-constrained greedy DE + BIC. Returns dict like greedy_de_fit,
    with spins as [A, B] and their regions recorded."""
    if not regions:
        return dict(best=dict(k=0, bic=np.inf, spins=[], rss=np.inf), trace=[])
    n_obs = m_data.size
    spins: list = []
    spin_regions: list = []
    rss0 = float(np.sum((m_data - 1.0) ** 2))
    bic0 = n_obs * np.log(rss0 / n_obs)
    trace = [dict(k=0, rss=rss0, bic=bic0)]
    best = dict(k=0, bic=bic0, spins=[], rss=rss0)
    stall = 0

    for k in range(1, max_spins + 1):
        fixed = np.array(spins, dtype=np.float64).reshape(-1, 2)

        best_step = None
        for r_idx, (lo, hi) in enumerate(regions):
            def objective(x, _fixed=fixed):
                ab = np.vstack([_fixed, np.atleast_2d(x)])
                return _rss(tau, m_data, ab, n_pulses, b_field_g)

            res = differential_evolution(
                objective, bounds=[(lo, hi), b_bounds],
                maxiter=de_maxiter, popsize=de_popsize,
                seed=seed + 97 * k + r_idx, polish=True)
            if best_step is None or res.fun < best_step[0]:
                best_step = (res.fun, list(res.x), r_idx)

        _, new_spin, r_idx = best_step
        spins.append(new_spin)
        spin_regions.append(r_idx)

        # joint polish: each spin's A constrained to its region
        x0 = np.array(spins, dtype=np.float64).ravel()
        lb, ub = [], []
        for s_idx in range(len(spins)):
            lo, hi = regions[spin_regions[s_idx]]
            lb += [lo, b_bounds[0]]
            ub += [hi, b_bounds[1]]
        res2 = minimize(lambda x: _rss(tau, m_data, x, n_pulses, b_field_g),
                        x0, method="L-BFGS-B", bounds=list(zip(lb, ub)))
        spins = res2.x.reshape(-1, 2).tolist()

        rss = float(res2.fun)
        bic = n_obs * np.log(rss / n_obs) + 2 * len(spins) * np.log(n_obs)
        trace.append(dict(k=k, rss=rss, bic=bic))
        if verbose:
            print(f"  hybrid k={k}: rss={rss:.2f} bic={bic:.1f} "
                  f"new=({new_spin[0]/1e3:+.1f},{new_spin[1]/1e3:.1f}) "
                  f"in region {r_idx}", flush=True)
        if bic < best["bic"]:
            best = dict(k=k, bic=bic, spins=[list(s) for s in spins], rss=rss)
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                break

    return dict(best=best, trace=trace, regions=regions)
