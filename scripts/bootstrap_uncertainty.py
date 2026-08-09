"""Reviewer action item 4: residual-bootstrap error bars for the NV1 final
ensemble spin list.

Procedure: residuals of the best fit are resampled (iid over tau; residual
autocorrelation is checked and reported) and added back to the model curve;
each replicate is re-fitted by local L-BFGS polish (A restricted to +-3 kHz
around the point estimate). Percentile CIs per spin are reported.

Output: results/benchmark_v2/nv1_bootstrap.json + printed table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpmg.physics import cpmg_M
from cpmg.represent import envelope_normalize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "benchmark_v2"
B_OURS = 440.1
N_PULSES = (8, 16, 20)
N_BOOT = 60


def model_m(tau, ab_flat):
    ab = np.asarray(ab_flat, dtype=np.float64).reshape(-1, 2)
    return np.array([cpmg_M(tau, ab, n, B_OURS) for n in N_PULSES])


def main():
    nv1 = pd.read_excel(ROOT / "dataset" / "exp_dataset" / "CPMG_NV1.xlsx")
    tau = nv1["a"].to_numpy(float)
    m_data = np.array([envelope_normalize(tau, nv1[c].to_numpy(float))[0]
                       for c in ["CPMG8", "CPMG16", "CPMG20"]])
    spins = np.array(json.loads((OUT / "hybrid_results.json").read_text())
                     ["nv1_ensemble"]["spins_khz"]) * 1e3  # (14, 2) Hz

    m_fit = model_m(tau, spins.ravel())
    resid = m_data - m_fit
    # residual autocorrelation check (lag-1)
    ac1 = [float(np.corrcoef(r[:-1], r[1:])[0, 1]) for r in resid]
    print("residual lag-1 autocorr per channel:", np.round(ac1, 3), flush=True)

    x0 = spins.ravel()
    lb, ub = [], []
    for a, b in spins:
        lb += [a - 3e3, max(3e3, b - 15e3)]
        ub += [a + 3e3, min(65e3, b + 15e3)]
    bounds = list(zip(lb, ub))

    rng = np.random.default_rng(0)
    T = len(tau)
    samples = []
    for it in range(N_BOOT):
        r_star = resid[:, rng.integers(0, T, size=T)]
        m_star = m_fit + r_star

        def obj(x):
            return float(np.sum((model_m(tau, x) - m_star) ** 2))

        res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds,
                       options=dict(maxiter=200))
        samples.append(res.x.reshape(-1, 2))
        if (it + 1) % 10 == 0:
            print(f"bootstrap {it+1}/{N_BOOT}", flush=True)

    S = np.stack(samples)  # (N_BOOT, 14, 2)
    rows = []
    print("\nspin | A [kHz] (95% CI)          | B [kHz] (95% CI)")
    for j, (a, b) in enumerate(spins):
        a_lo, a_hi = np.percentile(S[:, j, 0], [2.5, 97.5]) / 1e3
        b_lo, b_hi = np.percentile(S[:, j, 1], [2.5, 97.5]) / 1e3
        a_sd = S[:, j, 0].std() / 1e3
        b_sd = S[:, j, 1].std() / 1e3
        rows.append(dict(A_khz=round(a / 1e3, 1), B_khz=round(b / 1e3, 1),
                         A_ci=[round(a_lo, 2), round(a_hi, 2)],
                         B_ci=[round(b_lo, 2), round(b_hi, 2)],
                         A_sd=round(a_sd, 2), B_sd=round(b_sd, 2)))
        print(f"{j+1:4d} | {a/1e3:+7.1f} [{a_lo:+7.2f},{a_hi:+7.2f}] "
              f"| {b/1e3:6.1f} [{b_lo:6.2f},{b_hi:6.2f}]", flush=True)

    (OUT / "nv1_bootstrap.json").write_text(json.dumps(
        dict(n_boot=N_BOOT, resid_lag1_autocorr=[round(a, 3) for a in ac1],
             spins=rows), indent=1))
    print("saved ->", OUT / "nv1_bootstrap.json", flush=True)


if __name__ == "__main__":
    main()
