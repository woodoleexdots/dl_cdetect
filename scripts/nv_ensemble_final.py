"""FINAL ensemble-region hybrid run for NV1 and NV2.

Candidate regions = UNION of
  (a) every trained PeriodFormer model's thresholded P(spin) curve
      (NV1: +-60 / +-120 / +-200 grids; NV2: +-400 / +-600 grids), and
  (b) +-4 kHz intervals around every spin that plain wide-bound DE found
      on the real data (candidates PF grids occasionally missed).

One region-constrained greedy-DE fit per NV then produces the single
definitive spin list. Saves 'nv1_ensemble' / 'nv2_ensemble' into
hybrid_results.json and overlay figures 21/22.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpmg.hybrid import hybrid_greedy_fit, pf_candidate_regions
from cpmg.physics import cpmg_M
from cpmg.represent import envelope_normalize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "benchmark_v2"
PF = ROOT / "results" / "periodformer"
FIGS = ROOT / "results" / "figs"
B_OURS = 440.1
US = 1e-6

# plain-DE discoveries on real data (kHz) from earlier wide-bound runs
DE_CANDIDATES_NV1 = [9.2, -5.6, 91.2, 40.7, 3.3, 65.2, 23.7, -87.6]
DE_CANDIDATES_NV2 = [-51.5, 51.1, 346.2, -39.2, -152.3, -14.1]

NV1_CURVES = [("nv1_curve.npz", 0.25, 3e3),
              ("curve_NV1_NV1_wide.npz", 0.25, 3e3),
              ("curve_NV1_NV1_w200.npz", 0.25, 4e3)]
NV2_CURVES = [("curve_NV2_NV2_strong.npz", 0.15, 8e3),
              ("curve_NV2_NV2_w600.npz", 0.15, 10e3)]


def merge_intervals(intervals):
    out = []
    for lo, hi in sorted(intervals):
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [(lo, hi) for lo, hi in out]


def ensemble_regions(curve_specs, de_candidates_khz, de_margin=4e3):
    intervals = []
    for fname, thresh, margin in curve_specs:
        d = np.load(PF / fname)
        centers = d["centers_hz"] if "centers_hz" in d else d["centers"]
        intervals += [list(r) for r in
                      pf_candidate_regions(d["p"], centers, thresh, margin)]
    for a in de_candidates_khz:
        intervals.append([a * 1e3 - de_margin, a * 1e3 + de_margin])
    return merge_intervals(intervals)


def main():
    res = json.loads((OUT / "hybrid_results.json").read_text())

    # ---------------- NV1 ----------------
    nv1 = pd.read_excel(ROOT / "dataset" / "exp_dataset" / "CPMG_NV1.xlsx")
    tau1 = nv1["a"].to_numpy(float)
    m1 = np.array([envelope_normalize(tau1, nv1[c].to_numpy(float))[0]
                   for c in ["CPMG8", "CPMG16", "CPMG20"]], dtype=np.float32)
    regions1 = ensemble_regions(NV1_CURVES, DE_CANDIDATES_NV1)
    print(f"NV1 ensemble regions ({len(regions1)}): "
          f"{[(round(l/1e3), round(h/1e3)) for l, h in regions1]}", flush=True)
    fit1 = hybrid_greedy_fit(tau1, m1, (8, 16, 20), B_OURS, regions1,
                             max_spins=18, b_bounds=(3e3, 65e3),
                             de_maxiter=60, de_popsize=20, seed=0, verbose=True)
    spins1 = sorted([[round(s[0] / 1e3, 1), round(s[1] / 1e3, 1)]
                     for s in fit1["best"]["spins"]])
    rmse1 = [float(np.sqrt(np.mean(
        (cpmg_M(tau1, np.array(fit1["best"]["spins"]), n, B_OURS) - m1[j]) ** 2)))
        for j, n in enumerate((8, 16, 20))]
    print(f"NV1 ENSEMBLE k*={fit1['best']['k']}: {spins1}  RMSE={np.round(rmse1,3)}",
          flush=True)
    res["nv1_ensemble"] = dict(
        regions_khz=[[round(l / 1e3, 1), round(h / 1e3, 1)] for l, h in regions1],
        k=fit1["best"]["k"], spins_khz=spins1, rmse=[round(r, 3) for r in rmse1])

    # ---------------- NV2 ----------------
    nv2 = pd.read_excel(ROOT / "dataset" / "exp_dataset" / "CPMG_NV2.xlsx")
    tau2 = nv2["Time"].to_numpy(float)
    m2 = np.array([envelope_normalize(tau2, nv2["CPMG16"].to_numpy(float))[0]],
                  dtype=np.float32)
    regions2 = ensemble_regions(NV2_CURVES, DE_CANDIDATES_NV2, de_margin=8e3)
    print(f"NV2 ensemble regions ({len(regions2)}): "
          f"{[(round(l/1e3), round(h/1e3)) for l, h in regions2]}", flush=True)
    fit2 = hybrid_greedy_fit(tau2, m2, (16,), B_OURS, regions2,
                             max_spins=12, b_bounds=(30e3, 350e3),
                             de_maxiter=80, de_popsize=20, seed=0, verbose=True)
    spins2 = sorted([[round(s[0] / 1e3, 1), round(s[1] / 1e3, 1)]
                     for s in fit2["best"]["spins"]])
    rmse2 = float(np.sqrt(np.mean(
        (cpmg_M(tau2, np.array(fit2["best"]["spins"]), 16, B_OURS) - m2[0]) ** 2)))
    print(f"NV2 ENSEMBLE k*={fit2['best']['k']}: {spins2}  RMSE={rmse2:.3f}", flush=True)
    res["nv2_ensemble"] = dict(
        regions_khz=[[round(l / 1e3, 1), round(h / 1e3, 1)] for l, h in regions2],
        k=fit2["best"]["k"], spins_khz=spins2, rmse=round(rmse2, 3))

    (OUT / "hybrid_results.json").write_text(json.dumps(res, indent=2))
    print("saved ->", OUT / "hybrid_results.json", flush=True)

    # ---------------- overlays ----------------
    fig, axes = plt.subplots(3, 1, figsize=(14, 8.5))
    ab1 = np.array(fit1["best"]["spins"])
    for ax, (n, col) in zip(axes, [(8, "CPMG8"), (16, "CPMG16"), (20, "CPMG20")]):
        m, _ = envelope_normalize(tau1, nv1[col].to_numpy(float))
        m_f = cpmg_M(tau1, ab1, n, B_OURS)
        ax.plot(tau1 / US, m, ".", ms=2, color="0.55", alpha=0.8)
        ax.plot(tau1 / US, m_f, "-", lw=1.0, color="tab:red")
        ax.set_title(f"N={n}  RMSE={np.sqrt(np.mean((m_f-m)**2)):.3f}",
                     fontsize=9, loc="right")
        ax.set_ylabel("M")
    axes[-1].set_xlabel("tau (us)")
    fig.suptitle(f"NV1 ENSEMBLE final, k*={fit1['best']['k']}: "
                 + ", ".join(f"({a:+.0f},{b:.0f})" for a, b in spins1), fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "21_ensemble_overlay_NV1.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6.5))
    ab2 = np.array(fit2["best"]["spins"])
    m_f2 = cpmg_M(tau2, ab2, 16, B_OURS)
    for ax, zoom in zip(axes, [None, (0, 5)]):
        ax.plot(tau2 / US, m2[0], ".", ms=2.2, color="0.55", alpha=0.8)
        ax.plot(tau2 / US, m_f2, "-", lw=1.0, color="tab:red")
        if zoom:
            ax.set_xlim(*zoom)
        ax.set_ylabel("M")
    axes[0].set_title(f"full  RMSE={rmse2:.3f}", fontsize=9, loc="left")
    axes[1].set_title("zoom 0-5 us", fontsize=9, loc="left")
    axes[1].set_xlabel("tau (us)")
    fig.suptitle(f"NV2 ENSEMBLE final, k*={fit2['best']['k']}: "
                 + ", ".join(f"({a:+.0f},{b:.0f})" for a, b in spins2), fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "22_ensemble_overlay_NV2.png", dpi=150)
    plt.close(fig)
    print("figures 21/22 saved", flush=True)


if __name__ == "__main__":
    main()
