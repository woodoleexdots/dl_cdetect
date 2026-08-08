"""Widest-range hybrid fits + comparison against the pptx anchor couplings.

NV1: PF _NV1_w200 curve (+-200 kHz, 2.5 kHz step) -> regions -> hybrid DE.
NV2: PF _NV2_w600 curve (+-600 kHz, 6 kHz step)  -> regions -> hybrid DE
     (round-2 envelope from the previous refinement is NOT reused; the
      standard quantile envelope is used for comparability).

Prints a comparison table vs the anchor values (eyeball estimates from
NV_CPMG.pptx), accounting for the A_par sign-convention difference.
Saves 'nv1_w200' / 'nv2_w600' into hybrid_results.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpmg.hybrid import hybrid_greedy_fit, pf_candidate_regions
from cpmg.physics import cpmg_M
from cpmg.represent import envelope_normalize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "benchmark_v2"
PF = ROOT / "results" / "periodformer"
B_OURS = 440.1

ANCHORS_NV1 = [(-88, 47), (8, 31), (-38, 37), (-5, 35)]
ANCHORS_NV2 = [(-340, 290), (150, 110), (-42, 150)]


def spin_list(fit):
    return sorted([[round(s[0] / 1e3, 1), round(s[1] / 1e3, 1)]
                   for s in fit["best"]["spins"]])


def compare(anchors, spins, tol_khz=12.0):
    """Match anchors to found spins by |A| (sign-convention agnostic)."""
    rows = []
    for a_a, a_b in anchors:
        cands = [s for s in spins
                 if min(abs(s[0] - a_a), abs(s[0] + a_a)) <= tol_khz]
        best = min(cands, key=lambda s: min(abs(s[0] - a_a), abs(s[0] + a_a))) \
            if cands else None
        rows.append((a_a, a_b, best))
    return rows


def main():
    res = json.loads((OUT / "hybrid_results.json").read_text())

    # ---------------- NV1 +-200 ----------------
    nv1 = pd.read_excel(ROOT / "dataset" / "exp_dataset" / "CPMG_NV1.xlsx")
    tau1 = nv1["a"].to_numpy(float)
    m1 = np.array([envelope_normalize(tau1, nv1[c].to_numpy(float))[0]
                   for c in ["CPMG8", "CPMG16", "CPMG20"]], dtype=np.float32)
    d = np.load(PF / "curve_NV1_NV1_w200.npz")
    regions1 = pf_candidate_regions(d["p"], d["centers_hz"], thresh=0.25,
                                    margin_hz=4e3)
    print(f"NV1 w200 regions: {[(round(l/1e3), round(h/1e3)) for l, h in regions1]}",
          flush=True)
    fit1 = hybrid_greedy_fit(tau1, m1, (8, 16, 20), B_OURS, regions1,
                             max_spins=16, b_bounds=(3e3, 65e3),
                             de_maxiter=60, de_popsize=20, seed=0, verbose=True)
    spins1 = spin_list(fit1)
    rmse1 = [float(np.sqrt(np.mean(
        (cpmg_M(tau1, np.array(fit1["best"]["spins"]), n, B_OURS) - m1[j]) ** 2)))
        for j, n in enumerate((8, 16, 20))]
    print(f"NV1 w200 k*={fit1['best']['k']}: {spins1}  RMSE={np.round(rmse1,3)}",
          flush=True)
    res["nv1_w200"] = dict(k=fit1["best"]["k"], spins_khz=spins1,
                           rmse=[round(r, 3) for r in rmse1],
                           regions_khz=[[round(l / 1e3, 1), round(h / 1e3, 1)]
                                        for l, h in regions1])

    # ---------------- NV2 +-600 ----------------
    nv2 = pd.read_excel(ROOT / "dataset" / "exp_dataset" / "CPMG_NV2.xlsx")
    tau2 = nv2["Time"].to_numpy(float)
    m2 = np.array([envelope_normalize(tau2, nv2["CPMG16"].to_numpy(float))[0]],
                  dtype=np.float32)
    d2 = np.load(PF / "curve_NV2_NV2_w600.npz")
    regions2 = pf_candidate_regions(d2["p"], d2["centers_hz"], thresh=0.15,
                                    margin_hz=10e3)
    print(f"NV2 w600 regions: {[(round(l/1e3), round(h/1e3)) for l, h in regions2]}",
          flush=True)
    fit2 = hybrid_greedy_fit(tau2, m2, (16,), B_OURS, regions2,
                             max_spins=10, b_bounds=(30e3, 350e3),
                             de_maxiter=80, de_popsize=20, seed=0, verbose=True)
    spins2 = spin_list(fit2)
    rmse2 = float(np.sqrt(np.mean(
        (cpmg_M(tau2, np.array(fit2["best"]["spins"]), 16, B_OURS) - m2[0]) ** 2)))
    print(f"NV2 w600 k*={fit2['best']['k']}: {spins2}  RMSE={rmse2:.3f}", flush=True)
    res["nv2_w600"] = dict(k=fit2["best"]["k"], spins_khz=spins2,
                           rmse=round(rmse2, 3),
                           regions_khz=[[round(l / 1e3, 1), round(h / 1e3, 1)]
                                        for l, h in regions2])

    (OUT / "hybrid_results.json").write_text(json.dumps(res, indent=2))

    # ---------------- anchor comparison ----------------
    print("\n===== anchor comparison (|A| matching, sign-convention agnostic) =====")
    for name, anchors, spins in [("NV1", ANCHORS_NV1, spins1),
                                 ("NV2", ANCHORS_NV2, spins2)]:
        print(f"[{name}]")
        for a_a, a_b, best in compare(anchors, spins):
            if best is None:
                print(f"  anchor ({a_a:+d},{a_b}) -> NOT FOUND", flush=True)
            else:
                print(f"  anchor ({a_a:+d},{a_b}) -> ours ({best[0]:+.1f},{best[1]:.1f})",
                      flush=True)
    print("saved ->", OUT / "hybrid_results.json", flush=True)


if __name__ == "__main__":
    main()
