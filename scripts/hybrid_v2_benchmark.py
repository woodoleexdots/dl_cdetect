"""Hybrid v2: PF candidate regions + region-constrained RJMCMC.

Combines PF's zero-false-positive region proposals (precision/mismatch
robustness) with RJMCMC's uncapped in-region enumeration (recall).
Evaluated on the same benchmark-v2 suites; saves
results/benchmark_v2/hybrid_v2_results.json.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpmg.hybrid import pf_candidate_regions
from cpmg.periodformer import TokenBuilder
from cpmg.rjmcmc import RJMCMC

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "benchmark_v2"

spec = importlib.util.spec_from_file_location("bv2", ROOT / "scripts" / "benchmark_v2.py")
bv2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bv2)
spec2 = importlib.util.spec_from_file_location("pfb", ROOT / "scripts" / "pf_benchmark_v2.py")
pfb = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(pfb)

CENTERS = pfb.CENTERS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

bath, _ = bv2.load_bath()
box = (np.abs(bath[:, 0]) <= 60e3) & (bath[:, 1] >= 5e3)
GT_A = bath[box][:, 0]


def main():
    res_file = OUT / "hybrid_v2_results.json"
    res = json.loads(res_file.read_text()) if res_file.exists() else {}

    a_room = bv2.ARMS["B_room"]
    tau_room = np.arange(1, a_room["n_tau"] + 1) * a_room["dt"]
    cfg_room = pfb.make_gen_cfg("B_room", a_room["b_field"])
    cfg_mis = pfb.make_gen_cfg("C_mismatch", a_room["b_field"] + bv2.DB_MISMATCH)
    model_room, builder_room, _ = pfb.train_pf("pf_room", tau_room, cfg_room,
                                               13, DEVICE, 25000, 25, 1)
    builder_mis = TokenBuilder(tau_room, CENTERS, cfg_mis, DEVICE, s_max=13)

    a_cryo = bv2.ARMS["A_cryo"]
    tau_cryo = np.arange(1, a_cryo["n_tau"] + 1) * a_cryo["dt"]
    cfg_cryo = pfb.make_gen_cfg("A_cryo", a_cryo["b_field"])
    from cpmg.physics import target_period
    smax = int(np.floor(tau_cryo[-1] / target_period(0, 30e3, a_cryo["b_field"])))
    model_cryo, builder_cryo, _ = pfb.train_pf("pf_cryo", tau_cryo, cfg_cryo,
                                               smax, DEVICE, 15000, 25, 0)

    plans = [
        ("A_cryo", model_cryo, builder_cryo, tau_cryo, 4, 2e3,
         a_cryo["b_field"], 25, 2e3, 40000),
        ("B_room", model_room, builder_room, tau_room, 8, 4e3,
         a_room["b_field"], 15, 3e3, 30000),
        ("C_mismatch", model_room, builder_mis, tau_room, 8, 4e3,
         a_room["b_field"] + bv2.DB_MISMATCH, 15, 3e3, 30000),
    ]
    for arm, model, builder, tau, n_real, tol, ana, max_spins, margin, n_iter in plans:
        if arm in res:
            print(f"[{arm}] cached", flush=True)
            continue
        a = bv2.ARMS[arm]
        rows, t0 = [], time.time()
        for i in range(n_real):
            _, m_recs = bv2.make_arm_dataset(arm, bath, 100 + i)
            with torch.no_grad():
                p = torch.sigmoid(model(builder(
                    torch.from_numpy(m_recs)[None].to(DEVICE))))[0].cpu().numpy()
            regions = pf_candidate_regions(p, CENTERS, thresh=0.35,
                                           margin_hz=margin)
            if not regions:
                rows.append(dict(real=i, tp=0, fp=0, fn=len(GT_A), k=0))
                continue
            mc = RJMCMC(tau, m_recs, a["n_pulses"], ana, sigma=a["sigma"],
                        max_spins=max_spins, seed=i, regions=regions)
            out = mc.run(n_iter=n_iter)
            det = np.array([s[0] for s in out["map_spins"]])
            tp, fp, fn, _ = bv2.match_score(det, GT_A, tol)
            rows.append(dict(real=i, tp=tp, fp=fp, fn=fn,
                             k=len(out["map_spins"]),
                             modal_k=out["modal_k"], n_regions=len(regions)))
            print(f"[{arm}] {i+1}/{n_real} k={len(out['map_spins'])} "
                  f"tp={tp} fp={fp} regions={len(regions)} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        tp = sum(r["tp"] for r in rows)
        fp = sum(r["fp"] for r in rows)
        fn = sum(r["fn"] for r in rows)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        res[arm] = dict(rows=rows, precision=round(prec, 3),
                        recall=round(rec, 3),
                        f1=round(2 * prec * rec / max(prec + rec, 1e-9), 3),
                        mean_k=round(float(np.mean([r["k"] for r in rows])), 1))
        res_file.write_text(json.dumps(res, indent=1))
        print(f"[{arm}] SUMMARY "
              f"{json.dumps({k: res[arm][k] for k in ['precision','recall','f1','mean_k']})}",
              flush=True)
    print("saved ->", res_file, flush=True)


if __name__ == "__main__":
    main()
