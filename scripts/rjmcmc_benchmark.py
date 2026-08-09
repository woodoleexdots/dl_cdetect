"""Evaluate the native RJMCMC baseline on benchmark-v2 arms (item 2).

Arms B/C: 8 realizations x 30k iterations; arm A: 4 x 40k.
Saves results/benchmark_v2/rjmcmc_results.json (same metric protocol).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpmg.rjmcmc import RJMCMC

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "benchmark_v2"

spec = importlib.util.spec_from_file_location("bv2", ROOT / "scripts" / "benchmark_v2.py")
bv2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bv2)

bath, _ = bv2.load_bath()
box = (np.abs(bath[:, 0]) <= 60e3) & (bath[:, 1] >= 5e3)
GT_A = bath[box][:, 0]


def main():
    res_file = OUT / "rjmcmc_results.json"
    res = json.loads(res_file.read_text()) if res_file.exists() else {}
    plans = [
        ("B_room", 8, 4e3, 30000, 12, 0.0),
        ("C_mismatch", 8, 4e3, 30000, 12, bv2.DB_MISMATCH),
        ("A_cryo", 4, 2e3, 40000, 22, 0.0),
    ]
    for arm, n_real, tol, n_iter, max_spins, db in plans:
        if arm in res:
            print(f"[{arm}] cached", flush=True)
            continue
        a = bv2.ARMS[arm]
        tau = np.arange(1, a["n_tau"] + 1) * a["dt"]
        rows, ks, t0 = [], [], time.time()
        for i in range(n_real):
            _, m_recs = bv2.make_arm_dataset(arm, bath, 100 + i)
            mc = RJMCMC(tau, m_recs, a["n_pulses"], a["b_field"] + db,
                        sigma=a["sigma"], max_spins=max_spins, seed=i)
            out = mc.run(n_iter=n_iter)
            det = np.array([s[0] for s in out["map_spins"]])
            tp, fp, fn, errs = bv2.match_score(det, GT_A, tol)
            rows.append(dict(real=i, tp=tp, fp=fp, fn=fn,
                             modal_k=out["modal_k"],
                             accept=round(out["accept_rate"], 3),
                             err_khz=[round(e / 1e3, 2) for e in errs]))
            ks.append(len(out["map_spins"]))
            print(f"[{arm}] {i+1}/{n_real} map_k={len(out['map_spins'])} "
                  f"modal_k={out['modal_k']} tp={tp} fp={fp} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        tp = sum(r["tp"] for r in rows)
        fp = sum(r["fp"] for r in rows)
        fn = sum(r["fn"] for r in rows)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        res[arm] = dict(rows=rows, mean_map_k=round(float(np.mean(ks)), 1),
                        precision=round(prec, 3), recall=round(rec, 3),
                        f1=round(2 * prec * rec / max(prec + rec, 1e-9), 3))
        res_file.write_text(json.dumps(res, indent=1))
        print(f"[{arm}] SUMMARY {json.dumps({k: res[arm][k] for k in ['precision','recall','f1','mean_map_k']})}",
              flush=True)
    print("saved ->", res_file, flush=True)


if __name__ == "__main__":
    main()
