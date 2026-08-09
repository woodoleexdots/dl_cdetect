"""Reviewer action item 1: multi-seed training study with error bars and
significance tests.

- pf_room : 5 training seeds (1 existing + 4 new) -> PF-alone and PF->DE
            hybrid evaluated on benchmark-v2 arms B (room) and C (mismatch)
- pf_cryo : 3 training seeds (1 existing + 2 new) -> arm A
- DE-alone: 3 optimizer seeds per arm (no training)

Outputs results/benchmark_v2/multiseed.json with per-seed per-realization
rows, mean+-std F1 per method/arm, and Wilcoxon signed-rank p-values for
hybrid vs DE on matched realizations.
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpmg.datagen import GenConfig
from cpmg.defit import greedy_de_fit
from cpmg.hybrid import hybrid_greedy_fit, pf_candidate_regions
from cpmg.periodformer import TokenBuilder

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


def f1_of(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-9)


def eval_seed(arm, model, builder, tau, n_real, tol, ana_field, max_spins,
              margin, de_seed_base=0):
    """Return per-realization rows for PF-alone and hybrid."""
    from scipy.signal import find_peaks

    rows = []
    for i in range(n_real):
        _, m_recs = bv2.make_arm_dataset(arm, bath, 100 + i)
        p = pfb.__dict__["torch"].sigmoid(
            model(builder(torch.from_numpy(m_recs)[None].to(DEVICE)))
        )[0].detach().cpu().numpy()
        pk, _ = find_peaks(p, height=0.5, prominence=0.1)
        det_pf = CENTERS[pk]
        tp, fp, fn, _ = bv2.match_score(det_pf, GT_A, tol)
        regions = pf_candidate_regions(p, CENTERS, thresh=0.35, margin_hz=margin)
        fit = hybrid_greedy_fit(tau, m_recs, bv2.ARMS[arm]["n_pulses"],
                                ana_field, regions, max_spins=max_spins,
                                seed=de_seed_base + i)
        det_h = np.array([s[0] for s in fit["best"]["spins"]])
        tp2, fp2, fn2, _ = bv2.match_score(det_h, GT_A, tol)
        rows.append(dict(real=i, pf=(tp, fp, fn), hybrid=(tp2, fp2, fn2)))
    return rows


def main():
    res_file = OUT / "multiseed.json"
    res = json.loads(res_file.read_text()) if res_file.exists() else {}

    a_room = bv2.ARMS["B_room"]
    tau_room = np.arange(1, a_room["n_tau"] + 1) * a_room["dt"]
    cfg_room = pfb.make_gen_cfg("B_room", a_room["b_field"])
    cfg_mis = pfb.make_gen_cfg("C_mismatch", a_room["b_field"] + bv2.DB_MISMATCH)

    a_cryo = bv2.ARMS["A_cryo"]
    tau_cryo = np.arange(1, a_cryo["n_tau"] + 1) * a_cryo["dt"]
    cfg_cryo = pfb.make_gen_cfg("A_cryo", a_cryo["b_field"])
    from cpmg.physics import target_period
    smax_cryo = int(np.floor(tau_cryo[-1] / target_period(0, 30e3, a_cryo["b_field"])))

    # ---------------- pf_room seeds ----------------
    for seed in [1, 2, 3, 4, 5]:
        key = f"room_seed{seed}"
        if key in res:
            print(f"[{key}] cached", flush=True)
            continue
        t0 = time.time()
        name = "pf_room" if seed == 1 else f"pf_room_s{seed}"
        model, builder, _ = pfb.train_pf(name, tau_room, cfg_room, 13, DEVICE,
                                         n_train=25000, epochs=25, seed=seed)
        builder_mis = TokenBuilder(tau_room, CENTERS, cfg_mis, DEVICE, s_max=13)
        rows_b = eval_seed("B_room", model, builder, tau_room, 8, 4e3,
                           a_room["b_field"], 12, 3e3, de_seed_base=seed * 1000)
        rows_c = eval_seed("C_mismatch", model, builder_mis, tau_room, 8, 4e3,
                           a_room["b_field"] + bv2.DB_MISMATCH, 12, 3e3,
                           de_seed_base=seed * 1000)
        res[key] = dict(B_room=rows_b, C_mismatch=rows_c,
                        train_s=round(time.time() - t0, 1))
        res_file.write_text(json.dumps(res, indent=1))
        print(f"[{key}] done ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- pf_cryo seeds ----------------
    for seed in [0, 1, 2]:
        key = f"cryo_seed{seed}"
        if key in res:
            print(f"[{key}] cached", flush=True)
            continue
        t0 = time.time()
        name = "pf_cryo" if seed == 0 else f"pf_cryo_s{seed}"
        model, builder, _ = pfb.train_pf(name, tau_cryo, cfg_cryo, smax_cryo,
                                         DEVICE, n_train=15000, epochs=25,
                                         seed=seed)
        rows_a = eval_seed("A_cryo", model, builder, tau_cryo, 4, 2e3,
                           a_cryo["b_field"], 20, 2e3, de_seed_base=seed * 1000)
        res[key] = dict(A_cryo=rows_a, train_s=round(time.time() - t0, 1))
        res_file.write_text(json.dumps(res, indent=1))
        print(f"[{key}] done ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- DE-alone optimizer seeds ----------------
    for dseed in [0, 1, 2]:
        key = f"de_seed{dseed}"
        if key in res:
            print(f"[{key}] cached", flush=True)
            continue
        t0 = time.time()
        entry = {}
        for arm, tau, tol, ms in [("B_room", tau_room, 4e3, 10),
                                  ("C_mismatch", tau_room, 4e3, 10)]:
            ana = a_room["b_field"] + (bv2.DB_MISMATCH if arm == "C_mismatch" else 0)
            rows = []
            for i in range(8):
                _, m_recs = bv2.make_arm_dataset(arm, bath, 100 + i)
                fit = greedy_de_fit(tau, m_recs, a_room["n_pulses"], ana,
                                    max_spins=ms, a_bounds=(-60e3, 60e3),
                                    b_bounds=(3e3, 65e3), de_maxiter=50,
                                    de_popsize=20, seed=dseed * 500 + i,
                                    verbose=False)
                det = np.array([s[0] for s in fit["best"]["spins"]])
                rows.append(dict(real=i, de=bv2.match_score(det, GT_A, tol)[:3]))
            entry[arm] = rows
        res[key] = entry
        res_file.write_text(json.dumps(res, indent=1))
        print(f"[{key}] done ({time.time()-t0:.0f}s)", flush=True)

    # ---------------- aggregate + significance ----------------
    from scipy.stats import wilcoxon

    summary = {}
    for arm in ["B_room", "C_mismatch"]:
        pf_f1s, hy_f1s = [], []
        hy_by_real = {}
        for seed in [1, 2, 3, 4, 5]:
            rows = res[f"room_seed{seed}"][arm]
            pf_f1s.append(np.mean([f1_of(*r["pf"]) for r in rows]))
            hy_f1s.append(np.mean([f1_of(*r["hybrid"]) for r in rows]))
            for r in rows:
                hy_by_real.setdefault(r["real"], []).append(f1_of(*r["hybrid"]))
        de_f1s, de_by_real = [], {}
        for dseed in [0, 1, 2]:
            rows = res[f"de_seed{dseed}"][arm]
            de_f1s.append(np.mean([f1_of(*r["de"]) for r in rows]))
            for r in rows:
                de_by_real.setdefault(r["real"], []).append(f1_of(*r["de"]))
        # paired per-realization means (hybrid vs de)
        h = [np.mean(hy_by_real[i]) for i in sorted(hy_by_real)]
        d = [np.mean(de_by_real[i]) for i in sorted(de_by_real)]
        try:
            stat, pval = wilcoxon(h, d)
        except ValueError:
            pval = 1.0
        summary[arm] = dict(
            pf_f1=f"{np.mean(pf_f1s):.3f}±{np.std(pf_f1s):.3f}",
            hybrid_f1=f"{np.mean(hy_f1s):.3f}±{np.std(hy_f1s):.3f}",
            de_f1=f"{np.mean(de_f1s):.3f}±{np.std(de_f1s):.3f}",
            wilcoxon_hybrid_vs_de_p=round(float(pval), 4))
    pf_a, hy_a = [], []
    for seed in [0, 1, 2]:
        rows = res[f"cryo_seed{seed}"]["A_cryo"]
        pf_a.append(np.mean([f1_of(*r["pf"]) for r in rows]))
        hy_a.append(np.mean([f1_of(*r["hybrid"]) for r in rows]))
    summary["A_cryo"] = dict(
        pf_f1=f"{np.mean(pf_a):.3f}±{np.std(pf_a):.3f}",
        hybrid_f1=f"{np.mean(hy_a):.3f}±{np.std(hy_a):.3f}")
    res["summary"] = summary
    res_file.write_text(json.dumps(res, indent=1))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
