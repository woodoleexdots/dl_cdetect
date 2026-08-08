"""Evaluate the PF->DE hybrid on benchmark-v2 arms + real NV1.

Stage 1 uses the arm-matched PeriodFormer checkpoints from pf_benchmark_v2
(pf_cryo / pf_room). Stage 2 is the region-constrained greedy DE.
Metrics are directly comparable to results.json (2021 / cdetect-DE) and
pf_results.json (PF alone) — same suites, same seeds, same tolerances.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpmg.hybrid import hybrid_greedy_fit, pf_candidate_regions
from cpmg.periodformer import PeriodFormer, TokenBuilder

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "benchmark_v2"

spec = importlib.util.spec_from_file_location("bv2", ROOT / "scripts" / "benchmark_v2.py")
bv2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bv2)
spec2 = importlib.util.spec_from_file_location("pfb", ROOT / "scripts" / "pf_benchmark_v2.py")
pfb = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(pfb)

CENTERS = pfb.CENTERS


def load_pf(name, cfg, tau, s_max, device):
    builder = TokenBuilder(tau, CENTERS, cfg, device, s_max=s_max)
    model = PeriodFormer(in_ch=len(cfg.n_pulses), n_tokens=len(CENTERS),
                         s_max=s_max).to(device)
    model.load_state_dict(torch.load(OUT / f"{name}.pt", map_location=device))
    model.eval()
    return model, builder


def pf_curve(model, builder, m_recs, device):
    x = torch.from_numpy(np.asarray(m_recs, dtype=np.float32))[None].to(device)
    with torch.no_grad():
        return torch.sigmoid(model(builder(x)))[0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=8)
    ap.add_argument("--n-real-cryo", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thresh", type=float, default=0.35)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bath, labels = bv2.load_bath()
    box = (np.abs(bath[:, 0]) <= 60e3) & (bath[:, 1] >= 5e3)
    gt_a = bath[box][:, 0]

    from cpmg.physics import target_period

    a_cryo = bv2.ARMS["A_cryo"]
    tau_cryo = np.arange(1, a_cryo["n_tau"] + 1) * a_cryo["dt"]
    cfg_cryo = pfb.make_gen_cfg("A_cryo", a_cryo["b_field"])
    s_max_cryo = int(np.floor(tau_cryo[-1] / target_period(0, 30e3, a_cryo["b_field"])))
    model_cryo, builder_cryo = load_pf("pf_cryo", cfg_cryo, tau_cryo, s_max_cryo, device)

    a_room = bv2.ARMS["B_room"]
    tau_room = np.arange(1, a_room["n_tau"] + 1) * a_room["dt"]
    cfg_room = pfb.make_gen_cfg("B_room", a_room["b_field"])
    model_room, builder_room = load_pf("pf_room", cfg_room, tau_room, 13, device)
    cfg_mis = pfb.make_gen_cfg("C_mismatch", a_room["b_field"] + bv2.DB_MISMATCH)
    builder_mis = TokenBuilder(tau_room, CENTERS, cfg_mis, device, s_max=13)

    plans = [
        ("A_cryo", model_cryo, builder_cryo, tau_cryo, args.n_real_cryo, 2e3,
         a_cryo["b_field"], 20, 2e3),
        ("B_room", model_room, builder_room, tau_room, args.n_real, 4e3,
         a_room["b_field"], 12, 3e3),
        ("C_mismatch", model_room, builder_mis, tau_room, args.n_real, 4e3,
         a_room["b_field"] + bv2.DB_MISMATCH, 12, 3e3),
    ]

    results, details = {}, {}
    for arm, model, builder, tau, n_real, tol, ana_field, max_spins, margin in plans:
        a = bv2.ARMS[arm]
        rows, det_lists, ks, nreg = [], [], [], []
        t_arm = time.time()
        for i in range(n_real):
            _, m_recs = bv2.make_arm_dataset(arm, bath, args.seed + 100 + i)
            p = pf_curve(model, builder, m_recs, device)
            regions = pf_candidate_regions(p, CENTERS, thresh=args.thresh,
                                           margin_hz=margin)
            fit = hybrid_greedy_fit(tau, m_recs, a["n_pulses"], ana_field,
                                    regions, max_spins=max_spins,
                                    seed=args.seed + i)
            det = np.array([s[0] for s in fit["best"]["spins"]])
            rows.append(bv2.match_score(det, gt_a, tol))
            det_lists.append(sorted(round(v / 1e3, 1) for v in det))
            ks.append(fit["best"]["k"])
            nreg.append(len(regions))
            print(f"[{arm}] {i+1}/{n_real}: {len(regions)} regions, k*={fit['best']['k']} "
                  f"({time.time()-t_arm:.0f}s)", flush=True)
        m = bv2.agg(rows, len(gt_a))
        m["mean_k"] = round(float(np.mean(ks)), 1)
        m["mean_regions"] = round(float(np.mean(nreg)), 1)
        results[arm] = m
        details[arm] = det_lists
        print(f"[hybrid] {arm}: {json.dumps(m)}", flush=True)
        (OUT / "hybrid_results.json").write_text(
            json.dumps(dict(metrics=results, detections=details), indent=2))

    # ---- real NV1 ----
    import pandas as pd

    from cpmg.represent import envelope_normalize
    nv1 = pd.read_excel(ROOT / "dataset" / "exp_dataset" / "CPMG_NV1.xlsx")
    tau_nv1 = nv1["a"].to_numpy(float)
    m_exp = np.array([envelope_normalize(tau_nv1, nv1[c].to_numpy(float))[0]
                      for c in ["CPMG8", "CPMG16", "CPMG20"]], dtype=np.float32)
    from cpmg.datagen import GenConfig
    cfg_nv1 = GenConfig(n_pulses=(8, 16, 20), b_field_g=440.1, b_repr=30e3)
    builder_nv1 = TokenBuilder(tau_nv1, CENTERS, cfg_nv1, device, s_max=13)
    p = pf_curve(model_room, builder_nv1, m_exp, device)
    regions = pf_candidate_regions(p, CENTERS, thresh=args.thresh, margin_hz=3e3)
    print(f"NV1 regions (kHz): {[(round(l/1e3), round(h/1e3)) for l, h in regions]}",
          flush=True)
    fit = hybrid_greedy_fit(tau_nv1, m_exp, (8, 16, 20), 440.1, regions,
                            max_spins=12, seed=args.seed, verbose=True)
    nv1_spins = [[round(s[0] / 1e3, 1), round(s[1] / 1e3, 1)]
                 for s in fit["best"]["spins"]]
    print(f"NV1 hybrid k*={fit['best']['k']} spins (A,B kHz): {sorted(nv1_spins)}",
          flush=True)
    out = json.loads((OUT / "hybrid_results.json").read_text())
    out["nv1"] = dict(regions_khz=[[round(l / 1e3, 1), round(h / 1e3, 1)]
                                   for l, h in regions],
                      k=fit["best"]["k"], spins_khz=sorted(nv1_spins))
    (OUT / "hybrid_results.json").write_text(json.dumps(out, indent=2))
    print("saved ->", OUT / "hybrid_results.json", flush=True)


if __name__ == "__main__":
    main()
