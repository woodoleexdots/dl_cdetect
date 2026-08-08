"""PF->DE hybrid on the real NV2 data (strong-coupling regime).

Stage 1: periodformer_NV2_strong checkpoint (161 tokens, A in +-400 kHz,
step 5 kHz, 1 channel CPMG-16, b_repr 185 kHz).
Stage 2: region-constrained greedy DE with B in [40, 350] kHz.
Appends the result to results/benchmark_v2/hybrid_results.json under "nv2".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpmg.datagen import GenConfig
from cpmg.hybrid import hybrid_greedy_fit, pf_candidate_regions
from cpmg.periodformer import PeriodFormer, TokenBuilder
from cpmg.represent import envelope_normalize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "benchmark_v2"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nv2 = pd.read_excel(ROOT / "dataset" / "exp_dataset" / "CPMG_NV2.xlsx")
    tau = nv2["Time"].to_numpy(float)
    m_exp = np.array([envelope_normalize(tau, nv2["CPMG16"].to_numpy(float))[0]],
                     dtype=np.float32)

    centers = np.arange(-400, 400 + 1e-9, 5.0) * 1e3
    cfg = GenConfig(n_pulses=(16,), b_field_g=440.1, b_repr=185e3,
                    b_tgt_min=50e3, b_tgt_max=320e3)
    builder = TokenBuilder(tau, centers, cfg, device, s_max=13)
    model = PeriodFormer(in_ch=1, n_tokens=len(centers), s_max=13).to(device)
    model.load_state_dict(torch.load(
        ROOT / "results" / "periodformer" / "periodformer_NV2_strong.pt",
        map_location=device))
    model.eval()

    with torch.no_grad():
        p = torch.sigmoid(model(builder(
            torch.from_numpy(m_exp)[None].to(device))))[0].cpu().numpy()
    regions = pf_candidate_regions(p, centers, thresh=0.2, margin_hz=8e3)
    print(f"NV2 regions (kHz): {[(round(l/1e3), round(h/1e3)) for l, h in regions]}",
          flush=True)

    fit = hybrid_greedy_fit(tau, m_exp, (16,), 440.1, regions,
                            max_spins=8, b_bounds=(40e3, 350e3),
                            de_maxiter=60, de_popsize=20, seed=0, verbose=True)
    spins = sorted([[round(s[0] / 1e3, 1), round(s[1] / 1e3, 1)]
                    for s in fit["best"]["spins"]])
    print(f"NV2 hybrid k*={fit['best']['k']} spins (A,B kHz): {spins}", flush=True)

    f = OUT / "hybrid_results.json"
    out = json.loads(f.read_text()) if f.exists() else {}
    out["nv2"] = dict(regions_khz=[[round(l / 1e3, 1), round(h / 1e3, 1)]
                                   for l, h in regions],
                      k=fit["best"]["k"], spins_khz=spins)
    f.write_text(json.dumps(out, indent=2))
    print("saved ->", f, flush=True)


if __name__ == "__main__":
    main()
