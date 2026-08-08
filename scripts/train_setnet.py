"""Train SpinDETR and evaluate it on the shared ablation GT suite.

Usage:
  python scripts/train_setnet.py --n-train 50000 --epochs 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpmg.ablation import build_suite, match_score, summarize
from cpmg.datagen import GenConfig
from cpmg.setnet import (SpinDETR, build_dataset, hungarian_loss,
                         predict_spins)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "setnet"
SIGMAS = [0.03, 0.05, 0.08, 0.12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=50000)
    ap.add_argument("--n-val", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-per-sigma", type=int, default=15)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    tau = pd.read_excel(ROOT / "dataset" / "exp_dataset" / "CPMG_NV1.xlsx")["a"].to_numpy(float)
    cfg = GenConfig(noise_range=(0.02, 0.13))

    print("generating training data ...", flush=True)
    t0 = time.time()
    X, T, n_gt = build_dataset(tau, args.n_train, cfg, seed=args.seed)
    Xv, Tv, n_gtv = build_dataset(tau, args.n_val, cfg, seed=args.seed + 777)
    print(f"  {args.n_train} samples in {time.time()-t0:.0f}s", flush=True)

    model = SpinDETR(in_ch=X.shape[1]).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"SpinDETR params: {n_par/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    Xt = torch.from_numpy(X)
    Tt = torch.from_numpy(T)
    nt = torch.from_numpy(n_gt)
    Xvt = torch.from_numpy(Xv).to(device)
    Tvt = torch.from_numpy(Tv).to(device)

    best_val = float("inf")
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        tot, nb = 0.0, 0
        for i in range(0, len(perm), args.batch):
            b = perm[i : i + args.batch]
            xb = Xt[b].to(device)
            tb = Tt[b].to(device)
            nb_gt = nt[b]
            opt.zero_grad()
            exist, a, bb = model(xb)
            loss = hungarian_loss(exist, a, bb, tb, nb_gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            vl, vb = 0.0, 0
            for i in range(0, len(Xvt), 512):
                exist, a, bb = model(Xvt[i : i + 512])
                vl += hungarian_loss(exist, a, bb, Tvt[i : i + 512],
                                     n_gtv[i : i + 512]).item()
                vb += 1
        val = vl / vb
        star = ""
        if val < best_val:
            best_val = val
            torch.save(model.state_dict(), OUT / "spindetr.pt")
            star = " *"
        print(f"ep {ep+1:3d}/{args.epochs}  train {tot/nb:.4f}  val {val:.4f}{star}",
              flush=True)

    # ---- evaluate on the shared suite ----
    model.load_state_dict(torch.load(OUT / "spindetr.pt", map_location=device))
    model.eval()
    suite = build_suite(tau, SIGMAS, args.n_per_sigma)
    results = []
    for ds in suite:
        pred = predict_spins(model, ds["m_recs"], device)
        det = np.array([p[0] for p in pred])
        results.append(dict(sigma=ds["sigma"], gt_a=ds["gt"][:, 0],
                            detected_a=det))
    summ = summarize(results)
    payload = dict(config="spindetr", params_m=round(n_par / 1e6, 2),
                   per_sigma={str(k): v for k, v in summ.items()})
    (OUT / "summary_spindetr.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)

    # ---- real NV1 inference ----
    from cpmg.represent import envelope_normalize
    nv1 = pd.read_excel(ROOT / "dataset" / "exp_dataset" / "CPMG_NV1.xlsx")
    m_exp = np.array([envelope_normalize(tau, nv1[c].to_numpy(float))[0]
                      for c in ["CPMG8", "CPMG16", "CPMG20"]], dtype=np.float32)
    pred = predict_spins(model, m_exp, device, thresh=0.3)
    print("NV1 predicted spins (A kHz, B kHz, p):",
          [(round(a / 1e3, 1), round(b / 1e3, 1), round(p, 2)) for a, b, p in pred],
          flush=True)


if __name__ == "__main__":
    main()
