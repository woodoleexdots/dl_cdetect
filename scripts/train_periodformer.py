"""Train PeriodFormer and evaluate on the shared ablation GT suite.

Reuses the SpinDETR scene sampler (window-free random spin sets) — signals
are stored compactly (B, 3, 700) and window tokens are built on GPU.

Usage:
  python scripts/train_periodformer.py --n-train 25000 --epochs 30
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
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpmg.ablation import build_suite, summarize
from cpmg.datagen import GenConfig
from cpmg.periodformer import PeriodFormer, TokenBuilder, soft_labels
from cpmg.setnet import sample_random_scene

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "periodformer"
SIGMAS = [0.03, 0.05, 0.08, 0.12]


def build_signal_dataset(tau, n_samples, cfg, centers_hz, seed=0,
                         a_max=60e3, b_tgt=(15e3, 50e3), min_sep=3e3,
                         label_w=1500.0):
    rng = np.random.default_rng(seed)
    X = np.empty((n_samples, len(cfg.n_pulses), len(tau)), dtype=np.float32)
    Y = np.empty((n_samples, len(centers_hz)), dtype=np.float32)
    for i in range(n_samples):
        x, tg = sample_random_scene(rng, tau, cfg, a_max=a_max, b_tgt=b_tgt,
                                    min_sep=min_sep)
        X[i] = x
        Y[i] = soft_labels(tg[:, 0] if len(tg) else [], centers_hz,
                           width_hz=label_w)
    return X, Y


def detect(p_curve, centers_hz, height=0.5, prominence=0.1):
    from scipy.signal import find_peaks

    pk, _ = find_peaks(p_curve, height=height, prominence=prominence)
    return centers_hz[pk]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=25000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-per-sigma", type=int, default=15)
    ap.add_argument("--a-max-khz", type=float, default=60.0)
    ap.add_argument("--a-step-khz", type=float, default=2.0)
    ap.add_argument("--b-tgt-min-khz", type=float, default=15.0)
    ap.add_argument("--b-tgt-max-khz", type=float, default=50.0)
    ap.add_argument("--channels", default="all", choices=["all", "16"])
    ap.add_argument("--dataset", default="NV1", choices=["NV1", "NV2"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    tau = pd.read_excel(ROOT / "dataset" / "exp_dataset" / "CPMG_NV1.xlsx")["a"].to_numpy(float)
    n_pulses = (8, 16, 20) if args.channels == "all" else (16,)
    b_tgt = (args.b_tgt_min_khz * 1e3, args.b_tgt_max_khz * 1e3)
    cfg = GenConfig(noise_range=(0.02, 0.13), n_pulses=n_pulses,
                    b_repr=0.5 * (b_tgt[0] + b_tgt[1]),
                    b_tgt_min=b_tgt[0], b_tgt_max=b_tgt[1])
    centers_hz = np.arange(-args.a_max_khz, args.a_max_khz + 1e-9,
                           args.a_step_khz) * 1e3
    a_max = args.a_max_khz * 1e3
    min_sep = max(3e3, 1.5 * args.a_step_khz * 1e3)
    label_w = 0.75 * args.a_step_khz * 1e3
    tag = args.tag or ("" if args.a_max_khz == 60 else f"_{args.dataset}_wide")

    builder = TokenBuilder(tau, centers_hz, cfg, device)

    print("generating training data ...", flush=True)
    t0 = time.time()
    kw = dict(a_max=a_max, b_tgt=b_tgt, min_sep=min_sep, label_w=label_w)
    X, Y = build_signal_dataset(tau, args.n_train, cfg, centers_hz,
                                seed=args.seed, **kw)
    Xv, Yv = build_signal_dataset(tau, args.n_val, cfg, centers_hz,
                                  seed=args.seed + 777, **kw)
    print(f"  {args.n_train} samples in {time.time()-t0:.0f}s", flush=True)

    model = PeriodFormer(in_ch=X.shape[1], n_tokens=len(centers_hz)).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"PeriodFormer params: {n_par/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    bce = nn.BCEWithLogitsLoss()

    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    Xvt = torch.from_numpy(Xv).to(device)
    Yvt = torch.from_numpy(Yv).to(device)

    best_val = float("inf")
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        tot, nb = 0.0, 0
        for i in range(0, len(perm), args.batch):
            b = perm[i : i + args.batch]
            xb = Xt[b].to(device)
            yb = Yt[b].to(device)
            opt.zero_grad()
            logits = model(builder(xb))
            loss = bce(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            vl, vb = 0.0, 0
            for i in range(0, len(Xvt), 256):
                vl += bce(model(builder(Xvt[i : i + 256])),
                          Yvt[i : i + 256]).item()
                vb += 1
        val = vl / vb
        star = ""
        if val < best_val:
            best_val = val
            torch.save(model.state_dict(), OUT / f"periodformer{tag}.pt")
            star = " *"
        print(f"ep {ep+1:3d}/{args.epochs}  train {tot/nb:.4f}  val {val:.4f}{star}",
              flush=True)

    # ---- evaluate on the shared suite ----
    model.load_state_dict(torch.load(OUT / f"periodformer{tag}.pt", map_location=device))
    model.eval()
    suite = build_suite(tau, SIGMAS, args.n_per_sigma, n_pulses=n_pulses,
                        a_lim=a_max - 10e3, b_range=b_tgt,
                        min_sep=max(6e3, 2 * min_sep))
    results = []
    with torch.no_grad():
        for ds in suite:
            x = torch.from_numpy(ds["m_recs"].astype(np.float32))[None].to(device)
            p = torch.sigmoid(model(builder(x)))[0].cpu().numpy()
            results.append(dict(sigma=ds["sigma"], gt_a=ds["gt"][:, 0],
                                detected_a=detect(p, centers_hz)))
    summ = summarize(results)
    payload = dict(config="periodformer", params_m=round(n_par / 1e6, 2),
                   per_sigma={str(k): v for k, v in summ.items()})
    (OUT / f"summary_periodformer{tag}.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)

    # ---- real NV1 inference ----
    from cpmg.represent import envelope_normalize
    if args.dataset == "NV1":
        df = pd.read_excel(ROOT / "dataset" / "exp_dataset" / "CPMG_NV1.xlsx")
        cols = ["CPMG8", "CPMG16", "CPMG20"] if args.channels == "all" else ["CPMG16"]
    else:
        df = pd.read_excel(ROOT / "dataset" / "exp_dataset" / "CPMG_NV2.xlsx")
        cols = ["CPMG16"]
    m_exp = np.array([envelope_normalize(tau, df[c].to_numpy(float))[0]
                      for c in cols], dtype=np.float32)
    with torch.no_grad():
        p = torch.sigmoid(model(builder(torch.from_numpy(m_exp)[None].to(device))))[0].cpu().numpy()
    np.savez(OUT / f"curve_{args.dataset}{tag}.npz", centers_hz=centers_hz, p=p)
    det = detect(p, centers_hz, height=0.4)
    print(f"{args.dataset} PeriodFormer curve peaks (kHz):",
          [round(x / 1e3, 1) for x in det], flush=True)


if __name__ == "__main__":
    main()
