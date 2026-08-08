"""Per-window HPC classifier training + inference on experimental data.

For each A window: generate a fresh synthetic dataset (target/side/mid/far),
train HPCNet, validate, then classify the experimental NV1 slice-stack image
at that window's target period. The spin-existence score is 1 - P(class 0).

Usage:
  python scripts/train_hpc.py                      # full grid -60..60 kHz
  python scripts/train_hpc.py --windows -13 2      # quick check
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpmg.datagen import GenConfig, gen_window_dataset
from cpmg.models import HPCNet
from cpmg.physics import target_period
from cpmg.represent import envelope_normalize, slice_stack

DATA_DIR = Path(__file__).resolve().parents[1] / "dataset" / "exp_dataset"


def load_experimental_mrec():
    nv1 = pd.read_excel(DATA_DIR / "CPMG_NV1.xlsx")
    tau = nv1["a"].to_numpy(float)
    m_recs = []
    for col in ["CPMG8", "CPMG16", "CPMG20"]:
        m, _ = envelope_normalize(tau, nv1[col].to_numpy(float))
        m_recs.append(m)
    return tau, np.array(m_recs)


def exp_image(tau, m_recs, a_center_hz, cfg: GenConfig):
    tp = target_period(a_center_hz, cfg.b_repr, cfg.b_field_g)
    imgs = [
        np.nan_to_num(
            slice_stack(tau, m, tp, width=cfg.image_width, tau_start=tau[0]), nan=1.0
        )
        for m in m_recs
    ]
    return np.stack(imgs).astype(np.float32)


def train_one_window(X, y, device, epochs=20, batch=256, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    n = len(X)
    idx = torch.randperm(n)
    n_val = int(0.2 * n)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y).to(device)

    model = HPCNet(in_ch=X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    best_acc, best_state = 0.0, None
    for ep in range(epochs):
        model.train()
        perm = tr_idx[torch.randperm(len(tr_idx))]
        for i in range(0, len(perm), batch):
            b = perm[i : i + batch]
            opt.zero_grad()
            loss = crit(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xt[val_idx]).argmax(1)
            acc = (pred == yt[val_idx]).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-min", type=float, default=-60.0)
    ap.add_argument("--a-max", type=float, default=60.0)
    ap.add_argument("--a-step", type=float, default=2.0)
    ap.add_argument("--windows", type=float, nargs="*", default=None)
    ap.add_argument("--n-per-class", type=int, default=700)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parents[1] / "results" / "hpc"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = GenConfig()

    tau, m_recs = load_experimental_mrec()
    if args.windows is not None:
        centers_khz = np.array(args.windows, dtype=float)
    else:
        centers_khz = np.arange(args.a_min, args.a_max + 1e-9, args.a_step)

    rows = []
    tic = time.time()
    for i, a_khz in enumerate(centers_khz):
        t0 = time.time()
        X, y = gen_window_dataset(tau, a_khz * 1e3, args.n_per_class, cfg,
                                  seed=args.seed + 1000 + i)
        model, val_acc = train_one_window(X, y, device, epochs=args.epochs,
                                          seed=args.seed + i)
        xe = torch.from_numpy(exp_image(tau, m_recs, a_khz * 1e3, cfg)[None]).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(xe), dim=1)[0].cpu().numpy()
        p_spin = 1.0 - probs[0]
        torch.save(model.state_dict(), outdir / f"hpc_A{a_khz:+08.1f}kHz.pt")
        rows.append(dict(a_khz=a_khz, val_acc=val_acc, p0=probs[0], p1=probs[1],
                         p2=probs[2], p_spin=p_spin))
        print(f"[{i+1}/{len(centers_khz)}] A={a_khz:+6.1f} kHz  val_acc={val_acc:.3f}  "
              f"P(spin)={p_spin:.3f}  (p1={probs[1]:.2f}, p2={probs[2]:.2f})  "
              f"{time.time()-t0:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "hpc_scan_NV1.csv", index=False)
    print(f"total {time.time()-tic:.0f}s -> {outdir/'hpc_scan_NV1.csv'}")


if __name__ == "__main__":
    main()
