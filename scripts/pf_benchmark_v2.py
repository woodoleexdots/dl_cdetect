"""PeriodFormer on the benchmark-v2 arms (published Delft 50-spin bath).

Two arm-matched models are trained from scratch (checkpoints cached):

  pf_cryo : 1ch N=32,     tau = 4 ns x 7000 (28 us), B = 403.553 G,
            tokens 61 x (1, 24, 53)  [s_max = 24 slices]
  pf_room : 3ch N=8/16/20, tau = 20 ns x 700 (14 us), B = 440.1 G,
            tokens 61 x (3, 13, 53) -- B_perp target range widened to
            5-60 kHz to match the GT box (the earlier model used 15-50)

Training data: window-free random scenes (0-6 target spins, |A|<=60 kHz,
A_perp 5-60 kHz, min sep 3 kHz + Poisson(15) weak bath spins), envelope /
contrast / noise randomized per channel; Gaussian soft labels (1.5 kHz) on
the 61-token grid; val = 2000 held-out scenes (different seed).
The real 50-spin bath is NEVER seen in training -- test only.

Evaluation: the exact benchmark-v2 suites (same seeds) for arms A/B/C.
Arm C tokens are built at the WRONG analysis field (+1.5 G), same as the
other methods. Outputs: metrics, training curves, per-realization detected
spins vs GT.
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
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpmg.datagen import GenConfig
from cpmg.periodformer import PeriodFormer, TokenBuilder, soft_labels
from cpmg.setnet import sample_random_scene

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "benchmark_v2"

# import benchmark_v2 for ARMS / make_arm_dataset / gt / scoring
spec = importlib.util.spec_from_file_location("bv2", ROOT / "scripts" / "benchmark_v2.py")
bv2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bv2)

CENTERS = np.arange(-60, 60 + 1e-9, 2.0) * 1e3


def make_gen_cfg(arm_key, analysis_field):
    a = bv2.ARMS[arm_key]
    return GenConfig(n_pulses=a["n_pulses"], b_field_g=analysis_field,
                     b_tgt_min=5e3, b_tgt_max=60e3, b_repr=30e3,
                     noise_range=(max(0.01, a["sigma"] - 0.03), a["sigma"] + 0.06),
                     t2_range_us=a["t2_us"], stretch_range=a["stretch"],
                     contrast_range=a["contrast"])


def build_signals(tau, n, cfg, seed):
    rng = np.random.default_rng(seed)
    X = np.empty((n, len(cfg.n_pulses), len(tau)), dtype=np.float32)
    Y = np.empty((n, len(CENTERS)), dtype=np.float32)
    for i in range(n):
        x, tg = sample_random_scene(rng, tau, cfg, a_max=60e3,
                                    b_tgt=(5e3, 60e3), min_sep=3e3)
        X[i] = x
        Y[i] = soft_labels(tg[:, 0] if len(tg) else [], CENTERS)
    return X, Y


def train_pf(name, tau, cfg, s_max, device, n_train, epochs, seed, batch=128):
    ckpt = OUT / f"{name}.pt"
    curve_f = OUT / f"{name}_curve.json"
    builder = TokenBuilder(tau, CENTERS, cfg, device, s_max=s_max)
    model = PeriodFormer(in_ch=len(cfg.n_pulses), n_tokens=len(CENTERS),
                         s_max=s_max).to(device)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        print(f"[{name}] cached", flush=True)
        return model, builder, json.loads(curve_f.read_text())

    print(f"[{name}] generating {n_train} scenes ...", flush=True)
    t0 = time.time()
    X, Y = build_signals(tau, n_train, cfg, seed)
    Xv, Yv = build_signals(tau, 2000, cfg, seed + 777)
    print(f"  gen {time.time()-t0:.0f}s", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bce = nn.BCEWithLogitsLoss()
    Xt, Yt = torch.from_numpy(X), torch.from_numpy(Y)
    Xvt = torch.from_numpy(Xv).to(device)
    Yvt = torch.from_numpy(Yv).to(device)
    curve = {"train": [], "val": []}
    best = float("inf")
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        tot, nb = 0.0, 0
        for i in range(0, len(perm), batch):
            b = perm[i : i + batch]
            opt.zero_grad()
            loss = bce(model(builder(Xt[b].to(device))), Yt[b].to(device))
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
                vl += bce(model(builder(Xvt[i : i + 256])), Yvt[i : i + 256]).item()
                vb += 1
        val = vl / vb
        curve["train"].append(round(tot / nb, 5))
        curve["val"].append(round(val, 5))
        if val < best:
            best = val
            torch.save(model.state_dict(), ckpt)
        print(f"[{name}] ep {ep+1}/{epochs} train {tot/nb:.4f} val {val:.4f}",
              flush=True)
    curve_f.write_text(json.dumps(curve))
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model, builder, curve


def detect(p, height=0.5, prominence=0.1):
    from scipy.signal import find_peaks

    pk, _ = find_peaks(p, height=height, prominence=prominence)
    return CENTERS[pk], p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=8)
    ap.add_argument("--n-real-cryo", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    bath, labels = bv2.load_bath()
    box = (np.abs(bath[:, 0]) <= 60e3) & (bath[:, 1] >= 5e3)
    gt = bath[box]
    gt_labels = [labels[i] for i in np.where(box)[0]]
    gt_a = gt[:, 0]
    print(f"GT: {len(gt)} in-box spins", flush=True)

    # ---- train the two arm-matched models ----
    a_cryo = bv2.ARMS["A_cryo"]
    tau_cryo = np.arange(1, a_cryo["n_tau"] + 1) * a_cryo["dt"]
    cfg_cryo = make_gen_cfg("A_cryo", a_cryo["b_field"])
    from cpmg.physics import target_period
    s_max_cryo = int(np.floor(tau_cryo[-1] / target_period(0, 30e3, a_cryo["b_field"])))
    model_cryo, builder_cryo, curve_cryo = train_pf(
        "pf_cryo", tau_cryo, cfg_cryo, s_max_cryo, device,
        n_train=15000, epochs=25, seed=args.seed)

    a_room = bv2.ARMS["B_room"]
    tau_room = np.arange(1, a_room["n_tau"] + 1) * a_room["dt"]
    cfg_room = make_gen_cfg("B_room", a_room["b_field"])
    model_room, builder_room, curve_room = train_pf(
        "pf_room", tau_room, cfg_room, 13, device,
        n_train=25000, epochs=25, seed=args.seed + 1)

    # arm C: same trained room model, but tokens built at the WRONG field
    cfg_mis = make_gen_cfg("C_mismatch", a_room["b_field"] + bv2.DB_MISMATCH)
    builder_mis = TokenBuilder(tau_room, CENTERS, cfg_mis, device, s_max=13)

    # ---- evaluate on the exact benchmark-v2 suites ----
    plans = [
        ("A_cryo", model_cryo, builder_cryo, tau_cryo, args.n_real_cryo, 2e3),
        ("B_room", model_room, builder_room, tau_room, args.n_real, 4e3),
        ("C_mismatch", model_room, builder_mis, tau_room, args.n_real, 4e3),
    ]
    pf_results = {}
    detail = {}
    for arm, model, builder, tau, n_real, tol in plans:
        rows, det_lists, curves = [], [], []
        for i in range(n_real):
            _, m_recs = bv2.make_arm_dataset(arm, bath, args.seed + 100 + i)
            x = torch.from_numpy(m_recs)[None].to(device)
            with torch.no_grad():
                p = torch.sigmoid(model(builder(x)))[0].cpu().numpy()
            det, _ = detect(p)
            rows.append(bv2.match_score(det, gt_a, tol))
            det_lists.append(sorted(round(v / 1e3, 1) for v in det))
            curves.append(p)
        m = bv2.agg(rows, len(gt_a))
        pf_results[arm] = m
        detail[arm] = det_lists
        np.savez(OUT / f"pf_curves_{arm}.npz", centers=CENTERS,
                 p=np.stack(curves))
        print(f"[pf] {arm}: {json.dumps(m)}", flush=True)
        print(f"     detections r0: {det_lists[0]}", flush=True)

    out = dict(metrics=pf_results, detections=detail,
               gt=[dict(label=l, a_khz=round(a / 1e3, 1), b_khz=round(b / 1e3, 1))
                   for l, (a, b) in zip(gt_labels, gt)],
               curves=dict(pf_cryo=curve_cryo, pf_room=curve_room))
    (OUT / "pf_results.json").write_text(json.dumps(out, indent=2))
    print("saved ->", OUT / "pf_results.json", flush=True)


if __name__ == "__main__":
    main()
