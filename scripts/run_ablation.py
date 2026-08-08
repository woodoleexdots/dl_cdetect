"""Ablation study orchestration.

Configs (paper baseline -> our proposal):
  A1 mlp2021-1ch        2021 arch, single N=16, no denoiser
  A2 mlp2021-1ch-ae     2021 arch + 2021 conv-AE denoiser  (= faithful 2021 pipeline)
  A3 mlp2021-3ch        2021 arch, joint channels          (channel ablation)
  B1 cnn-1ch            our CNN, single N=16               (arch ablation)
  B2 cnn-3ch            our CNN, joint channels
  B3 cnn-3ch-naf        our CNN + NAFNet1D denoiser        (= proposed)
  DE de-greedy          classical DE+BIC baseline

Evaluated on a shared GT suite at sigma = 0.03 / 0.05 / 0.08 / 0.12
(sigma ~0.02-0.03 = cryogenic-like SNR, ~0.05-0.08 = our room-temp data).

Results cached per config in results/ablation/. Safe to re-run.
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

from cpmg.ablation import (AblationConfig, build_suite, eval_bank,
                           match_score, summarize, train_bank, train_denoiser)
from cpmg.datagen import GenConfig
from cpmg.defit import greedy_de_fit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "ablation"

CONFIGS = [
    AblationConfig("mlp2021-1ch", arch="mlp2021", n_pulses=(16,), denoise=None),
    AblationConfig("mlp2021-1ch-ae", arch="mlp2021", n_pulses=(16,), denoise="ae2021"),
    AblationConfig("mlp2021-3ch", arch="mlp2021", n_pulses=(8, 16, 20), denoise=None),
    AblationConfig("cnn-1ch", arch="cnn", n_pulses=(16,), denoise=None),
    AblationConfig("cnn-3ch", arch="cnn", n_pulses=(8, 16, 20), denoise=None),
    AblationConfig("cnn-3ch-naf", arch="cnn", n_pulses=(8, 16, 20), denoise="nafnet"),
]

SIGMAS = [0.03, 0.05, 0.08, 0.12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-sigma", type=int, default=15)
    ap.add_argument("--n-per-class", type=int, default=700)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--a-step", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    tau = pd.read_excel(ROOT / "dataset" / "exp_dataset" / "CPMG_NV1.xlsx")["a"].to_numpy(float)
    centers_hz = np.arange(-60, 60 + 1e-9, args.a_step) * 1e3

    print(f"device={device}  windows={len(centers_hz)}  "
          f"suite={len(SIGMAS)}x{args.n_per_sigma}", flush=True)
    suite = build_suite(tau, SIGMAS, args.n_per_sigma)

    # ---- denoisers (shared across configs) ----
    denoisers = {}
    for kind in {c.denoise for c in CONFIGS if c.denoise}:
        ckpt = OUT / f"denoiser_{kind}.pt"
        gen_cfg = GenConfig(noise_range=(0.02, 0.13))
        if ckpt.exists():
            from cpmg.models import DenoiseAE2021, NAFNet1D
            model = (DenoiseAE2021() if kind == "ae2021" else NAFNet1D()).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()
            print(f"loaded cached denoiser {kind}", flush=True)
        else:
            print(f"training denoiser {kind} ...", flush=True)
            model = train_denoiser(kind, tau, gen_cfg, device, seed=args.seed)
            torch.save(model.state_dict(), ckpt)
        denoisers[kind] = model

    # ---- NN configs ----
    all_summaries = {}
    for cfg in CONFIGS:
        res_file = OUT / f"summary_{cfg.name}.json"
        if res_file.exists():
            all_summaries[cfg.name] = json.loads(res_file.read_text())
            print(f"[{cfg.name}] cached", flush=True)
            continue
        t0 = time.time()
        bankdir = OUT / f"bank_{cfg.name}"
        denoiser = denoisers.get(cfg.denoise)
        marker = bankdir / "DONE"
        if not marker.exists():
            print(f"[{cfg.name}] training bank ...", flush=True)
            accs = train_bank(cfg, tau, centers_hz, bankdir, device,
                              denoiser=denoiser, n_per_class=args.n_per_class,
                              epochs=args.epochs, seed=args.seed)
            marker.write_text(json.dumps({"val_accs": accs}))
        print(f"[{cfg.name}] evaluating ...", flush=True)
        results = eval_bank(cfg, tau, centers_hz, bankdir, device, suite,
                            denoiser=denoiser)
        summ = summarize(results)
        val_accs = json.loads(marker.read_text())["val_accs"]
        payload = dict(config=cfg.name, arch=cfg.arch,
                       n_pulses=list(cfg.n_pulses), denoise=cfg.denoise,
                       mean_val_acc=round(float(np.mean(val_accs)), 3),
                       per_sigma={str(k): v for k, v in summ.items()},
                       train_plus_eval_s=round(time.time() - t0, 1))
        res_file.write_text(json.dumps(payload, indent=2))
        np.savez(OUT / f"curves_{cfg.name}.npz",
                 sigmas=[r["sigma"] for r in results],
                 p_curves=np.stack([r["p_curve"] for r in results]),
                 centers_hz=centers_hz)
        all_summaries[cfg.name] = payload
        print(f"[{cfg.name}] done ({time.time()-t0:.0f}s): "
              f"{json.dumps(payload['per_sigma'])}", flush=True)

    # ---- DE baseline ----
    res_file = OUT / "summary_de.json"
    if res_file.exists():
        all_summaries["de"] = json.loads(res_file.read_text())
        print("[de] cached", flush=True)
    else:
        print("[de] running greedy DE on suite ...", flush=True)
        t0 = time.time()
        de_results = []
        for k, ds in enumerate(suite):
            fit = greedy_de_fit(tau, ds["m_recs"], (8, 16, 20),
                                GenConfig().b_field_g, max_spins=7,
                                seed=args.seed + k, verbose=False)
            det = np.array([s[0] for s in fit["best"]["spins"]])
            de_results.append(dict(sigma=ds["sigma"], gt_a=ds["gt"][:, 0],
                                   detected_a=det))
            if (k + 1) % 10 == 0:
                print(f"  de {k+1}/{len(suite)}", flush=True)
        summ = summarize(de_results)
        payload = dict(config="de-greedy",
                       per_sigma={str(k): v for k, v in summ.items()},
                       train_plus_eval_s=round(time.time() - t0, 1))
        res_file.write_text(json.dumps(payload, indent=2))
        all_summaries["de"] = payload
        print(f"[de] done: {json.dumps(payload['per_sigma'])}", flush=True)

    (OUT / "all_summaries.json").write_text(json.dumps(all_summaries, indent=2))
    print("\nALL DONE ->", OUT / "all_summaries.json", flush=True)


if __name__ == "__main__":
    main()
