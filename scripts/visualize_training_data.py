"""Visualize generated HPC training samples.

Figures (written to results/figs):
  06_train_signals.png      raw synthetic Px for one class-1 sample (3 channels)
  07_train_examples.png     slice-stack images: rows = class 0/1/2, cols = N 8/16/20
  08_train_mean_images.png  per-class mean image over many samples
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpmg.datagen import GenConfig, make_sample
from cpmg.physics import target_period

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_training_data import experimental_tau

US = 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-center", type=float, default=-13.0, help="window center (kHz)")
    ap.add_argument("--n-mean", type=int, default=200, help="samples for mean images")
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parents[1] / "results" / "figs"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = GenConfig()
    tau = experimental_tau()
    a_c = args.a_center * 1e3
    tp = target_period(a_c, cfg.b_repr, cfg.b_field_g)
    rng = np.random.default_rng(7)

    # ---------------- 06: raw signals of one class-1 sample ----------------
    img, _, info = make_sample(rng, tau, a_c, 1, cfg, return_signals=True)
    tgt = info["targets"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)
    for ax, n_pulse, px in zip(axes, cfg.n_pulses, info["signals"]):
        ax.plot(tau / US, px, lw=0.6)
        # mark expected dip positions of the target spin
        for k in range(1, 30):
            t_dip = (2 * k - 1) * tp / 2
            if t_dip > tau[-1]:
                break
            ax.axvline(t_dip / US, color="r", alpha=0.25, lw=0.8)
        ax.set_ylabel("Px")
        ax.set_title(f"N={n_pulse}", fontsize=9)
    axes[-1].set_xlabel("tau (us)")
    fig.suptitle(
        f"Synthetic class-1 sample  |  window A={args.a_center:.0f} kHz  |  "
        f"target (A,B)=({tgt[0,0]/1e3:.1f}, {tgt[0,1]/1e3:.1f}) kHz, "
        f"{len(info['others'])} interferer+bath spins  (red: target dip positions)"
    )
    fig.tight_layout()
    fig.savefig(outdir / "06_train_signals.png", dpi=150)
    plt.close(fig)

    # ---------------- 07: example images per class ----------------
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    for row, cls in enumerate([0, 1, 2]):
        img, _, info = make_sample(rng, tau, a_c, cls, cfg, return_signals=True)
        for col in range(3):
            ax = axes[row, col]
            im = ax.imshow(img[col], aspect="auto", origin="lower", cmap="viridis",
                           extent=[0, tp / US, 0, img.shape[1]], vmin=-1.0, vmax=1.2)
            if row == 0:
                ax.set_title(f"N={cfg.n_pulses[col]}", fontsize=10)
            if col == 0:
                tstr = ", ".join(f"({a/1e3:.1f},{b/1e3:.0f})" for a, b in info["targets"]) or "none"
                ax.set_ylabel(f"class {cls}\ntargets: {tstr}", fontsize=8)
            ax.set_xlabel("tau mod TP (us)", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.7, label="M (env-normalized)")
    fig.suptitle(f"Training examples, window A={args.a_center:.0f} kHz "
                 f"(TP={tp/US:.4f} us) — class-1/2 targets align vertically")
    fig.savefig(outdir / "07_train_examples.png", dpi=150)
    plt.close(fig)

    # ---------------- 08: per-class mean images ----------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for col, cls in enumerate([0, 1, 2]):
        acc = []
        for i in range(args.n_mean):
            img, _ = make_sample(rng, tau, a_c, cls, cfg)
            acc.append(img[1])  # N=16 channel
        mean_img = np.mean(acc, axis=0)
        ax = axes[col]
        im = ax.imshow(mean_img, aspect="auto", origin="lower", cmap="viridis",
                       extent=[0, tp / US, 0, mean_img.shape[0]])
        ax.set_title(f"class {cls} mean (n={args.n_mean}, N=16)", fontsize=10)
        ax.set_xlabel("tau mod TP (us)")
    axes[0].set_ylabel("slice #")
    fig.colorbar(im, ax=axes, shrink=0.8, label="mean M")
    fig.suptitle(f"Class-mean slice-stack images, window A={args.a_center:.0f} kHz — "
                 "the central vertical line is the learnable signature")
    fig.savefig(outdir / "08_train_mean_images.png", dpi=150)
    plt.close(fig)

    print(f"figures written to {outdir}")


if __name__ == "__main__":
    main()
