"""Figures for the PeriodFormer benchmark-v2 run.

  15_pf_training_curves.png : train/val loss curves of pf_cryo & pf_room
  16_pf_gt_vs_detected.png  : per-arm P(spin) curves, GT positions, detections
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "benchmark_v2"
FIGS = ROOT / "results" / "figs"


def main():
    res = json.loads((OUT / "pf_results.json").read_text())
    gt = res["gt"]
    gt_a = np.array([g["a_khz"] for g in gt])
    gt_b = np.array([g["b_khz"] for g in gt])

    # ---- training curves ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, name in zip(axes, ["pf_cryo", "pf_room"]):
        c = res["curves"][name]
        ax.plot(np.arange(1, len(c["train"]) + 1), c["train"], "o-", ms=3,
                label="train BCE")
        ax.plot(np.arange(1, len(c["val"]) + 1), c["val"], "s-", ms=3,
                label="val BCE (held-out 2000 scenes)")
        ax.set_xlabel("epoch")
        ax.set_ylabel("BCE loss")
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("PeriodFormer training/validation curves (synthetic scenes; "
                 "real 50-spin bath never seen in training)")
    fig.tight_layout()
    fig.savefig(FIGS / "15_pf_training_curves.png", dpi=150)
    plt.close(fig)

    # ---- GT vs detected ----
    arms = ["A_cryo", "B_room", "C_mismatch"]
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    for ax, arm in zip(axes, arms):
        d = np.load(OUT / f"pf_curves_{arm}.npz")
        centers = d["centers"] / 1e3
        p = d["p"]
        for row in p:
            ax.plot(centers, row, lw=0.7, color="tab:blue", alpha=0.35)
        ax.plot(centers, p.mean(0), lw=1.8, color="tab:blue",
                label="P(spin), mean over realizations")
        # GT sticks: height ~ coupling strength (A_perp)
        for a, b in zip(gt_a, gt_b):
            ax.axvline(a, color="tab:red", alpha=0.55, lw=1.0)
            ax.annotate(f"{b:.0f}", (a, 1.03), fontsize=6, color="tab:red",
                        ha="center", annotation_clip=False)
        ax.axhline(0.5, color="gray", ls=":", lw=0.8)
        m = res["metrics"][arm]
        ax.set_title(f"{arm}   P={m['precision']:.2f} R={m['recall']:.2f} "
                     f"F1={m['f1']:.2f}", fontsize=10, loc="left")
        ax.set_ylabel("P(spin)")
        ax.set_ylim(0, 1.1)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("A (kHz)   [red lines: 27 GT spins, labels = A_perp kHz]")
    fig.suptitle("PeriodFormer on the published 50-spin bath: GT vs inferred spins")
    fig.tight_layout()
    fig.savefig(FIGS / "16_pf_gt_vs_detected.png", dpi=150)
    plt.close(fig)
    print("saved 15_pf_training_curves.png / 16_pf_gt_vs_detected.png")


if __name__ == "__main__":
    main()
