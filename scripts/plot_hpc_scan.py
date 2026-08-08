"""Plot the HPC classifier scan against the simple period-scan contrast.

Reads results/hpc/hpc_scan_NV1.csv (+ results/figs/period_scan.npz) and
writes results/figs/09_hpc_scan.png with detected A-candidates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]


def main():
    df = pd.read_csv(ROOT / "results" / "hpc" / "hpc_scan_NV1.csv")
    scan = np.load(ROOT / "results" / "figs" / "period_scan.npz")

    a = df["a_khz"].to_numpy()
    p_spin = df["p_spin"].to_numpy()
    val_acc = df["val_acc"].to_numpy()

    pk, props = find_peaks(p_spin, height=0.6, prominence=0.1)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    ax = axes[0]
    ax.plot(a, p_spin, "o-", lw=1.2, ms=3.5, color="tab:blue")
    ax.axhline(0.5, color="gray", ls=":", lw=0.8)
    for x in a[pk]:
        ax.axvline(x, color="r", alpha=0.35, lw=1)
        ax.annotate(f"{x:+.0f}", (x, 1.02), ha="center", fontsize=8, color="r")
    ax.set_ylabel("P(spin) = 1 - P(class0)")
    ax.set_ylim(0, 1.1)
    ax.set_title("HPC classifier spin-existence score (NV1, N=8/16/20 joint)")

    ax = axes[1]
    ax.plot(a, val_acc, "s-", lw=1, ms=3, color="tab:green")
    ax.axhline(1 / 3, color="gray", ls=":", lw=0.8, label="chance (3-class)")
    ax.set_ylabel("val accuracy")
    ax.set_ylim(0.3, 1.0)
    ax.legend(fontsize=8)
    ax.set_title("Per-window validation accuracy (trustworthiness of the score above)")

    ax = axes[2]
    ag = scan["a_grid"] / 1e3
    for key, style in [("NV1_N8", "-"), ("NV1_N16", "-"), ("NV1_N20", "-")]:
        c = scan[key]
        ax.plot(ag, (c - c.min()) / (c.max() - c.min()), style, lw=0.9,
                alpha=0.8, label=key.replace("NV1_", "N="))
    ax.set_ylabel("period-scan contrast (norm.)")
    ax.set_xlabel("A (kHz)")
    ax.legend(fontsize=8)
    ax.set_title("Reference: simple period-scan contrast (previous step)")

    fig.tight_layout()
    out = ROOT / "results" / "figs" / "09_hpc_scan.png"
    fig.savefig(out, dpi=150)
    print("candidates (kHz):", [f"{x:+.0f}" for x in a[pk]])
    print(f"saved {out}")


if __name__ == "__main__":
    main()
