"""Ablation result figures.

  10_ablation_f1.png    F1 / precision / recall vs sigma, per config
  11_ablation_bars.png  room-temp operating point (sigma=0.08) bar chart
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
OUT = ROOT / "results" / "ablation"
FIGS = ROOT / "results" / "figs"

LABELS = {
    "mlp2021-1ch": ("2021 MLP, single-N", "tab:gray", "--"),
    "mlp2021-1ch-ae": ("2021 full pipeline (MLP+AE)", "tab:brown", "--"),
    "mlp2021-3ch": ("2021 MLP + joint-N (abl.)", "tab:olive", "--"),
    "cnn-1ch": ("CNN, single-N (abl.)", "tab:cyan", "-"),
    "cnn-3ch": ("CNN + joint-N (ours)", "tab:blue", "-"),
    "cnn-3ch-naf": ("CNN + joint-N + NAFNet", "tab:pink", "-"),
    "spindetr": ("SpinDETR (raw signal, set pred.)", "tab:purple", "-"),
    "periodformer": ("PeriodFormer (window tokens + attn)", "tab:red", "-"),
    "de": ("classical DE+BIC", "black", ":"),
}


def main():
    summaries = json.loads((OUT / "all_summaries.json").read_text())
    metrics = ["f1", "precision", "recall"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
    for key, payload in summaries.items():
        name = payload.get("config", key)
        label, color, ls = LABELS.get(key, (name, None, "-"))
        per_sigma = payload["per_sigma"]
        sig = sorted(float(s) for s in per_sigma)
        for ax, m in zip(axes, metrics):
            vals = [per_sigma[str(s)][m] for s in sig]
            ax.plot(sig, vals, marker="o", label=label, color=color, ls=ls, lw=1.6)
    for ax, m in zip(axes, metrics):
        ax.set_xlabel("noise sigma")
        ax.set_ylabel(m)
        ax.set_ylim(0, 1.05)
        ax.axvspan(0.02, 0.035, color="tab:blue", alpha=0.06)
        ax.axvspan(0.05, 0.085, color="tab:red", alpha=0.06)
        ax.grid(alpha=0.3)
    axes[0].annotate("cryo-like", (0.031, 0.04), fontsize=8, color="tab:blue")
    axes[0].annotate("room-temp", (0.06, 0.04), fontsize=8, color="tab:red")
    axes[0].legend(fontsize=7.5, loc="lower left")
    fig.suptitle("Ablation: detection performance vs noise level "
                 "(shared GT suite, 15 datasets/sigma, tol=4 kHz)")
    fig.tight_layout()
    fig.savefig(FIGS / "10_ablation_f1.png", dpi=150)
    plt.close(fig)

    # bar chart at the room-temperature operating point
    sigma_op = "0.08"
    names, f1s, precs, recs = [], [], [], []
    for key, payload in summaries.items():
        label = LABELS.get(key, (payload.get("config", key),))[0]
        row = payload["per_sigma"].get(sigma_op)
        if row is None:
            continue
        names.append(label)
        f1s.append(row["f1"])
        precs.append(row["precision"])
        recs.append(row["recall"])
    order = np.argsort(f1s)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(x - 0.2, np.array(f1s)[order], 0.2, label="F1", color="tab:red")
    ax.barh(x, np.array(precs)[order], 0.2, label="precision", color="tab:blue")
    ax.barh(x + 0.2, np.array(recs)[order], 0.2, label="recall", color="tab:green")
    ax.set_yticks(x)
    ax.set_yticklabels(np.array(names)[order], fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(f"score at sigma={sigma_op} (room-temperature regime)")
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Room-temperature operating point comparison")
    fig.tight_layout()
    fig.savefig(FIGS / "11_ablation_bars.png", dpi=150)
    plt.close(fig)

    print("saved", FIGS / "10_ablation_f1.png", "and 11_ablation_bars.png")


if __name__ == "__main__":
    main()
