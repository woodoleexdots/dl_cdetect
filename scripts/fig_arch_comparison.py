"""Publication-style architecture comparison figure.

Rows (a)-(d): Jung et al. 2021 baseline, SpinDETR, PeriodFormer (ours),
PF->DE hybrid (ours, final). Consistent visual grammar: input (gray),
representation (blue), network (green), output (orange); tensor shapes in
gray italics; small glyphs (waveform / slice-stack / attention / queries).

Output: results/figs/fig_arch_comparison.png (300 dpi) + .pdf (vector).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "results" / "figs"

plt.rcParams["font.family"] = ["DejaVu Sans"]
C_IN, C_REP, C_NET, C_OUT = "#e8e8e8", "#dbe9f8", "#e2f0dc", "#fdebd0"


def block(ax, x, y, w, h, title, shape=None, fc=C_NET, lw=1.0, ec="0.25",
          fs=8.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004",
                                fc=fc, ec=ec, lw=lw))
    ty = y + h * (0.62 if shape else 0.5)
    ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=fs,
            weight="bold")
    if shape:
        ax.text(x + w / 2, y + h * 0.24, shape, ha="center", va="center",
                fontsize=fs - 1.6, color="0.35", style="italic")
    return x + w


def arr(ax, x1, x2, y):
    ax.add_patch(FancyArrowPatch((x1 + 0.002, y), (x2 - 0.002, y),
                                 arrowstyle="-|>", mutation_scale=11,
                                 lw=1.1, color="0.15"))


def waveform_glyph(ax, x, y, w, h):
    t = np.linspace(0, 1, 200)
    sig = 1 - 0.35 * np.exp(-((t % 0.23 - 0.115) / 0.02) ** 2)
    sig += 0.04 * np.random.default_rng(0).normal(size=t.size)
    ax.plot(x + t * w, y + (sig - 0.55) * h, lw=0.7, color="0.2")


def stack_glyph(ax, x, y, w, h):
    rng = np.random.default_rng(1)
    img = 0.85 + 0.1 * rng.random((6, 12))
    img[:, 5:7] -= 0.45
    ax.imshow(img, extent=[x, x + w, y, y + h], cmap="viridis", vmin=0.2,
              vmax=1.0, aspect="auto", zorder=5)


def attention_glyph(ax, x, y, w, h, n=7):
    xs = x + (np.arange(n) + 0.5) * w / n
    for xi in xs:
        ax.add_patch(plt.Rectangle((xi - w / n * 0.32, y), w / n * 0.64,
                                   h * 0.28, fc="#9fc5e8", ec="0.3", lw=0.5))
    for i in [1, 3, 5]:
        for j in range(n):
            if j != i:
                ax.annotate("", xy=(xs[j], y + h * 0.3),
                            xytext=(xs[i], y + h * 0.3),
                            arrowprops=dict(arrowstyle="-", lw=0.4,
                                            color="0.45",
                                            connectionstyle="arc3,rad=0.25"))


def row_label(ax, y, letter, name, sub, color):
    ax.text(0.002, y + 0.115, letter, fontsize=13, weight="bold")
    ax.text(0.050, y + 0.125, name, fontsize=10.5, weight="bold", color=color)
    ax.text(0.050, y + 0.085, sub, fontsize=8, color="0.35")


def main():
    fig, ax = plt.subplots(figsize=(13.2, 7.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    H = 0.115
    XL = 0.20

    # ---------- (a) 2021 baseline ----------
    y = 0.845
    row_label(ax, y, "(a)", "Jung et al. (2021) — window-bank MLP",
              "reproduced baseline", "#8B3A2F")
    x = XL
    x2 = block(ax, x, y, 0.085, H, "CPMG signal", "single N\n(700,)", C_IN)
    waveform_glyph(ax, x + 0.008, y + 0.088, 0.07, 0.03)
    arr(ax, x2, x2 + 0.02, y + H / 2); x = x2 + 0.02
    x2 = block(ax, x, y, 0.135, H, "fold at candidate\nperiod TP(A)",
               "image (13×53) → 689", C_REP)
    stack_glyph(ax, x + 0.095, y + 0.012, 0.033, 0.038)
    arr(ax, x2, x2 + 0.02, y + H / 2); x = x2 + 0.02
    x2 = block(ax, x, y, 0.15, H, "MLP 2048–1024–512\n(one per window)",
               "×61 independent models", C_NET)
    arr(ax, x2, x2 + 0.02, y + H / 2); x = x2 + 0.02
    x2 = block(ax, x, y, 0.105, H, "3-class prob.", "(3,) per window", C_NET)
    arr(ax, x2, x2 + 0.02, y + H / 2); x = x2 + 0.02
    block(ax, x, y, 0.135, H, "peak picking", "A positions only\n(B needs extra stage)",
          C_OUT)
    ax.text(XL, y - 0.028, "no information shared across windows · separate "
            "denoiser harms narrow dips at room temperature", fontsize=7.8,
            color="0.4", style="italic")

    # ---------- (b) SpinDETR ----------
    y = 0.615
    row_label(ax, y, "(b)", "SpinDETR — end-to-end set prediction",
              "no physics prior (ablation)", "#38761D")
    x = XL
    x2 = block(ax, x, y, 0.085, H, "M signal", "3 channels\n(3, 700)", C_IN)
    arr(ax, x2, x2 + 0.02, y + H / 2); x = x2 + 0.02
    x2 = block(ax, x, y, 0.115, H, "1D conv stem", "(175, 128)", C_NET)
    arr(ax, x2, x2 + 0.02, y + H / 2); x = x2 + 0.02
    x2 = block(ax, x, y, 0.125, H, "Transformer\nencoder ×4", "(175, 128)", C_NET)
    arr(ax, x2, x2 + 0.02, y + H / 2); x = x2 + 0.02
    x2 = block(ax, x, y, 0.145, H, "10 learned queries\n+ decoder ×4",
               "(10, 128)", C_NET)
    arr(ax, x2, x2 + 0.02, y + H / 2); x = x2 + 0.02
    block(ax, x, y, 0.135, H, "set output", "10 × (p, A∥, A⊥)\nHungarian-matched",
          C_OUT)
    ax.text(XL, y - 0.028, "direct (A, B) regression in one forward pass · "
            "learns periodicity from data alone", fontsize=7.8, color="0.4",
            style="italic")

    # ---------- (c) PeriodFormer ----------
    y = 0.385
    row_label(ax, y, "(c)", "PeriodFormer (ours) — window tokens + attention",
              "physics prior + attention", "#1155CC")
    x = XL
    x2 = block(ax, x, y, 0.085, H, "M signal", "3 channels\n(3, 700)", C_IN)
    arr(ax, x2, x2 + 0.02, y + H / 2); x = x2 + 0.02
    ax.add_patch(FancyBboxPatch((x, y), 0.15, H, boxstyle="round,pad=0.004",
                                fc=C_REP, ec="0.25", lw=1.0))
    ax.text(x + 0.075, y + 0.088, "fold at ALL 61 candidate periods",
            ha="center", va="center", fontsize=8.0, weight="bold")
    ax.text(x + 0.075, y + 0.055, "tokens (61, 3, 13, 53)", ha="center",
            fontsize=7.0, color="0.35", style="italic")
    x2 = x + 0.15
    stack_glyph(ax, x + 0.022, y + 0.010, 0.028, 0.024)
    stack_glyph(ax, x + 0.061, y + 0.010, 0.028, 0.024)
    stack_glyph(ax, x + 0.100, y + 0.010, 0.028, 0.024)
    arr(ax, x2, x2 + 0.02, y + H / 2); x = x2 + 0.02
    x2 = block(ax, x, y, 0.115, H, "shared CNN\nembedding", "(61, 128)", C_NET)
    arr(ax, x2, x2 + 0.02, y + H / 2); x = x2 + 0.02
    ax.add_patch(FancyBboxPatch((x, y), 0.16, H, boxstyle="round,pad=0.004",
                                fc=C_NET, ec="0.25", lw=1.0))
    ax.text(x + 0.08, y + 0.088, "Transformer ACROSS windows ×4",
            ha="center", va="center", fontsize=8.0, weight="bold")
    ax.text(x + 0.08, y + 0.058, "(61, 128)", ha="center", fontsize=7.0,
            color="0.35", style="italic")
    x2 = x + 0.16
    attention_glyph(ax, x + 0.014, y + 0.007, 0.132, 0.036)
    arr(ax, x2, x2 + 0.02, y + H / 2); x = x2 + 0.02
    block(ax, x, y, 0.125, H, "existence curve", "P(spin) over A grid\n(61,)",
          C_OUT)
    ax.text(XL, y - 0.028, "keeps the period-folding prior of (a), replaces 61 "
            "independent classifiers with ONE model attending across windows "
            "→ zero false positives in all benchmarks", fontsize=7.8,
            color="0.4", style="italic")

    # ---------- (d) hybrid ----------
    y = 0.13
    row_label(ax, y, "(d)", "PF→DE hybrid (ours, final)",
              "detect × enumerate", "#CC0000")
    x = XL
    x2 = block(ax, x, y, 0.13, H, "P(spin) curve\nfrom (c)", "(61,)", C_IN)
    arr(ax, x2, x2 + 0.02, y + H / 2); x = x2 + 0.02
    x2 = block(ax, x, y, 0.135, H, "threshold →\ncandidate regions",
               "interval list", C_REP)
    arr(ax, x2, x2 + 0.02, y + H / 2); x = x2 + 0.02
    x2 = block(ax, x, y, 0.165, H, "region-constrained\ngreedy DE (∏Mᵢ fit)",
               "add one spin per step", C_NET)
    arr(ax, x2, x2 + 0.02, y + H / 2); x = x2 + 0.02
    x2 = block(ax, x, y, 0.11, H, "BIC model\nselection", "k* automatic", C_NET)
    arr(ax, x2, x2 + 0.02, y + H / 2); x = x2 + 0.02
    block(ax, x, y, 0.125, H, "spin list", "{(A∥, A⊥)} × k*\nNV1: 14 · NV2: 10",
          C_OUT)
    ax.text(XL, y - 0.028, "multiple spins allowed per region (cluster "
            "enumeration) · best F1 in every ground-truth benchmark arm "
            "(cryo / room-T / model-mismatch)", fontsize=7.8, color="0.4",
            style="italic")

    # legend
    lx = 0.30
    for c, t in [(C_IN, "input"), (C_REP, "representation"),
                 (C_NET, "computation"), (C_OUT, "output")]:
        ax.add_patch(plt.Rectangle((lx, 0.012), 0.018, 0.022, fc=c, ec="0.3",
                                   lw=0.7))
        ax.text(lx + 0.024, 0.023, t, fontsize=8, va="center")
        lx += 0.115

    fig.savefig(FIGS / "fig_arch_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGS / "fig_arch_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved fig_arch_comparison.png/.pdf")


if __name__ == "__main__":
    main()
