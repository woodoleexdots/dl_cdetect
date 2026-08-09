"""Assets for the two new summary-deck slides.

1. results/slides_assets/delft50_overview.png
   - the 50 published spins: (A_par, A_perp) scatter + both distributions
2. results/slides_assets/noise_model.png
   - four zoom panels of the raw NV1/NV2 data with the rolling-median signal
     and the +-2*sigma_hat band from the successive-difference estimator
   - one summary panel: sigma_hat per channel vs adopted sigma and the
     training randomization range

Run with any python that has numpy/matplotlib/openpyxl (no torch needed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from openpyxl import load_workbook

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "results" / "slides_assets"

plt.rcParams["font.family"] = ["Noto Sans CJK KR", "Apple SD Gothic Neo",
                               "AppleGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def read_channels():
    """(name, tau_us, y) for NV1 N=8/16/20 and NV2 N=16, px scale."""
    out = []
    for nv, cols in [("NV1", ["CPMG8", "CPMG16", "CPMG20"]),
                     ("NV2", ["CPMG16"])]:
        ws = load_workbook(ROOT / "dataset" / "exp_dataset" / f"CPMG_{nv}.xlsx",
                           read_only=True).active
        rows = list(ws.values)
        head = list(rows[0])
        data = np.array([[float(v) for v in r] for r in rows[1:]])
        tau = np.arange(1, len(data) + 1) * 20e-9 / 1e-6  # 20 ns grid, in us
        for c in cols:
            y = data[:, head.index(c)]
            out.append((f"{nv} N={c[4:]}", tau, y))
    return out


def sigma_hat(y):
    return float(np.std(np.diff(y)) / np.sqrt(2))


def fig_noise_model():
    chans = read_channels()
    fig, axes = plt.subplots(1, 5, figsize=(15.5, 3.1),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.85]})
    zoom = (4.0, 7.0)
    for ax, (name, tau, y) in zip(axes[:4], chans):
        s = sigma_hat(y)
        m = (tau >= zoom[0]) & (tau <= zoom[1])
        # rolling median as the slow "signal"; the scatter around it is noise
        k = 9
        pad = np.pad(y, k // 2, mode="edge")
        med = np.array([np.median(pad[i:i + k]) for i in range(len(y))])
        ax.plot(tau[m], y[m], ".", ms=2.6, color="0.35", label="실측")
        ax.fill_between(tau[m], med[m] - 2 * s, med[m] + 2 * s,
                        color="tab:orange", alpha=0.30, lw=0,
                        label="이동중앙값 ±2σ̂")
        ax.plot(tau[m], med[m], "-", lw=0.9, color="tab:red")
        ax.set_title(f"{name}   σ̂ = {s:.3f}", fontsize=10)
        ax.set_xlabel("τ (µs)", fontsize=9)
        ax.tick_params(labelsize=8)
        if ax is axes[0]:
            ax.set_ylabel("px (0–1)", fontsize=9)
            ax.legend(fontsize=7, loc="lower left")

    ax = axes[4]
    names = [c[0] for c in chans]
    sigs = [sigma_hat(c[2]) for c in chans]
    ax.bar(range(4), sigs, color=["#4C78A8"] * 3 + ["#72B7B2"], width=0.62)
    ax.axhspan(0.03, 0.12, color="tab:orange", alpha=0.15, lw=0)
    ax.axhline(0.06, color="tab:red", ls="--", lw=1.4)
    ax.text(3.45, 0.062, "채택 σ=0.06", color="tab:red", fontsize=8.5,
            ha="right", va="bottom")
    ax.text(3.45, 0.113, "학습 범위 0.03–0.12", color="tab:orange",
            fontsize=8, ha="right", va="top")
    for i, s in enumerate(sigs):
        ax.text(i, s + 0.002, f"{s:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(4))
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
    ax.set_ylim(0, 0.135)
    ax.set_ylabel("σ̂ (px)", fontsize=9)
    ax.set_title("채널별 σ̂ vs 채택값", fontsize=10)
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(ASSETS / "noise_model.png", dpi=150)
    print("saved ->", ASSETS / "noise_model.png")


def fig_delft50_overview():
    d = np.load(ROOT / "dataset" / "delft_public" / "bath50.npz")
    par, perp = d["a_par"], d["a_perp"]
    n = len(par)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12.6, 3.4),
                                        gridspec_kw={"width_ratios": [1.25, 1, 1]})
    ax1.scatter(par / 1e3, perp / 1e3, s=34, color="tab:blue", alpha=0.85)
    ax1.set_xlabel("A∥ (kHz)", fontsize=10)
    ax1.set_ylabel("A⊥ (kHz)", fontsize=10)
    ax1.set_title(f"공개된 ¹³C 스핀 {n}개의 hyperfine (A∥, A⊥)", fontsize=10)
    ax1.tick_params(labelsize=8.5)

    ax2.hist(par / 1e3, bins=30, color="tab:blue", alpha=0.85)
    ax2.set_xlabel("A∥ (kHz)", fontsize=10)
    ax2.set_ylabel("스핀 수", fontsize=10)
    ax2.set_title("A∥ 분포 — 대부분 약결합(|A∥|<50 kHz)", fontsize=10)
    ax2.tick_params(labelsize=8.5)

    ax3.hist(perp / 1e3, bins=30, color="tab:green", alpha=0.85)
    ax3.set_xlabel("A⊥ (kHz)", fontsize=10)
    ax3.set_ylabel("스핀 수", fontsize=10)
    ax3.set_title("A⊥ 분포", fontsize=10)
    ax3.tick_params(labelsize=8.5)

    fig.tight_layout()
    fig.savefig(ASSETS / "delft50_overview.png", dpi=150)
    print("saved ->", ASSETS / "delft50_overview.png")


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig_noise_model()
    fig_delft50_overview()
