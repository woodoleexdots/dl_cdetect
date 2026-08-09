"""Assets for the two new summary-deck slides.

1. results/slides_assets/delft50_overview.png
   - left : the 50 published spins in (A_par, A_perp) with the scoring box
   - right: per-spin CPMG dip depth (cryo vs room window) vs noise floor
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

from cpmg.physics import cpmg_M

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "results" / "slides_assets"
B_CRYO = 403.553
B_OURS = 440.1

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
    in_box = (np.abs(par) <= 60e3) & (perp >= 5e3)

    tau_cryo = np.arange(1, 7001) * 4e-9
    tau_room = np.arange(1, 701) * 20e-9
    depth_cryo = np.array([0.5 * (1 - cpmg_M(tau_cryo, [[a, b]], 32, B_CRYO).min())
                           for a, b in zip(par, perp)])
    depth_room = np.array([0.5 * (1 - cpmg_M(tau_room, [[a, b]], 16, B_OURS).min())
                           for a, b in zip(par, perp)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 3.6),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    ax1.axvspan(-60, 60, color="tab:blue", alpha=0.07)
    ax1.axhline(5, color="tab:red", ls=":", lw=1)
    ax1.scatter(par[~in_box] / 1e3, perp[~in_box] / 1e3, s=30, color="0.6",
                label=f"박스 밖 {int((~in_box).sum())}개")
    ax1.scatter(par[in_box] / 1e3, perp[in_box] / 1e3, s=34, color="tab:red",
                label=f"채점 GT {int(in_box.sum())}개")
    ax1.set_xlabel("A∥ (kHz)", fontsize=10)
    ax1.set_ylabel("A⊥ (kHz)", fontsize=10)
    ax1.set_title(f"공개 스핀 {n}개 — 채점 박스 |A∥|≤60 kHz · A⊥≥5 kHz",
                  fontsize=10)
    ax1.legend(fontsize=8.5, loc="upper right")
    ax1.tick_params(labelsize=8.5)

    order = np.argsort(-depth_cryo)
    ax2.semilogy(np.arange(n), depth_cryo[order], "o-", ms=3, lw=1,
                 label="저온 창 (N=32, 28 µs)")
    ax2.semilogy(np.arange(n), np.sort(depth_room)[::-1], "s-", ms=3, lw=1,
                 color="tab:orange", label="상온 창 (N=16, 14 µs)")
    ax2.axhline(0.04, color="tab:blue", ls=":", lw=1.2, label="2σ 저온 (0.04)")
    ax2.axhline(0.12, color="tab:red", ls=":", lw=1.2, label="2σ 상온 (0.12)")
    ax2.set_xlabel("스핀 순위 (dip 깊이 내림차순)", fontsize=10)
    ax2.set_ylabel("최대 CPMG dip 깊이", fontsize=10)
    ax2.set_title("검출 가능성: dip 깊이 vs 노이즈 바닥", fontsize=10)
    ax2.legend(fontsize=8, loc="lower left")
    ax2.tick_params(labelsize=8.5)

    fig.tight_layout()
    fig.savefig(ASSETS / "delft50_overview.png", dpi=150)
    print("saved ->", ASSETS / "delft50_overview.png")


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig_noise_model()
    fig_delft50_overview()
