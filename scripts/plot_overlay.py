"""Overlay fitted spin models on the real CPMG data (paper Fig. 4b style).

Reads the wide-range DE spin lists (results/benchmark_wide.json if present,
else built-in lists from the analysis log) and draws, per channel:
  - envelope-normalized experimental M(tau) (dots)
  - forward-model M(tau) from the fitted spins (solid line)
  - per-spin dip positions marked, per-channel RMSE reported
Figures: 12_overlay_NV1.png, 13_overlay_NV2.png (+ zoom panels).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpmg.physics import B_FIELD_G, cpmg_M, target_period
from cpmg.represent import envelope_normalize

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "results" / "figs"
US = 1e-6

# wide-range greedy-DE fits (A, B) in kHz — see conversation log 2026-08-08
NV1_SPINS = [(9.2, 31.0), (-5.6, 26.2), (3.3, 30.3), (23.7, 23.7),
             (40.7, 33.3), (65.2, 30.3), (91.2, 36.4), (-87.6, 19.4)]
NV2_SPINS = [(-51.5, 198.9), (51.1, 94.3), (346.2, 263.3), (-39.2, 67.4),
             (-152.3, 98.6), (-14.1, 50.1)]


def overlay(ax, tau, m_data, ab_khz, n_pulse, zoom=None, show_legend=False):
    ab = np.array(ab_khz) * 1e3
    m_fit = cpmg_M(tau, ab, n_pulse, B_FIELD_G)
    rmse = float(np.sqrt(np.mean((m_fit - m_data) ** 2)))
    ax.plot(tau / US, m_data, ".", ms=2.2, color="0.55", alpha=0.8,
            label="experiment (env-normalized M)")
    ax.plot(tau / US, m_fit, "-", lw=1.1, color="tab:red",
            label=f"fitted model ({len(ab)} spins)")
    if zoom:
        ax.set_xlim(*zoom)
    ax.set_ylabel("M")
    ax.set_title(f"N={n_pulse}   RMSE={rmse:.3f}", fontsize=9, loc="right")
    if show_legend:
        ax.legend(fontsize=8, loc="lower right")
    return rmse


def spin_dip_markers(ax, ab_khz, colors):
    for (a, b), c in zip(ab_khz, colors):
        tp = target_period(a * 1e3, b * 1e3, B_FIELD_G)
        for k in range(1, 40):
            t = (2 * k - 1) * tp / 2 / US
            if t > ax.get_xlim()[1]:
                break
            if t >= ax.get_xlim()[0]:
                ax.axvline(t, color=c, alpha=0.16, lw=0.8)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)

    # ---------------- NV1 ----------------
    nv1 = pd.read_excel(ROOT / "dataset" / "exp_dataset" / "CPMG_NV1.xlsx")
    tau = nv1["a"].to_numpy(float)
    fig, axes = plt.subplots(4, 1, figsize=(14, 11))
    rmses = []
    for ax, (n_pulse, col) in zip(axes[:3], [(8, "CPMG8"), (16, "CPMG16"),
                                             (20, "CPMG20")]):
        m_data, _ = envelope_normalize(tau, nv1[col].to_numpy(float))
        rmses.append(overlay(ax, tau, m_data, NV1_SPINS, n_pulse,
                             show_legend=(n_pulse == 8)))
    # zoom panel: N=16, 0-6 us with per-spin dip markers
    m_data, _ = envelope_normalize(tau, nv1["CPMG16"].to_numpy(float))
    ax = axes[3]
    overlay(ax, tau, m_data, NV1_SPINS, 16, zoom=(0.0, 6.0))
    colors = plt.cm.tab10(np.linspace(0, 1, len(NV1_SPINS)))
    spin_dip_markers(ax, NV1_SPINS, colors)
    ax.set_title("N=16 zoom 0-6 us (vertical lines: per-spin dip positions)",
                 fontsize=9, loc="left")
    ax.set_xlabel("tau (us)")
    spins_str = ", ".join(f"({a:+.0f},{b:.0f})" for a, b in NV1_SPINS)
    fig.suptitle(f"NV1: experiment vs fitted forward model  |  "
                 f"spins (A,B kHz): {spins_str}", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "12_overlay_NV1.png", dpi=150)
    plt.close(fig)
    print("NV1 RMSE per N:", [round(r, 3) for r in rmses])

    # ---------------- NV2 ----------------
    nv2 = pd.read_excel(ROOT / "dataset" / "exp_dataset" / "CPMG_NV2.xlsx")
    tau2 = nv2["Time"].to_numpy(float)
    m2, _ = envelope_normalize(tau2, nv2["CPMG16"].to_numpy(float))
    fig, axes = plt.subplots(3, 1, figsize=(14, 8.5))
    overlay(axes[0], tau2, m2, NV2_SPINS, 16, show_legend=True)
    axes[0].set_title("full range", fontsize=9, loc="left")
    overlay(axes[1], tau2, m2, NV2_SPINS, 16, zoom=(0.0, 4.0))
    axes[1].set_title("zoom 0-4 us", fontsize=9, loc="left")
    overlay(axes[2], tau2, m2, NV2_SPINS, 16, zoom=(4.0, 8.0))
    axes[2].set_title("zoom 4-8 us", fontsize=9, loc="left")
    axes[2].set_xlabel("tau (us)")
    spins_str = ", ".join(f"({a:+.0f},{b:.0f})" for a, b in NV2_SPINS)
    fig.suptitle(f"NV2 (CPMG-16): experiment vs fitted forward model  |  "
                 f"spins (A,B kHz): {spins_str}", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "13_overlay_NV2.png", dpi=150)
    plt.close(fig)
    print("saved 12_overlay_NV1.png / 13_overlay_NV2.png")


if __name__ == "__main__":
    main()
