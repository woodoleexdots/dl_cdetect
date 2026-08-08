"""Overlay the PF->DE hybrid fits on the real CPMG data.

Reads results/benchmark_v2/hybrid_results.json (nv1 / nv2 entries) and
renders forward-model curves over the experimental signals, with the PF
candidate regions shaded. Figures: 17_hybrid_overlay_NV1.png,
18_hybrid_overlay_NV2.png.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpmg.physics import cpmg_M
from cpmg.represent import envelope_normalize

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "results" / "figs"
US = 1e-6


def overlay_panel(ax, tau, m_data, ab_khz, n_pulse, b_field, regions_khz=None,
                  zoom=None, legend=False):
    ab = np.array(ab_khz) * 1e3
    m_fit = cpmg_M(tau, ab, n_pulse, b_field)
    rmse = float(np.sqrt(np.mean((m_fit - m_data) ** 2)))
    ax.plot(tau / US, m_data, ".", ms=2.2, color="0.55", alpha=0.8,
            label="experiment (env-normalized M)")
    ax.plot(tau / US, m_fit, "-", lw=1.1, color="tab:red",
            label=f"hybrid fit ({len(ab)} spins)")
    if zoom:
        ax.set_xlim(*zoom)
    ax.set_ylabel("M")
    ax.set_title(f"N={n_pulse}   RMSE={rmse:.3f}", fontsize=9, loc="right")
    if legend:
        ax.legend(fontsize=8, loc="lower right")
    return rmse


def main():
    res = json.loads((ROOT / "results" / "benchmark_v2" / "hybrid_results.json").read_text())

    # ---------------- NV1 ----------------
    if "nv1" in res:
        spins = res["nv1"]["spins_khz"]
        nv1 = pd.read_excel(ROOT / "dataset" / "exp_dataset" / "CPMG_NV1.xlsx")
        tau = nv1["a"].to_numpy(float)
        fig, axes = plt.subplots(4, 1, figsize=(14, 11))
        rmses = []
        for ax, (n, col) in zip(axes[:3], [(8, "CPMG8"), (16, "CPMG16"),
                                           (20, "CPMG20")]):
            m, _ = envelope_normalize(tau, nv1[col].to_numpy(float))
            rmses.append(overlay_panel(ax, tau, m, spins, n, 440.1,
                                       legend=(n == 8)))
        m, _ = envelope_normalize(tau, nv1["CPMG16"].to_numpy(float))
        overlay_panel(axes[3], tau, m, spins, 16, 440.1, zoom=(0, 6))
        axes[3].set_title("N=16 zoom 0-6 us", fontsize=9, loc="left")
        axes[3].set_xlabel("tau (us)")
        s_str = ", ".join(f"({a:+.0f},{b:.0f})" for a, b in spins)
        fig.suptitle(f"NV1: PF->DE hybrid fit overlay  |  "
                     f"k*={res['nv1']['k']}  spins (A,B kHz): {s_str}",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(FIGS / "17_hybrid_overlay_NV1.png", dpi=150)
        plt.close(fig)
        print("NV1 RMSE per N:", [round(r, 3) for r in rmses])

    # ---------------- NV2 ----------------
    if "nv2" in res:
        spins = res["nv2"]["spins_khz"]
        nv2 = pd.read_excel(ROOT / "dataset" / "exp_dataset" / "CPMG_NV2.xlsx")
        tau2 = nv2["Time"].to_numpy(float)
        m2, _ = envelope_normalize(tau2, nv2["CPMG16"].to_numpy(float))
        fig, axes = plt.subplots(3, 1, figsize=(14, 8.5))
        overlay_panel(axes[0], tau2, m2, spins, 16, 440.1, legend=True)
        axes[0].set_title("full range", fontsize=9, loc="left")
        overlay_panel(axes[1], tau2, m2, spins, 16, 440.1, zoom=(0, 4))
        axes[1].set_title("zoom 0-4 us", fontsize=9, loc="left")
        overlay_panel(axes[2], tau2, m2, spins, 16, 440.1, zoom=(4, 8))
        axes[2].set_title("zoom 4-8 us", fontsize=9, loc="left")
        axes[2].set_xlabel("tau (us)")
        s_str = ", ".join(f"({a:+.0f},{b:.0f})" for a, b in spins)
        fig.suptitle(f"NV2 (CPMG-16): PF->DE hybrid fit overlay  |  "
                     f"k*={res['nv2']['k']}  spins (A,B kHz): {s_str}",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(FIGS / "18_hybrid_overlay_NV2.png", dpi=150)
        plt.close(fig)
        print("saved overlay figures 17/18")


if __name__ == "__main__":
    main()
