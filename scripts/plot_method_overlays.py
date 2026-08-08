"""Per-algorithm forward-model overlays on the real data (fig-22 style).

Each method that outputs full (A, B) pairs is rendered as its own colored
model curve, stacked with a vertical offset against the (repeated) gray
experimental trace, so each fit can be judged against the data separately
and across methods. Window-bank methods (2021-MLP / CNN / PF) output A
positions only (no B) and therefore cannot be rendered as curves.

Figures: 24_method_overlays_NV1.png, 25_method_overlays_NV2.png
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
BV2 = ROOT / "results" / "benchmark_v2"
FIGS = ROOT / "results" / "figs"
US = 1e-6
B_OURS = 440.1

DE_NV1 = [(9.2, 31.0), (-5.6, 26.2), (91.2, 36.4), (40.7, 33.3),
          (3.3, 30.3), (65.2, 30.3), (23.7, 23.7), (-87.6, 19.4)]
SPINDETR_NV1 = [(-6.2, 28.6), (7.4, 28.6), (42.5, 30.8), (13.1, 25.0),
                (-15.5, 20.9), (55.9, 24.3), (-2.2, 23.0), (37.6, 27.4),
                (-34.4, 17.1)]
DE_NV2 = [(-51.5, 198.9), (51.1, 94.3), (346.2, 263.3), (-39.2, 67.4),
          (-152.3, 98.6), (-14.1, 50.1)]

COLORS = ["tab:blue", "tab:green", "tab:purple", "tab:red"]


def stacked_panel(ax, tau, m_data, method_specs, n_pulse, zoom=None, step=1.6):
    for i, (name, spins, color) in enumerate(method_specs):
        off = -step * i
        ab = np.array(spins) * 1e3
        m_fit = cpmg_M(tau, ab, n_pulse, B_OURS)
        rmse = float(np.sqrt(np.mean((m_fit - m_data) ** 2)))
        ax.plot(tau / US, m_data + off, ".", ms=1.8, color="0.6", alpha=0.6)
        ax.plot(tau / US, m_fit + off, "-", lw=1.0, color=color,
                label=f"{name} ({len(spins)} spins, RMSE {rmse:.3f})")
        ax.text(0.01, off + 1.28, name, fontsize=8, color=color,
                transform=ax.get_yaxis_transform())
    if zoom:
        ax.set_xlim(*zoom)
    ax.set_yticks([])
    ax.set_ylabel("M (offset per method)")


def make_figure(tau, m_data, method_specs, n_pulse, title, fname,
                zooms=((None), (0, 6))):
    fig, axes = plt.subplots(len(zooms), 1,
                             figsize=(14, 3.2 * len(method_specs)))
    for ax, zoom in zip(np.atleast_1d(axes), zooms):
        stacked_panel(ax, tau, m_data, method_specs, n_pulse, zoom=zoom)
        ax.set_title("full range" if zoom is None else f"zoom {zoom[0]}-{zoom[1]} us",
                     fontsize=9, loc="left")
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="upper right", framealpha=0.9)
    np.atleast_1d(axes)[-1].set_xlabel("tau (us)")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / fname, dpi=150)
    plt.close(fig)
    print("saved", fname)


def main():
    hyb = json.loads((BV2 / "hybrid_results.json").read_text())

    # ---------------- NV1 (N=16 channel) ----------------
    nv1 = pd.read_excel(ROOT / "dataset" / "exp_dataset" / "CPMG_NV1.xlsx")
    tau1 = nv1["a"].to_numpy(float)
    m1, _ = envelope_normalize(tau1, nv1["CPMG16"].to_numpy(float))
    specs1 = [
        ("cdetect-DE (wide)", DE_NV1, COLORS[0]),
        ("SpinDETR", SPINDETR_NV1, COLORS[1]),
        ("hybrid (narrow)", hyb["nv1"]["spins_khz"], COLORS[2]),
        ("ENSEMBLE (final)", hyb["nv1_ensemble"]["spins_khz"], COLORS[3]),
    ]
    make_figure(tau1, m1, specs1, 16,
                "NV1 (CPMG-16): per-algorithm forward-model overlays "
                "(gray = experiment, repeated per row)",
                "24_method_overlays_NV1.png")

    # ---------------- NV2 (CPMG-16) ----------------
    nv2 = pd.read_excel(ROOT / "dataset" / "exp_dataset" / "CPMG_NV2.xlsx")
    tau2 = nv2["Time"].to_numpy(float)
    m2, _ = envelope_normalize(tau2, nv2["CPMG16"].to_numpy(float))
    specs2 = [
        ("cdetect-DE (wide)", DE_NV2, COLORS[0]),
        ("hybrid v1", hyb["nv2"]["spins_khz"], COLORS[1]),
        ("hybrid refined", hyb["nv2_refined"]["spins_khz"], COLORS[2]),
        ("ENSEMBLE (final)", hyb["nv2_ensemble"]["spins_khz"], COLORS[3]),
    ]
    make_figure(tau2, m2, specs2, 16,
                "NV2 (CPMG-16): per-algorithm forward-model overlays "
                "(gray = experiment, repeated per row)",
                "25_method_overlays_NV2.png", zooms=(None, (0, 5)))


if __name__ == "__main__":
    main()
