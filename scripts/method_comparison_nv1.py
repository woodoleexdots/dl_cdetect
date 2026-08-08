"""Per-method NV1 detections: comparison table + stacked overlay figure.

Methods compared (all on the same real NV1 data):
  2021-mlp   : reproduced 2021 pipeline (MLP window bank, N=16) --
               evaluated here on real NV1 for the first time
  cnn-3ch    : our CNN window bank scan (results/hpc/hpc_scan_NV1.csv)
  cdetect-de : plain wide-bound greedy DE (+-120 kHz run)
  spindetr   : SpinDETR set predictions (p >= 0.5)
  pf-*       : PeriodFormer peak detections on each grid
  hybrid     : narrow-grid PF->DE hybrid
  ENSEMBLE   : final ensemble-region hybrid (reference, 14 spins)
  anchor     : pptx eyeball estimates (|A| positions)

Outputs: results/figs/23_method_comparison_NV1.png + a markdown table.
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
import torch
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpmg.datagen import GenConfig, signals_to_image
from cpmg.models import HPC_MLP2021
from cpmg.physics import target_period
from cpmg.represent import envelope_normalize

ROOT = Path(__file__).resolve().parents[1]
BV2 = ROOT / "results" / "benchmark_v2"
PF = ROOT / "results" / "periodformer"


def eval_2021_bank_on_nv1(device):
    nv1 = pd.read_excel(ROOT / "dataset" / "exp_dataset" / "CPMG_NV1.xlsx")
    tau = nv1["a"].to_numpy(float)
    m16 = envelope_normalize(tau, nv1["CPMG16"].to_numpy(float))[0]
    m_recs = np.array([m16], dtype=np.float32)
    gen_cfg = GenConfig(n_pulses=(16,), b_field_g=440.1, b_repr=20e3)
    centers = np.arange(-60, 60 + 1e-9, 2.0) * 1e3
    p_list = []
    for a_c in centers:
        f = BV2 / "bank_B_room" / f"w_{a_c/1e3:+08.1f}kHz.pt"
        tp = target_period(a_c, gen_cfg.b_repr, gen_cfg.b_field_g)
        s = int(np.floor((tau[-1] - tau[0]) / tp))
        model = HPC_MLP2021(1 * s * gen_cfg.image_width).to(device)
        model.load_state_dict(torch.load(f, map_location=device))
        model.eval()
        img = signals_to_image(tau, m_recs, tp, gen_cfg)
        with torch.no_grad():
            logits = model(torch.from_numpy(img[None]).to(device))
            p_list.append(1.0 - torch.sigmoid(logits[:, 0]).item())
    p = np.array(p_list)
    pk, _ = find_peaks(p, height=0.6, prominence=0.1)
    return centers[pk] / 1e3, (centers / 1e3, p)


def pf_peaks(fname, thresh=0.4):
    d = np.load(PF / fname)
    centers = (d["centers_hz"] if "centers_hz" in d else d["centers"]) / 1e3
    pk, _ = find_peaks(d["p"], height=thresh, prominence=0.1)
    return centers[pk]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hyb = json.loads((BV2 / "hybrid_results.json").read_text())

    a2021, _ = eval_2021_bank_on_nv1(device)
    print("2021-mlp NV1 detections (kHz):", [round(x, 1) for x in a2021], flush=True)

    df_cnn = pd.read_csv(ROOT / "results" / "hpc" / "hpc_scan_NV1.csv")
    pk, _ = find_peaks(df_cnn["p_spin"].to_numpy(), height=0.6, prominence=0.1)
    a_cnn = df_cnn["a_khz"].to_numpy()[pk]

    methods = {
        "anchor (ppt, |A|)": [88, 8, 38, 5],
        "2021-mlp (N=16)": list(np.round(a2021, 1)),
        "cnn-3ch bank": list(a_cnn),
        "cdetect-DE (wide)": [9.2, -5.6, 91.2, 40.7, 3.3, 65.2, 23.7, -87.6],
        "SpinDETR": [-6.2, 7.4, 42.5, 13.1, -15.5, 55.9, -2.2, 37.6, -34.4],
        "PF +-60": list(pf_peaks("nv1_curve.npz")),
        "PF +-120": list(pf_peaks("curve_NV1_NV1_wide.npz", 0.3)),
        "PF +-200": list(pf_peaks("curve_NV1_NV1_w200.npz", 0.3)),
        "hybrid (narrow)": [s[0] for s in hyb["nv1"]["spins_khz"]],
        "ENSEMBLE (final)": [s[0] for s in hyb["nv1_ensemble"]["spins_khz"]],
    }
    ens = sorted(methods["ENSEMBLE (final)"])
    ens_b = {s[0]: s[1] for s in hyb["nv1_ensemble"]["spins_khz"]}

    # -------- table: ensemble spins x methods (match within 4 kHz, |A| for anchor)
    tol = 4.0
    lines = ["| A (kHz) | B (kHz) | " + " | ".join(k for k in methods if k != "ENSEMBLE (final)") + " |"]
    lines.append("|" + "---|" * (2 + len(methods) - 1))
    for a in ens:
        row = [f"| {a:+.1f} | {ens_b[a]:.0f} "]
        for name, dets in methods.items():
            if name == "ENSEMBLE (final)":
                continue
            if name.startswith("anchor"):
                hit = any(abs(abs(a) - d) <= tol for d in dets)
            else:
                hit = any(abs(a - d) <= tol for d in dets)
            row.append("✅ " if hit else "— ")
        lines.append("|".join(row) + "|")
    table = "\n".join(lines)
    (BV2 / "method_comparison_NV1.md").write_text(table)
    print(table, flush=True)

    # -------- figure: stacked strip plot
    fig, ax = plt.subplots(figsize=(14, 7))
    names = list(methods.keys())
    for a in ens:
        ax.axvline(a, color="tab:red", alpha=0.25, lw=1.2)
    for y, name in enumerate(names):
        dets = methods[name]
        if name.startswith("anchor"):
            xs = []
            for d in dets:
                xs += [d, -d]
            ax.plot(xs, [y] * len(xs), "D", ms=7, color="tab:orange",
                    alpha=0.7, label=None)
        else:
            color = "tab:red" if name.startswith("ENSEMBLE") else "tab:blue"
            ms = 9 if name.startswith("ENSEMBLE") else 6
            ax.plot(dets, [y] * len(dets), "o", ms=ms, color=color, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("A (kHz)")
    ax.set_xlim(-130, 130)
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("NV1: detections per method (red lines = final ensemble spins; "
                 "orange diamonds = ppt anchors at +-|A|)")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "figs" / "23_method_comparison_NV1.png", dpi=150)
    print("figure -> results/figs/23_method_comparison_NV1.png", flush=True)


if __name__ == "__main__":
    main()
