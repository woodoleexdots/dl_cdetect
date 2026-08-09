"""Figure 26: validation on the published 50-spin bath, shown explicitly.

For one cryo-arm and one room-arm realization of the digital twin built from
the van de Stolpe 2024 public dataset:
  left  : the twin CPMG signal (gray) with the forward-model curve of the
          spins DETECTED by hybrid v2 (red) overlaid
  right : ladder comparing published GT spin positions (top) with detected
          positions (bottom); matched pairs connected.

Output: results/figs/26_twin_validation.png
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpmg.hybrid import pf_candidate_regions
from cpmg.periodformer import TokenBuilder
from cpmg.physics import cpmg_M, target_period
from cpmg.rjmcmc import RJMCMC

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bv2", ROOT / "scripts" / "benchmark_v2.py")
bv2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bv2)
spec2 = importlib.util.spec_from_file_location("pfb", ROOT / "scripts" / "pf_benchmark_v2.py")
pfb = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(pfb)

CENTERS = pfb.CENTERS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
US = 1e-6

bath, labels = bv2.load_bath()
box = (np.abs(bath[:, 0]) <= 60e3) & (bath[:, 1] >= 5e3)
GT = bath[box]
GT_LB = [labels[i] for i in np.where(box)[0]]


def detect_v2(arm, model, builder, tau, ana, max_spins, margin, n_iter, seed=100):
    a = bv2.ARMS[arm]
    _, m_recs = bv2.make_arm_dataset(arm, bath, seed)
    with torch.no_grad():
        p = torch.sigmoid(model(builder(
            torch.from_numpy(m_recs)[None].to(DEVICE))))[0].cpu().numpy()
    regions = pf_candidate_regions(p, CENTERS, thresh=0.35, margin_hz=margin)
    mc = RJMCMC(tau, m_recs, a["n_pulses"], ana, sigma=a["sigma"],
                max_spins=max_spins, seed=0, regions=regions)
    out = mc.run(n_iter=n_iter)
    return m_recs, np.array(out["map_spins"]), regions


def ladder(ax, det_a, tol):
    gt_a = GT[:, 0] / 1e3
    for a, b in zip(gt_a, GT[:, 1] / 1e3):
        ax.plot([a, a], [0.65, 1.0], color="tab:red", lw=1.6, alpha=0.8)
    for a in det_a / 1e3:
        ax.plot([a, a], [0.0, 0.35], color="tab:blue", lw=1.6, alpha=0.9)
    # match lines
    det = list(det_a)
    for g in sorted(GT[:, 0]):
        if not det:
            break
        d = min(det, key=lambda x: abs(x - g))
        if abs(d - g) <= tol:
            ax.plot([g / 1e3, d / 1e3], [0.65, 0.35], color="0.5", lw=0.7)
            det.remove(d)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0.17, 0.83])
    ax.set_yticklabels(["검출 (hybrid v2)", "공개 GT (27 스핀)"], fontsize=9)
    ax.set_xlabel("A (kHz)")
    ax.set_xlim(-62, 62)


def main():
    # models
    a_room = bv2.ARMS["B_room"]
    tau_room = np.arange(1, a_room["n_tau"] + 1) * a_room["dt"]
    cfg_room = pfb.make_gen_cfg("B_room", a_room["b_field"])
    model_room, builder_room, _ = pfb.train_pf("pf_room", tau_room, cfg_room,
                                               13, DEVICE, 25000, 25, 1)
    a_cryo = bv2.ARMS["A_cryo"]
    tau_cryo = np.arange(1, a_cryo["n_tau"] + 1) * a_cryo["dt"]
    cfg_cryo = pfb.make_gen_cfg("A_cryo", a_cryo["b_field"])
    smax = int(np.floor(tau_cryo[-1] / target_period(0, 30e3, a_cryo["b_field"])))
    model_cryo, builder_cryo, _ = pfb.train_pf("pf_cryo", tau_cryo, cfg_cryo,
                                               smax, DEVICE, 15000, 25, 0)

    m_cryo, det_cryo, _ = detect_v2("A_cryo", model_cryo, builder_cryo,
                                    tau_cryo, a_cryo["b_field"], 25, 2e3, 40000)
    m_room, det_room, _ = detect_v2("B_room", model_room, builder_room,
                                    tau_room, a_room["b_field"], 15, 3e3, 30000)

    plt.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.2),
                             gridspec_kw={"width_ratios": [1.5, 1]})

    for row, (arm_name, tau, m_recs, det, n_pulse, zoom, tol) in enumerate([
        ("저온 트윈 (N=32, 403.55 G — 그들의 조건)", tau_cryo, m_cryo,
         det_cryo, 32, (0, 12), 2e3),
        ("상온 트윈 (N=16 채널, 440.1 G — 우리 조건)", tau_room, m_room,
         det_room, 16, (0, 12), 4e3),
    ]):
        ax = axes[row, 0]
        ch = 0 if row == 0 else 1
        m_fit = cpmg_M(tau, det, n_pulse, bv2.ARMS["A_cryo" if row == 0 else "B_room"]["b_field"])
        rmse = np.sqrt(np.mean((m_fit - m_recs[ch]) ** 2))
        ax.plot(tau / US, m_recs[ch], ".", ms=1.8, color="0.55", alpha=0.7,
                label="50-스핀 공개 배스 트윈 신호")
        ax.plot(tau / US, m_fit, "-", lw=1.0, color="tab:red",
                label=f"검출 스핀 {len(det)}개의 forward model (RMSE {rmse:.3f})")
        ax.set_xlim(*zoom)
        ax.set_ylabel("M")
        ax.set_title(arm_name, fontsize=10, loc="left")
        ax.legend(fontsize=8, loc="lower right")
        if row == 1:
            ax.set_xlabel("τ (µs)")

        tp, fp, fn, _ = bv2.match_score(det[:, 0], GT[:, 0], tol)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        ax2 = axes[row, 1]
        ladder(ax2, det[:, 0], tol)
        ax2.set_title(f"GT 대조: {tp}/{len(GT)} 회수 · 오탐 {fp} "
                      f"(P {prec:.2f} / R {rec:.2f})", fontsize=10, loc="left")

    fig.suptitle("공개 50-스핀 실측 배스(van de Stolpe 2024, 4TU)의 디지털 트윈 검증 — "
                 "신호는 공개 실측값으로 합성, 스핀 값은 학습에 미사용(테스트 전용)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "figs" / "26_twin_validation.png", dpi=150)
    print("saved 26_twin_validation.png", flush=True)


if __name__ == "__main__":
    main()
