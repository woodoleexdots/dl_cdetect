"""Final spin tables for NV1 (wide grid) and NV2 (refined).

(1) NV1 +-120 kHz: PF NV1-wide curve (121 tokens) -> candidate regions
    (threshold 0.25 so the -86/-88 kHz feature is included) -> region-
    constrained greedy DE over the 3 channels.
(2) NV2 refined: threshold 0.15 (more regions), max_spins 12, wider B
    bounds, and a 2-round envelope refinement: after round 1 the
    stretched-exp envelope is re-fitted on (2Px-1)/M_fit using only
    points where the model is far from zero, then everything reruns.

Outputs: entries 'nv1_wide' and 'nv2_refined' in hybrid_results.json,
plus overlay figures 19/20.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpmg.datagen import GenConfig
from cpmg.hybrid import hybrid_greedy_fit, pf_candidate_regions
from cpmg.periodformer import PeriodFormer, TokenBuilder
from cpmg.physics import cpmg_M, stretched_exp
from cpmg.represent import envelope_normalize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "benchmark_v2"
B_OURS = 440.1


def spin_list(fit):
    return sorted([[round(s[0] / 1e3, 1), round(s[1] / 1e3, 1)]
                   for s in fit["best"]["spins"]])


def refit_envelope(tau, px, ab_hz, n_pulse):
    """Re-fit amp*exp(-(t/T2)^n) on (2Px-1)/M_fit where |M_fit| > 0.3."""
    from scipy.optimize import curve_fit

    m_fit = cpmg_M(tau, ab_hz, n_pulse, B_OURS)
    mask = np.abs(m_fit) > 0.3
    ratio = (2 * px[mask] - 1) / m_fit[mask]
    t = tau[mask]
    good = (ratio > 0.05) & (ratio < 2.0)

    def model(t, amp, T2, n):
        return amp * stretched_exp(t, T2, n)

    p0 = (0.8, 300e-6, 0.7)
    bounds = ([0.2, 20e-6, 0.2], [1.5, 5e-3, 2.5])
    (amp, T2, n), _ = curve_fit(model, t[good], ratio[good], p0=p0,
                                bounds=bounds, maxfev=20000)
    env = amp * stretched_exp(tau, T2, n)
    return np.clip((2 * px - 1) / np.maximum(env, 1e-3), -1.5, 1.5), (amp, T2, n)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = json.loads((OUT / "hybrid_results.json").read_text())

    # ---------------- (1) NV1 wide ----------------
    nv1 = pd.read_excel(ROOT / "dataset" / "exp_dataset" / "CPMG_NV1.xlsx")
    tau1 = nv1["a"].to_numpy(float)
    m1 = np.array([envelope_normalize(tau1, nv1[c].to_numpy(float))[0]
                   for c in ["CPMG8", "CPMG16", "CPMG20"]], dtype=np.float32)
    d = np.load(ROOT / "results" / "periodformer" / "curve_NV1_NV1_wide.npz")
    centers_w, p_w = d["centers_hz"], d["p"]
    regions1 = pf_candidate_regions(p_w, centers_w, thresh=0.25, margin_hz=3e3)
    print(f"NV1 wide regions (kHz): {[(round(l/1e3), round(h/1e3)) for l, h in regions1]}",
          flush=True)
    fit1 = hybrid_greedy_fit(tau1, m1, (8, 16, 20), B_OURS, regions1,
                             max_spins=14, b_bounds=(3e3, 65e3),
                             de_maxiter=60, de_popsize=20, seed=0, verbose=True)
    spins1 = spin_list(fit1)
    print(f"NV1 wide k*={fit1['best']['k']}: {spins1}", flush=True)
    res["nv1_wide"] = dict(
        regions_khz=[[round(l / 1e3, 1), round(h / 1e3, 1)] for l, h in regions1],
        k=fit1["best"]["k"], spins_khz=spins1)

    # ---------------- (2) NV2 refined ----------------
    nv2 = pd.read_excel(ROOT / "dataset" / "exp_dataset" / "CPMG_NV2.xlsx")
    tau2 = nv2["Time"].to_numpy(float)
    px2 = nv2["CPMG16"].to_numpy(float)
    centers2 = np.arange(-400, 400 + 1e-9, 5.0) * 1e3
    cfg2 = GenConfig(n_pulses=(16,), b_field_g=B_OURS, b_repr=185e3,
                     b_tgt_min=50e3, b_tgt_max=320e3)
    builder2 = TokenBuilder(tau2, centers2, cfg2, device, s_max=13)
    model2 = PeriodFormer(in_ch=1, n_tokens=len(centers2), s_max=13).to(device)
    model2.load_state_dict(torch.load(
        ROOT / "results" / "periodformer" / "periodformer_NV2_strong.pt",
        map_location=device))
    model2.eval()

    def pf_curve(m_recs):
        x = torch.from_numpy(np.asarray(m_recs, dtype=np.float32))[None].to(device)
        with torch.no_grad():
            return torch.sigmoid(model2(builder2(x)))[0].cpu().numpy()

    m2 = np.array([envelope_normalize(tau2, px2)[0]], dtype=np.float32)
    for rnd in range(2):
        p2 = pf_curve(m2)
        regions2 = pf_candidate_regions(p2, centers2, thresh=0.15, margin_hz=8e3)
        print(f"NV2 round {rnd+1} regions: "
              f"{[(round(l/1e3), round(h/1e3)) for l, h in regions2]}", flush=True)
        fit2 = hybrid_greedy_fit(tau2, m2, (16,), B_OURS, regions2,
                                 max_spins=12, b_bounds=(30e3, 350e3),
                                 de_maxiter=80, de_popsize=20, seed=rnd,
                                 verbose=True)
        if rnd == 0:
            ab = np.array(fit2["best"]["spins"])
            m_rec2, env_par = refit_envelope(tau2, px2, ab, 16)
            print(f"NV2 envelope refit: amp={env_par[0]:.3f} "
                  f"T2={env_par[1]*1e6:.0f}us n={env_par[2]:.2f}", flush=True)
            m2 = np.array([m_rec2], dtype=np.float32)
    spins2 = spin_list(fit2)
    rmse2 = float(np.sqrt(np.mean(
        (cpmg_M(tau2, np.array(fit2["best"]["spins"]), 16, B_OURS) - m2[0]) ** 2)))
    print(f"NV2 refined k*={fit2['best']['k']}: {spins2}  RMSE={rmse2:.3f}", flush=True)
    res["nv2_refined"] = dict(
        regions_khz=[[round(l / 1e3, 1), round(h / 1e3, 1)] for l, h in regions2],
        k=fit2["best"]["k"], spins_khz=spins2, rmse=round(rmse2, 3),
        envelope=dict(amp=round(env_par[0], 3), T2_us=round(env_par[1] * 1e6, 1),
                      n=round(env_par[2], 2)))

    (OUT / "hybrid_results.json").write_text(json.dumps(res, indent=2))
    print("saved ->", OUT / "hybrid_results.json", flush=True)

    # ---------------- overlays ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    US = 1e-6
    FIGS = ROOT / "results" / "figs"

    fig, axes = plt.subplots(3, 1, figsize=(14, 8.5))
    ab1 = np.array(spins1) * 1e3
    for ax, (n, col) in zip(axes, [(8, "CPMG8"), (16, "CPMG16"), (20, "CPMG20")]):
        m, _ = envelope_normalize(tau1, nv1[col].to_numpy(float))
        m_f = cpmg_M(tau1, ab1, n, B_OURS)
        rmse = np.sqrt(np.mean((m_f - m) ** 2))
        ax.plot(tau1 / US, m, ".", ms=2, color="0.55", alpha=0.8)
        ax.plot(tau1 / US, m_f, "-", lw=1.0, color="tab:red")
        ax.set_title(f"N={n}  RMSE={rmse:.3f}", fontsize=9, loc="right")
        ax.set_ylabel("M")
    axes[-1].set_xlabel("tau (us)")
    fig.suptitle(f"NV1 final (wide +-120 kHz) hybrid fit, k*={fit1['best']['k']}: "
                 + ", ".join(f"({a:+.0f},{b:.0f})" for a, b in spins1), fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "19_final_overlay_NV1.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6.5))
    ab2 = np.array(spins2) * 1e3
    m_f2 = cpmg_M(tau2, ab2, 16, B_OURS)
    for ax, zoom in zip(axes, [None, (0, 5)]):
        ax.plot(tau2 / US, m2[0], ".", ms=2.2, color="0.55", alpha=0.8)
        ax.plot(tau2 / US, m_f2, "-", lw=1.0, color="tab:red")
        if zoom:
            ax.set_xlim(*zoom)
        ax.set_ylabel("M")
    axes[0].set_title(f"full range  RMSE={rmse2:.3f}", fontsize=9, loc="left")
    axes[1].set_title("zoom 0-5 us", fontsize=9, loc="left")
    axes[1].set_xlabel("tau (us)")
    fig.suptitle(f"NV2 refined hybrid fit (2-round envelope), k*={fit2['best']['k']}: "
                 + ", ".join(f"({a:+.0f},{b:.0f})" for a, b in spins2), fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "20_final_overlay_NV2.png", dpi=150)
    plt.close(fig)
    print("figures 19/20 saved", flush=True)


if __name__ == "__main__":
    main()
