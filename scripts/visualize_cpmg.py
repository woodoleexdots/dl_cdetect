"""First-look visualization of the experimental NV CPMG data.

Produces, per dataset (NV1: CPMG-8/16/20, NV2: CPMG-16):
  1. raw signal + fitted decoherence envelope
  2. periodicity check (autocorrelation vs expected weak-coupling TP)
  3. slice-stack 2D images for a few candidate A values
  4. period-scan contrast trace over a weak-coupling A grid (poor-man's HPC)

Usage:
  python scripts/visualize_cpmg.py [--outdir results/figs]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpmg.physics import B_FIELD_G, cpmg_M, larmor_omega, target_period
from cpmg.represent import envelope_normalize, period_scan_contrast, slice_stack

DATA_DIR = Path(__file__).resolve().parents[1] / "dataset" / "exp_dataset"

US = 1e-6


def load_datasets():
    nv1 = pd.read_excel(DATA_DIR / "CPMG_NV1.xlsx")
    nv2 = pd.read_excel(DATA_DIR / "CPMG_NV2.xlsx")
    tau1 = nv1["a"].to_numpy(float)
    tau2 = nv2["Time"].to_numpy(float)
    sets = [
        ("NV1", 8, tau1, nv1["CPMG8"].to_numpy(float)),
        ("NV1", 16, tau1, nv1["CPMG16"].to_numpy(float)),
        ("NV1", 20, tau1, nv1["CPMG20"].to_numpy(float)),
        ("NV2", 16, tau2, nv2["CPMG16"].to_numpy(float)),
    ]
    return sets


def autocorr_period(tau, m_rec):
    """Dominant dip spacing from the autocorrelation of the detrended signal."""
    x = m_rec - np.nanmean(m_rec)
    x = np.nan_to_num(x)
    ac = np.correlate(x, x, mode="full")[len(x) - 1 :]
    ac /= ac[0]
    dt = np.median(np.diff(tau))
    # first local maximum after the zero-lag peak decays
    lo = int(0.3e-6 / dt)  # ignore lags < 0.3 us
    hi = int(3e-6 / dt)
    lag = lo + np.argmax(ac[lo:hi])
    return lag * dt, ac, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parents[1] / "results" / "figs"))
    ap.add_argument("--width", type=int, default=53)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sets = load_datasets()
    wL_hz = larmor_omega(B_FIELD_G) / (2 * np.pi)
    tp0 = target_period(0.0, 25e3, B_FIELD_G)
    print(f"B = {B_FIELD_G} G  ->  f_L = {wL_hz/1e3:.2f} kHz, weak-coupling TP ~ {tp0/US:.4f} us")

    # ---------------- 1. raw + envelope ----------------
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    env_params = {}
    m_recs = {}
    for ax, (nv, n_pulse, tau, px) in zip(axes, sets):
        m_rec, (T2, n_exp) = envelope_normalize(tau, px)
        env_params[(nv, n_pulse)] = (T2, n_exp)
        m_recs[(nv, n_pulse)] = (tau, m_rec)
        ax.plot(tau / US, px, lw=0.6, color="tab:blue")
        tt = np.linspace(tau[0], tau[-1], 400)
        amp_env = 0.5 * np.exp(-((tt / T2) ** n_exp))
        ax.plot(tt / US, 0.5 + amp_env, "r--", lw=1.2, label=f"envelope T2={T2/US:.1f} us, n={n_exp:.2f}")
        ax.plot(tt / US, 0.5 - amp_env, "r--", lw=1.2)
        ax.set_ylabel("Px")
        ax.set_title(f"{nv} CPMG-{n_pulse}", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("tau (us)")
    fig.suptitle("Raw CPMG signals + stretched-exp envelope fit (Eq. 5)")
    fig.tight_layout()
    fig.savefig(outdir / "01_raw_envelope.png", dpi=150)
    plt.close(fig)

    # ---------------- 2. periodicity check ----------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, (nv, n_pulse, tau, px) in zip(axes.ravel(), sets):
        tau_r, m_rec = m_recs[(nv, n_pulse)]
        period_obs, ac, dt = autocorr_period(tau_r, m_rec)
        lags = np.arange(len(ac)) * dt / US
        ax.plot(lags, ac, lw=0.8)
        ax.axvline(tp0 / US, color="r", ls="--", lw=1, label=f"expected TP={tp0/US:.3f} us")
        ax.axvline(period_obs / US, color="g", ls=":", lw=1.5, label=f"observed {period_obs/US:.3f} us")
        ax.set_xlim(0, 5)
        ax.set_title(f"{nv} CPMG-{n_pulse}", fontsize=10)
        ax.set_xlabel("lag (us)")
        ax.set_ylabel("autocorr")
        ax.legend(fontsize=8)
    fig.suptitle("Dip-spacing check: observed autocorrelation vs weak-coupling TP")
    fig.tight_layout()
    fig.savefig(outdir / "02_periodicity.png", dpi=150)
    plt.close(fig)

    # ---------------- 3. slice-stack images ----------------
    a_examples_khz = [-40, -20, 0, 20, 40]
    b_repr = 25e3
    for nv, n_pulse, tau, px in sets:
        tau_r, m_rec = m_recs[(nv, n_pulse)]
        fig, axes = plt.subplots(1, len(a_examples_khz), figsize=(16, 4.2), sharey=False)
        for ax, a_khz in zip(axes, a_examples_khz):
            tp = target_period(a_khz * 1e3, b_repr, B_FIELD_G)
            img = slice_stack(tau_r, m_rec, tp, width=args.width)
            im = ax.imshow(
                img, aspect="auto", origin="lower", cmap="viridis",
                extent=[0, tp / US, 0, img.shape[0]], vmin=-1.0, vmax=1.2,
            )
            ax.set_title(f"A={a_khz} kHz\nTP={tp/US:.4f} us", fontsize=9)
            ax.set_xlabel("tau mod TP (us)")
        axes[0].set_ylabel("slice #")
        fig.colorbar(im, ax=axes, shrink=0.8, label="M (env-normalized)")
        fig.suptitle(f"{nv} CPMG-{n_pulse}: slice-stack images (B_repr={b_repr/1e3:.0f} kHz)")
        fig.savefig(outdir / f"03_slicestack_{nv}_N{n_pulse}.png", dpi=150)
        plt.close(fig)

    # ---------------- 4. period-scan contrast ----------------
    a_grid = np.arange(-60e3, 60e3 + 1, 500.0)
    fig, axes = plt.subplots(len(sets), 1, figsize=(12, 10), sharex=True)
    scan_results = {}
    for ax, (nv, n_pulse, tau, px) in zip(axes, sets):
        tau_r, m_rec = m_recs[(nv, n_pulse)]
        contrast = period_scan_contrast(tau_r, m_rec, a_grid, b_repr, B_FIELD_G, width=args.width)
        scan_results[(nv, n_pulse)] = contrast
        ax.plot(a_grid / 1e3, contrast, lw=0.9)
        ax.set_ylabel("contrast")
        ax.set_title(f"{nv} CPMG-{n_pulse}", fontsize=10)
    axes[-1].set_xlabel("candidate A (kHz)   [B_repr = 25 kHz]")
    fig.suptitle("Period-scan vertical-line contrast (peaks = candidate 13C spins)")
    fig.tight_layout()
    fig.savefig(outdir / "04_period_scan.png", dpi=150)
    plt.close(fig)
    np.savez(
        outdir / "period_scan.npz",
        a_grid=a_grid,
        **{f"{nv}_N{n}": c for (nv, n), c in scan_results.items()},
    )

    # ---------------- 5. synthetic sanity check ----------------
    # what a single weak spin looks like under OUR sampling (700 pts, 20 ns)
    nv, n_pulse, tau, px = sets[1]  # NV1 CPMG-16 grid
    ab = np.array([[20e3, 25e3]])
    m_syn = cpmg_M(tau, ab, n_pulse)
    rng = np.random.default_rng(0)
    T2, n_exp = env_params[("NV1", 16)]
    px_syn = 0.5 + 0.5 * m_syn * np.exp(-((tau / T2) ** n_exp)) + rng.normal(0, 0.05, len(tau))
    m_syn_rec, _ = envelope_normalize(tau, px_syn)

    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(2, 3)
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(tau / US, px_syn, lw=0.6, label="synthetic Px (A=20, B=25 kHz + noise 0.05)")
    ax0.plot(tau / US, 0.5 + 0.5 * m_syn, lw=0.8, alpha=0.7, label="noiseless")
    ax0.set_xlabel("tau (us)")
    ax0.legend(fontsize=8)
    for j, a_khz in enumerate([0, 20, 40]):
        ax = fig.add_subplot(gs[1, j])
        tp = target_period(a_khz * 1e3, b_repr, B_FIELD_G)
        img = slice_stack(tau, m_syn_rec, tp, width=args.width)
        ax.imshow(img, aspect="auto", origin="lower", cmap="viridis",
                  extent=[0, tp / US, 0, img.shape[0]], vmin=-1.0, vmax=1.2)
        ax.set_title(f"stacked at A={a_khz} kHz", fontsize=9)
        ax.set_xlabel("tau mod TP (us)")
    fig.suptitle(f"Synthetic single spin (A=20, B=25 kHz), N=16, our sampling — "
                 f"vertical line must appear only at A=20")
    fig.tight_layout()
    fig.savefig(outdir / "05_synthetic_check.png", dpi=150)
    plt.close(fig)

    print(f"figures written to {outdir}")
    for (nv, n), (T2, n_exp) in env_params.items():
        print(f"  {nv} N={n}: T2 = {T2/US:.2f} us, stretch n = {n_exp:.2f}")


if __name__ == "__main__":
    main()
