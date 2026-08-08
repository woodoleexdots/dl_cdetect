"""EDA of the published Delft 50-spin bath (van de Stolpe 2024, 4TU dataset).

Questions answered:
  - how many nuclear spins, and with what (A_par, A_perp) distribution
  - local-period (TP) distribution at the cryo field (403.553 G)
  - per-spin CPMG dip depth under cryo (N=32, 28 us) vs room-temp
    (N=16, 14 us) measurement windows -> detectability vs noise floor
  - nuclear-nuclear coupling distribution (model-mismatch scale for CCE-1)

Output: results/figs/14_eda_delft50.png + printed statistics.
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

from cpmg.physics import cpmg_M, larmor_omega, target_period

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "delft_public"
FIGS = ROOT / "results" / "figs"

B_CRYO = 403.553   # Gauss, Jung 2021 / Delft conditions
B_OURS = 440.1

def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    params = json.loads((DATA / "data_fig2_spin_params.json").read_text())
    nitrogen = params.pop("N", None)
    labels = list(params.keys())
    par = np.array([params[k]["par"] for k in labels])    # Hz
    perp = np.array([params[k]["perp"] for k in labels])  # Hz
    n_spins = len(labels)

    print(f"=== Delft 50-spin bath EDA ===")
    print(f"13C spins: {n_spins}   (+ nitrogen: {nitrogen})")
    print(f"A_par : min {par.min()/1e3:+.1f} / median {np.median(par)/1e3:+.1f} "
          f"/ max {par.max()/1e3:+.1f} kHz;  |A_par| median {np.median(abs(par))/1e3:.1f} kHz")
    print(f"A_perp: min {perp.min()/1e3:.2f} / median {np.median(perp)/1e3:.1f} "
          f"/ max {perp.max()/1e3:.1f} kHz")
    for th in [2e3, 5e3, 10e3, 20e3]:
        print(f"  A_perp >= {th/1e3:>4.0f} kHz : {np.sum(perp >= th):2d} spins")
    in_box = (np.abs(par) <= 60e3) & (perp >= 5e3)
    print(f"search box |A_par|<=60 kHz & A_perp>=5 kHz : {in_box.sum()} spins")
    strong = np.abs(par) > 60e3
    print(f"outside |A_par|>60 kHz: {strong.sum()} spins "
          f"({[f'{labels[i]}:{par[i]/1e3:+.0f}k' for i in np.where(strong)[0]]})")

    # ---- per-spin dip depth under the two measurement windows ----
    tau_cryo = np.arange(1, 7001) * 4e-9          # 4 ns x 7000 -> 28 us
    tau_room = np.arange(1, 701) * 20e-9          # 20 ns x 700 -> 14 us
    depth_cryo, depth_room = [], []
    for a, b in zip(par, perp):
        m32 = cpmg_M(tau_cryo, [[a, b]], 32, B_CRYO)
        m16 = cpmg_M(tau_room, [[a, b]], 16, B_OURS)
        depth_cryo.append(0.5 * (1 - m32.min()))
        depth_room.append(0.5 * (1 - m16.min()))
    depth_cryo = np.array(depth_cryo)
    depth_room = np.array(depth_room)
    for name, d, sig in [("cryo  N=32/28us/4ns", depth_cryo, 0.02),
                         ("room  N=16/14us/20ns", depth_room, 0.06)]:
        print(f"{name}: detectable (dip > 2*sigma={2*sig}) : "
              f"{np.sum(d > 2*sig)}/{n_spins} spins")

    # ---- nuclear-nuclear couplings ----
    cpl = pd.read_excel(DATA / "spin_measured_couplings.xlsx", index_col=0)
    cvals = np.abs(cpl.to_numpy(float))
    cvals = cvals[np.isfinite(cvals) & (cvals > 0)]
    print(f"measured nuclear-nuclear couplings: n={len(cvals)}, "
          f"median {np.median(cvals):.2f} Hz, max {cvals.max():.1f} Hz")

    # ---------------- figure ----------------
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3)

    ax = fig.add_subplot(gs[0, 0])
    sc = ax.scatter(par / 1e3, perp / 1e3, c=depth_room, s=28, cmap="viridis")
    ax.axvspan(-60, 60, color="tab:blue", alpha=0.06)
    ax.axhline(5, color="r", ls=":", lw=1)
    ax.set_xlabel("A_par (kHz)")
    ax.set_ylabel("A_perp (kHz)")
    ax.set_title(f"{n_spins} published spins (color: room-temp dip depth)", fontsize=9)
    fig.colorbar(sc, ax=ax, shrink=0.8)

    ax = fig.add_subplot(gs[0, 1])
    ax.hist(par / 1e3, bins=30, color="tab:blue", alpha=0.8)
    ax.set_xlabel("A_par (kHz)")
    ax.set_title("A_par distribution", fontsize=9)

    ax = fig.add_subplot(gs[0, 2])
    ax.hist(perp / 1e3, bins=30, color="tab:green", alpha=0.8)
    ax.axvline(5, color="r", ls=":", lw=1, label="A_perp=5 kHz")
    ax.set_xlabel("A_perp (kHz)")
    ax.set_title("A_perp distribution", fontsize=9)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    tp = np.array([target_period(a, b, B_CRYO) for a, b in zip(par, perp)])
    ax.hist(tp / 1e-6, bins=30, color="tab:purple", alpha=0.8)
    ax.set_xlabel("local period TP at 403.55 G (us)")
    ax.set_title("TP distribution (cryo field)", fontsize=9)

    ax = fig.add_subplot(gs[1, 1])
    order = np.argsort(-depth_cryo)
    ax.semilogy(np.arange(n_spins), depth_cryo[order], "o-", ms=3,
                label="cryo N=32, 28 us")
    ax.semilogy(np.arange(n_spins), np.sort(depth_room)[::-1], "s-", ms=3,
                label="room N=16, 14 us")
    ax.axhline(2 * 0.02, color="tab:blue", ls=":", lw=1, label="2*sigma cryo (0.04)")
    ax.axhline(2 * 0.06, color="tab:red", ls=":", lw=1, label="2*sigma room (0.12)")
    ax.set_xlabel("spin rank")
    ax.set_ylabel("max CPMG dip depth")
    ax.set_title("detectability: dip depth vs noise floor", fontsize=9)
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[1, 2])
    ax.hist(cvals, bins=40, color="tab:orange", alpha=0.85)
    ax.set_xlabel("|nuclear-nuclear coupling| (Hz)")
    ax.set_title(f"measured spin-spin couplings (n={len(cvals)})", fontsize=9)

    fig.suptitle("EDA: published Delft 50-spin bath (van de Stolpe 2024, same NV lineage as Jung 2021)")
    fig.tight_layout()
    fig.savefig(FIGS / "14_eda_delft50.png", dpi=150)
    print(f"figure -> {FIGS/'14_eda_delft50.png'}")

    # save the bath for benchmark v2
    np.savez(ROOT / "dataset" / "delft_public" / "bath50.npz",
             labels=labels, a_par=par, a_perp=perp)
    print("bath saved -> dataset/delft_public/bath50.npz")


if __name__ == "__main__":
    main()
