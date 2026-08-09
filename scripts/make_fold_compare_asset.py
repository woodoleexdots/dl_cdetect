"""Window-folding comparison: cryo twin vs room twin of the 50-spin bath.

Three rows x two columns:
  row 1: cryo twin, single N=32 channel      (crisp vertical stripe)
  row 2: room twin, single N=16 channel      (stripe buried in noise)
  row 3: room twin, mean of N=8/16/20 images (stripe partially recovered
         -- the reason our detector takes all three channels)
  col 1: window centered on a real spin (A=-36.3 kHz, A_perp=26.6 kHz)
  col 2: empty window (A=+52 kHz)
Under each folded image: the column-mean dip profile.

Output: results/slides_assets/fold_compare.png
Numpy-only (no torch): safe to run on a laptop.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpmg.physics import cpmg_M, stretched_exp, target_period
from cpmg.represent import slice_stack

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "results" / "slides_assets"

B_CRYO = 403.553
B_OURS = 440.1
B_REPR = 30e3
WIDTH = 53

plt.rcParams["font.family"] = ["Noto Sans CJK KR", "Apple SD Gothic Neo",
                               "AppleGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ARMS = {
    "cryo": dict(b_field=B_CRYO, n_pulses=(32,), dt=4e-9, n_tau=7000,
                 sigma=0.02, t2_us=(800, 1500), stretch=(0.8, 1.2),
                 contrast=(0.92, 1.0)),
    "room": dict(b_field=B_OURS, n_pulses=(8, 16, 20), dt=20e-9, n_tau=700,
                 sigma=0.06, t2_us=(150, 400), stretch=(0.4, 1.0),
                 contrast=(0.75, 0.95)),
}


def twin_signals(arm, bath, seed):
    """Replicates benchmark_v2.make_arm_dataset (numpy only)."""
    a = ARMS[arm]
    rng = np.random.default_rng(seed)
    tau = np.arange(1, a["n_tau"] + 1) * a["dt"]
    m_recs = []
    for n_pulse in a["n_pulses"]:
        m = cpmg_M(tau, bath, n_pulse, a["b_field"])
        t2 = rng.uniform(*a["t2_us"]) * 1e-6
        st = rng.uniform(*a["stretch"])
        a0 = rng.uniform(*a["contrast"])
        env = stretched_exp(tau, t2, st)
        px = 0.5 + 0.5 * a0 * m * env + rng.normal(0, a["sigma"], len(tau))
        m_recs.append(np.clip((2 * px - 1) / np.maximum(a0 * env, 1e-3),
                              -1.5, 1.5))
    return tau, m_recs


def fold(tau, m_rec, a_center, b_field, b_perp=B_REPR):
    tp = target_period(a_center, b_perp, b_field)
    return np.nan_to_num(
        slice_stack(tau, m_rec, tp, width=WIDTH, tau_start=tau[0]), nan=1.0)


def main():
    d = np.load(ROOT / "dataset" / "delft_public" / "bath50.npz")
    bath = np.column_stack([d["a_par"], d["a_perp"]])
    box = (np.abs(bath[:, 0]) <= 60e3) & (bath[:, 1] >= 5e3)

    cand = bath[box & (np.abs(bath[:, 0]) > 25e3)]
    spin = cand[np.argmax(cand[:, 1])]
    a_spin, b_spin = spin
    a_empty = 52e3
    assert np.min(np.abs(bath[:, 0] - a_empty)) > 4e3

    tau_c, m_cryo = twin_signals("cryo", bath, seed=7)
    tau_r, m_room = twin_signals("room", bath, seed=7)

    def imgs_for(a_c, b_c):
        i_cryo = fold(tau_c, m_cryo[0], a_c, B_CRYO, b_c)
        i_r16 = fold(tau_r, m_room[1], a_c, B_OURS, b_c)
        i_r3 = np.mean([fold(tau_r, m, a_c, B_OURS, b_c) for m in m_room],
                       axis=0)
        return [i_cryo, i_r16, i_r3]

    cols = [
        (f"스핀 있는 윈도우  A∥ = {a_spin/1e3:+.1f} kHz (A⊥ = {b_spin/1e3:.0f} kHz)",
         imgs_for(a_spin, b_spin)),
        (f"스핀 없는 윈도우  A∥ = {a_empty/1e3:+.0f} kHz",
         imgs_for(a_empty, B_REPR)),
    ]
    row_labels = ["저온 트윈\nN=32 1채널\nσ=0.02",
                  "상온 트윈\nN=16 1채널\nσ=0.06",
                  "상온 트윈\nN=8/16/20\n3채널 평균"]

    fig = plt.figure(figsize=(13.0, 9.2))
    gs = fig.add_gridspec(6, 2, height_ratios=[2.4, 1, 2.4, 1, 2.4, 1],
                          hspace=0.35, wspace=0.15,
                          left=0.11, right=0.985, top=0.90, bottom=0.055)

    for c, (col_title, imgs) in enumerate(cols):
        for r, img in enumerate(imgs):
            ax = fig.add_subplot(gs[2 * r, c])
            ax.imshow(img, cmap="gray", aspect="auto", vmin=-0.6, vmax=1.3)
            if r == 0:
                ax.set_title(col_title, fontsize=11.5, pad=10)
            if c == 0 and r == 0:
                ax.annotate("스핀 줄무늬 (세로)", xy=(26, 4), xytext=(36, 2),
                            color="tab:red", fontsize=10, fontweight="bold",
                            arrowprops=dict(arrowstyle="->", color="tab:red"))
            if c == 0 and r == 1:
                ax.annotate("이웃 스핀 간섭 (대각선)", xy=(17, 6), xytext=(28, 2.2),
                            color="tab:blue", fontsize=10, fontweight="bold",
                            arrowprops=dict(arrowstyle="->", color="tab:blue"))
            ax.set_ylabel(f"{img.shape[0]}주기", fontsize=8)
            ax.set_xticks([])
            ax.tick_params(labelsize=7)

            axp = fig.add_subplot(gs[2 * r + 1, c])
            prof = img.mean(axis=0)
            axp.plot(prof, color="tab:red", lw=1.4)
            axp.axhline(1.0, color="0.6", ls=":", lw=0.8)
            axp.set_ylim(min(0.3, prof.min() - 0.05), 1.15)
            axp.set_ylabel("열 평균", fontsize=8)
            axp.set_xticks([] if r < 2 else axp.get_xticks())
            if r == 2:
                axp.set_xlabel("윈도우 내 위상 (53열)", fontsize=9)
            axp.tick_params(labelsize=7)
            axp.set_xlim(0, WIDTH - 1)

    for r, lbl in enumerate(row_labels):
        top, bot = gs[2 * r, 0].get_position(fig), gs[2 * r + 1, 0].get_position(fig)
        y = (top.y1 + bot.y0) / 2
        fig.text(0.008, y, lbl, fontsize=10, fontweight="bold", va="center")

    fig.suptitle("주기 접기 비교 — 같은 50-스핀 배스를 조건별로 접었을 때:  저온 1채널은 스핀 줄무늬(세로)가 선명해 윈도우 단독 판정 가능 ·\n"
                 "상온은 이웃 스핀 간섭(대각선)과 노이즈에 묻히고, 3채널 평균은 노이즈만 줄일 뿐 간섭은 남음 → 윈도우 간 어텐션(회의)이 필요한 이유",
                 fontsize=12)
    fig.savefig(ASSETS / "fold_compare.png", dpi=150)
    print("saved ->", ASSETS / "fold_compare.png")


if __name__ == "__main__":
    main()
