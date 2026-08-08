"""Generate HPC training datasets over an A-window grid.

Usage:
  python scripts/gen_training_data.py --a-min -60 --a-max 60 --a-step 2 \
      --n-per-class 300 --outdir results/train_data
  python scripts/gen_training_data.py --windows -58 -40 -13 2 --n-per-class 150
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpmg.datagen import GenConfig, gen_window_dataset

DATA_DIR = Path(__file__).resolve().parents[1] / "dataset" / "exp_dataset"


def experimental_tau() -> np.ndarray:
    nv1 = pd.read_excel(DATA_DIR / "CPMG_NV1.xlsx")
    return nv1["a"].to_numpy(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-min", type=float, default=-60.0, help="kHz")
    ap.add_argument("--a-max", type=float, default=60.0, help="kHz")
    ap.add_argument("--a-step", type=float, default=2.0, help="kHz")
    ap.add_argument("--windows", type=float, nargs="*", default=None,
                    help="explicit window centers in kHz (overrides grid)")
    ap.add_argument("--n-per-class", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parents[1] / "results" / "train_data"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tau = experimental_tau()
    cfg = GenConfig()

    if args.windows is not None:
        centers_khz = np.array(args.windows, dtype=float)
    else:
        centers_khz = np.arange(args.a_min, args.a_max + 1e-9, args.a_step)

    tic = time.time()
    for i, a_khz in enumerate(centers_khz):
        X, y = gen_window_dataset(
            tau, a_khz * 1e3, args.n_per_class, cfg, seed=args.seed + i
        )
        fname = outdir / f"window_A{a_khz:+08.1f}kHz.npz"
        np.savez_compressed(fname, X=X, y=y, a_center_hz=a_khz * 1e3,
                            tau=tau, n_pulses=np.array(cfg.n_pulses))
        print(f"[{i+1}/{len(centers_khz)}] {fname.name}  X={X.shape}  "
              f"({time.time()-tic:.1f}s elapsed)", flush=True)

    print(f"done: {len(centers_khz)} windows -> {outdir}")


if __name__ == "__main__":
    main()
