"""Head-to-head benchmark: paper-style HPC classifier vs old-cdetect DE fit.

Protocol
--------
Generate synthetic NV1-like datasets (700 tau pts, N=8/16/20, envelope +
noise) with KNOWN ground-truth spins in the weak-coupling band, then:

  method A (paper style) : slice-stack image per A window -> trained HPCNet
                           bank -> P(spin) peaks = detected A list
  method B (old cdetect) : sequential-greedy DE fit with BIC selection
                           -> fitted (A, B) list

Score both against ground truth (match tolerance on A), then run both on
the real NV1 data and compare candidate lists.

Usage:
  python scripts/benchmark_compare.py --n-datasets 6 --hpc-dir results/hpc
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpmg.datagen import GenConfig
from cpmg.defit import greedy_de_fit
from cpmg.models import HPCNet
from cpmg.physics import cpmg_M, stretched_exp, target_period
from cpmg.represent import slice_stack

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "exp_dataset"


def experimental_tau():
    return pd.read_excel(DATA_DIR / "CPMG_NV1.xlsx")["a"].to_numpy(float)


def make_gt_dataset(rng, tau, cfg: GenConfig, n_spins, a_lim=50e3, min_sep=6e3):
    """Synthetic 3-channel dataset with known ground-truth spins."""
    while True:
        a = rng.uniform(-a_lim, a_lim, size=n_spins)
        if n_spins < 2 or np.min(np.diff(np.sort(a))) > min_sep:
            break
    b = rng.uniform(15e3, 50e3, size=n_spins)
    gt = np.column_stack([a, b])

    m_recs = []
    for n_pulse in cfg.n_pulses:
        m = cpmg_M(tau, gt, n_pulse, cfg.b_field_g)
        t2 = rng.uniform(150, 400) * 1e-6
        st = rng.uniform(0.4, 1.0)
        a0 = rng.uniform(0.75, 0.95)
        sigma = 0.05
        env = stretched_exp(tau, t2, st)
        px = 0.5 + 0.5 * a0 * m * env + rng.normal(0, sigma, len(tau))
        m_rec = np.clip((2 * px - 1) / np.maximum(a0 * env, 1e-3), -1.5, 1.5)
        m_recs.append(m_rec)
    return gt, np.array(m_recs)


def hpc_detect(tau, m_recs, model_dir: Path, cfg: GenConfig, device,
               height=0.6, prominence=0.1):
    """Run the trained HPC model bank -> peak A candidates (Hz)."""
    model_files = sorted(model_dir.glob("hpc_A*.pt"))
    a_list, p_list = [], []
    for f in model_files:
        a_khz = float(f.stem.replace("hpc_A", "").replace("kHz", ""))
        tp = target_period(a_khz * 1e3, cfg.b_repr, cfg.b_field_g)
        imgs = [
            np.nan_to_num(slice_stack(tau, m, tp, width=cfg.image_width,
                                      tau_start=tau[0]), nan=1.0)
            for m in m_recs
        ]
        x = torch.from_numpy(np.stack(imgs).astype(np.float32)[None]).to(device)
        model = HPCNet(in_ch=len(m_recs)).to(device)
        model.load_state_dict(torch.load(f, map_location=device))
        model.eval()
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
        a_list.append(a_khz * 1e3)
        p_list.append(1.0 - probs[0])
    a_arr, p_arr = np.array(a_list), np.array(p_list)
    order = np.argsort(a_arr)
    a_arr, p_arr = a_arr[order], p_arr[order]
    pk, _ = find_peaks(p_arr, height=height, prominence=prominence)
    return a_arr[pk], (a_arr, p_arr)


def match_score(detected_a, gt_a, tol=4e3):
    """Greedy one-to-one matching within tolerance -> (tp, fp, fn, errors)."""
    detected = list(detected_a)
    gt = list(gt_a)
    tp, errors = 0, []
    for g in sorted(gt):
        if not detected:
            break
        d = min(detected, key=lambda x: abs(x - g))
        if abs(d - g) <= tol:
            tp += 1
            errors.append(abs(d - g))
            detected.remove(d)
    fp = len(detected_a) - tp
    fn = len(gt_a) - tp
    return tp, fp, fn, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-datasets", type=int, default=6)
    ap.add_argument("--hpc-dir", default=str(ROOT / "results" / "hpc"))
    ap.add_argument("--max-spins", type=int, default=7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "results" / "benchmark_compare.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = GenConfig()
    tau = experimental_tau()
    rng = np.random.default_rng(args.seed)
    model_dir = Path(args.hpc_dir)

    results = []
    for i in range(args.n_datasets):
        n_spins = int(rng.integers(3, 6))
        gt, m_recs = make_gt_dataset(rng, tau, cfg, n_spins)
        gt_a = gt[:, 0]
        print(f"\n=== dataset {i+1}/{args.n_datasets}: {n_spins} GT spins "
              f"A(kHz)={[round(x, 1) for x in sorted(gt_a / 1e3)]} ===", flush=True)

        t0 = time.time()
        hpc_a, _ = hpc_detect(tau, m_recs, model_dir, cfg, device)
        t_hpc = time.time() - t0
        tp, fp, fn, err = match_score(hpc_a, gt_a)
        hpc_res = dict(method="hpc", dataset=i, n_gt=n_spins, tp=tp, fp=fp, fn=fn,
                       mean_err_khz=float(np.mean(err) / 1e3) if err else None,
                       detected_khz=[round(x / 1e3, 1) for x in hpc_a],
                       runtime_s=t_hpc)
        print(f"  HPC : det={hpc_res['detected_khz']}  tp={tp} fp={fp} fn={fn}  "
              f"({t_hpc:.1f}s)", flush=True)

        t0 = time.time()
        de = greedy_de_fit(tau, m_recs, cfg.n_pulses, cfg.b_field_g,
                           max_spins=args.max_spins, seed=args.seed + i,
                           verbose=False)
        t_de = time.time() - t0
        de_a = np.array([s[0] for s in de["best"]["spins"]])
        tp, fp, fn, err = match_score(de_a, gt_a)
        de_res = dict(method="de", dataset=i, n_gt=n_spins, tp=tp, fp=fp, fn=fn,
                      mean_err_khz=float(np.mean(err) / 1e3) if err else None,
                      detected_khz=[round(x / 1e3, 1) for x in sorted(de_a)],
                      best_k=de["best"]["k"], runtime_s=t_de)
        print(f"  DE  : det={de_res['detected_khz']} (k*={de['best']['k']})  "
              f"tp={tp} fp={fp} fn={fn}  ({t_de:.1f}s)", flush=True)

        results.append(dict(gt_khz=[round(x / 1e3, 1) for x in sorted(gt_a)],
                            hpc=hpc_res, de=de_res))

    # aggregate
    def agg(key):
        rows = [r[key] for r in results]
        tp = sum(r["tp"] for r in rows)
        fp = sum(r["fp"] for r in rows)
        fn = sum(r["fn"] for r in rows)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        errs = [r["mean_err_khz"] for r in rows if r["mean_err_khz"] is not None]
        return dict(tp=tp, fp=fp, fn=fn, precision=round(prec, 3),
                    recall=round(rec, 3),
                    mean_err_khz=round(float(np.mean(errs)), 2) if errs else None,
                    mean_runtime_s=round(float(np.mean([r["runtime_s"] for r in rows])), 1))

    summary = dict(hpc=agg("hpc"), de=agg("de"))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    # real NV1 comparison with the DE method
    from cpmg.represent import envelope_normalize
    nv1 = pd.read_excel(DATA_DIR / "CPMG_NV1.xlsx")
    m_exp = np.array([envelope_normalize(tau, nv1[c].to_numpy(float))[0]
                      for c in ["CPMG8", "CPMG16", "CPMG20"]])
    print("\n=== real NV1: greedy DE fit ===", flush=True)
    de_exp = greedy_de_fit(tau, m_exp, cfg.n_pulses, cfg.b_field_g,
                           max_spins=args.max_spins, seed=args.seed, verbose=True)
    nv1_spins = [[round(a / 1e3, 1), round(b / 1e3, 1)] for a, b in de_exp["best"]["spins"]]
    print(f"NV1 DE best k={de_exp['best']['k']}  spins(A,B kHz)={nv1_spins}")

    out = dict(per_dataset=results, summary=summary,
               nv1_de=dict(k=de_exp["best"]["k"], spins_khz=nv1_spins,
                           trace=[dict(k=t["k"], bic=round(t["bic"], 1)) for t in de_exp["trace"]]))
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
