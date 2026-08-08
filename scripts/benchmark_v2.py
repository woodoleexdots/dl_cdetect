"""Benchmark v2: the published Delft 50-spin bath as ground truth.

Three arms, comparing the reproduced 2021 pipeline (MLP window bank) against
the user's cdetect method (sequential-greedy DE + BIC):

  A  cryo digital twin : B=403.553 G, N=32, tau=4ns*7000 (28 us), sigma=0.02
  B  room-temp         : B=440.1 G, N=8/16/20, tau=20ns*700 (14 us), sigma=0.06
                         (2021 bank uses its native single-N=16 input)
  C  model mismatch    : arm B + fitted (not known) envelope + wrong-field
                         analysis (dB=+1.5 G) + per-channel gain drift

Ground truth for scoring: the 27 in-box spins (|A_par|<=60 kHz, A_perp>=5 kHz).
Also reported: recall vs the arm's physically detectable subset.

Results cached in results/benchmark_v2/. Safe to re-run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cpmg.ablation import AblationConfig, eval_bank, train_bank
from cpmg.datagen import GenConfig
from cpmg.defit import greedy_de_fit
from cpmg.physics import cpmg_M, stretched_exp
from cpmg.represent import envelope_normalize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "benchmark_v2"

B_CRYO = 403.553
B_OURS = 440.1

ARMS = {
    "A_cryo": dict(b_field=B_CRYO, n_pulses=(32,), dt=4e-9, n_tau=7000,
                   sigma=0.02, t2_us=(800, 1500), stretch=(0.8, 1.2),
                   contrast=(0.92, 1.0), mismatch=False),
    "B_room": dict(b_field=B_OURS, n_pulses=(8, 16, 20), dt=20e-9, n_tau=700,
                   sigma=0.06, t2_us=(150, 400), stretch=(0.4, 1.0),
                   contrast=(0.75, 0.95), mismatch=False),
    "C_mismatch": dict(b_field=B_OURS, n_pulses=(8, 16, 20), dt=20e-9, n_tau=700,
                       sigma=0.06, t2_us=(150, 400), stretch=(0.4, 1.0),
                       contrast=(0.75, 0.95), mismatch=True),
}
DB_MISMATCH = 1.5  # Gauss analysis-field error in arm C


def load_bath():
    d = np.load(ROOT / "dataset" / "delft_public" / "bath50.npz")
    return np.column_stack([d["a_par"], d["a_perp"]]), list(d["labels"])


def gt_in_box(bath):
    m = (np.abs(bath[:, 0]) <= 60e3) & (bath[:, 1] >= 5e3)
    return bath[m]


def make_arm_dataset(arm, bath, seed):
    """One noise realization of the bath measured under the arm's conditions."""
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
        if a["mismatch"]:
            px = px * (1.0 + rng.normal(0, 0.01))  # per-channel gain drift
            m_rec, _ = envelope_normalize(tau, px)  # fitted, not known
        else:
            m_rec = np.clip((2 * px - 1) / np.maximum(a0 * env, 1e-3), -1.5, 1.5)
        m_recs.append(np.clip(m_rec, -1.5, 1.5))
    return tau, np.array(m_recs, dtype=np.float32)


def match_score(det_a, gt_a, tol):
    det = list(det_a)
    tp, errs = 0, []
    for g in sorted(gt_a):
        if not det:
            break
        d = min(det, key=lambda x: abs(x - g))
        if abs(d - g) <= tol:
            tp += 1
            errs.append(abs(d - g))
            det.remove(d)
    return tp, len(det_a) - tp, len(gt_a) - tp, errs


def agg(rows, gt_n):
    tp = sum(r[0] for r in rows)
    fp = sum(r[1] for r in rows)
    fn = sum(r[2] for r in rows)
    errs = [e for r in rows for e in r[3]]
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return dict(precision=round(prec, 3), recall=round(rec, 3),
                f1=round(2 * prec * rec / max(prec + rec, 1e-9), 3),
                mean_err_khz=round(float(np.mean(errs)) / 1e3, 2) if errs else None,
                tp=tp, fp=fp, fn=fn, gt_per_dataset=gt_n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=8, help="noise realizations/arm")
    ap.add_argument("--n-real-cryo", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    bath, labels = load_bath()
    gt = gt_in_box(bath)
    gt_a = gt[:, 0]
    print(f"bath: {len(bath)} spins, in-box GT: {len(gt)}", flush=True)
    centers_hz = np.arange(-60, 60 + 1e-9, 2.0) * 1e3

    results = {}
    for arm in ARMS:
        a = ARMS[arm]
        n_real = args.n_real_cryo if arm == "A_cryo" else args.n_real
        tol = 2e3 if arm == "A_cryo" else 4e3
        print(f"\n===== arm {arm} ({n_real} realizations, tol {tol/1e3:.0f} kHz) =====",
              flush=True)
        suite = []
        for i in range(n_real):
            tau, m_recs = make_arm_dataset(arm, bath, args.seed + 100 + i)
            suite.append(dict(sigma=a["sigma"], gt=gt, m_recs=m_recs,
                              n_pulses=a["n_pulses"]))

        # analysis field: in arm C both methods believe a slightly wrong field
        analysis_field = a["b_field"] + (DB_MISMATCH if a["mismatch"] else 0.0)

        # ---- method 1: reproduced 2021 pipeline (MLP window bank) ----
        mlp_pulses = (32,) if arm == "A_cryo" else (16,)
        cfg = AblationConfig(f"mlp2021-{arm}", arch="mlp2021", n_pulses=mlp_pulses)
        gen_cfg = GenConfig(n_pulses=mlp_pulses,
                            b_field_g=analysis_field,
                            b_tgt_min=5e3, b_tgt_max=60e3, b_repr=20e3,
                            noise_range=(max(0.01, a["sigma"] - 0.03),
                                         a["sigma"] + 0.05),
                            t2_range_us=a["t2_us"], stretch_range=a["stretch"],
                            contrast_range=a["contrast"])
        bankdir = OUT / f"bank_{arm}"
        marker = bankdir / "DONE"
        if not marker.exists():
            t0 = time.time()
            print(f"[2021] training bank for {arm} ...", flush=True)
            accs = train_bank(cfg, tau, centers_hz, bankdir, device,
                              n_per_class=500, epochs=15, seed=args.seed,
                              gen_cfg=gen_cfg)
            marker.write_text(json.dumps({"val_accs": accs,
                                          "train_s": round(time.time() - t0, 1)}))
        info = json.loads(marker.read_text())
        res_2021 = eval_bank(cfg, tau, centers_hz, bankdir, device, suite,
                             gen_cfg=gen_cfg)
        rows = [match_score(r["detected_a"], gt_a, tol) for r in res_2021]
        m2021 = agg(rows, len(gt_a))
        m2021["mean_val_acc"] = round(float(np.mean(info["val_accs"])), 3)
        print(f"[2021] {json.dumps(m2021)}", flush=True)

        # ---- method 2: cdetect greedy DE + BIC ----
        de_rows, de_ks, t0 = [], [], time.time()
        max_spins = 15 if arm == "A_cryo" else 10
        for i, ds in enumerate(suite):
            fit = greedy_de_fit(
                tau, ds["m_recs"], a["n_pulses"], analysis_field,
                max_spins=max_spins, a_bounds=(-60e3, 60e3),
                b_bounds=(3e3, 65e3), de_maxiter=50, de_popsize=20,
                seed=args.seed + i, verbose=False)
            det = np.array([s[0] for s in fit["best"]["spins"]])
            de_rows.append(match_score(det, gt_a, tol))
            de_ks.append(fit["best"]["k"])
            print(f"  [de] {i+1}/{n_real} k*={fit['best']['k']} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        mde = agg(de_rows, len(gt_a))
        mde["mean_k"] = round(float(np.mean(de_ks)), 1)
        print(f"[de]   {json.dumps(mde)}", flush=True)

        results[arm] = {"2021": m2021, "cdetect_de": mde}
        (OUT / "results.json").write_text(json.dumps(results, indent=2))

    print("\n===== FINAL =====")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
