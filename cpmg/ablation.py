"""Ablation-study infrastructure: train/evaluate HPC pipeline variants on a
shared ground-truth benchmark suite across noise levels.

Axes:
  arch     : 'mlp2021' (paper-faithful Dense+BCE) | 'cnn' (ours)
  channels : (16,) single-N  | (8, 16, 20) joint
  denoise  : None | 'ae2021' | 'nafnet'

The noise level sigma is the room-temperature proxy: the 2021 experiment
(3.7 K, resonant readout) corresponds to sigma ~ 0.02-0.03; our
room-temperature data sits at sigma ~ 0.05-0.08.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .datagen import GenConfig, make_signals, signals_to_image
from .models import HPC_MLP2021, DenoiseAE2021, HPCNet, NAFNet1D
from .physics import cpmg_M, stretched_exp, target_period


@dataclass
class AblationConfig:
    name: str
    arch: str = "cnn"                 # 'cnn' | 'mlp2021'
    n_pulses: tuple = (8, 16, 20)     # channels
    denoise: str | None = None        # None | 'ae2021' | 'nafnet'


# ---------------------------------------------------------------- suite

def build_suite(tau, sigmas, n_per_sigma, seed=123, n_pulses=(8, 16, 20),
                b_field_g=None, a_lim=50e3, b_range=(15e3, 50e3),
                min_sep=6e3):
    """Fixed GT benchmark datasets. Always generated with ALL given channels;
    single-channel configs slice what they need. Returns list of dicts."""
    cfg = GenConfig(n_pulses=n_pulses)
    if b_field_g is not None:
        cfg.b_field_g = b_field_g
    suite = []
    for sigma in sigmas:
        for i in range(n_per_sigma):
            rng = np.random.default_rng(seed + int(sigma * 1e4) * 100 + i)
            n_spins = int(rng.integers(3, 6))
            while True:
                a = rng.uniform(-a_lim, a_lim, size=n_spins)
                if n_spins < 2 or np.min(np.diff(np.sort(a))) > min_sep:
                    break
            b = rng.uniform(b_range[0], b_range[1], size=n_spins)
            gt = np.column_stack([a, b])
            m_recs = []
            for n_pulse in cfg.n_pulses:
                m = cpmg_M(tau, gt, n_pulse, cfg.b_field_g)
                t2 = rng.uniform(150, 400) * 1e-6
                st = rng.uniform(0.4, 1.0)
                a0 = rng.uniform(0.75, 0.95)
                env = stretched_exp(tau, t2, st)
                px = 0.5 + 0.5 * a0 * m * env + rng.normal(0, sigma, len(tau))
                m_recs.append(np.clip((2 * px - 1) / np.maximum(a0 * env, 1e-3),
                                      -1.5, 1.5))
            suite.append(dict(sigma=sigma, gt=gt, m_recs=np.array(m_recs)))
    return suite


# ---------------------------------------------------------------- denoiser

def train_denoiser(kind, tau, gen_cfg: GenConfig, device, n_samples=6000,
                   epochs=8, batch=256, seed=0, verbose=True):
    """Train a 1-channel denoiser on (noisy m_rec -> clean M) pairs drawn
    from random windows/classes over the full A range."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    single = GenConfig(**{**gen_cfg.__dict__, "n_pulses": gen_cfg.n_pulses})
    n_sig = 0
    while n_sig < n_samples:
        a_c = rng.uniform(-60e3, 60e3)
        cls = int(rng.integers(0, 3))
        m_recs, m_cleans, _ = make_signals(rng, tau, a_c, cls, single)
        xs.append(m_recs)
        ys.append(m_cleans)
        n_sig += m_recs.shape[0]
    X = torch.from_numpy(np.concatenate(xs).astype(np.float32))[:, None, :]
    Y = torch.from_numpy(np.concatenate(ys).astype(np.float32))[:, None, :]

    model = (DenoiseAE2021() if kind == "ae2021" else NAFNet1D()).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.L1Loss()
    n = len(X)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            b = perm[i : i + batch]
            xb, yb = X[b].to(device), Y[b].to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        if verbose:
            print(f"    denoiser[{kind}] ep{ep+1}/{epochs} L1={tot/n:.4f}", flush=True)
    model.eval()
    return model


def apply_denoiser(model, m_recs, device):
    """(C, T) -> denoised (C, T)."""
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(m_recs, dtype=np.float32))[:, None, :].to(device)
        return model(x)[:, 0, :].cpu().numpy()


# ---------------------------------------------------------------- bank

def _build_classifier(cfg: AblationConfig, img_shape, device):
    c, s, w = img_shape
    if cfg.arch == "mlp2021":
        return HPC_MLP2021(c * s * w).to(device)
    return HPCNet(in_ch=c).to(device)


def _loss_fn(cfg: AblationConfig):
    if cfg.arch == "mlp2021":
        bce = nn.BCEWithLogitsLoss()

        def f(logits, y):
            return bce(logits, nn.functional.one_hot(y, 3).float())

        return f
    ce = nn.CrossEntropyLoss()
    return lambda logits, y: ce(logits, y)


def _p_spin(cfg: AblationConfig, logits):
    if cfg.arch == "mlp2021":
        return 1.0 - torch.sigmoid(logits[:, 0])
    return 1.0 - torch.softmax(logits, dim=1)[:, 0]


def train_bank(cfg: AblationConfig, tau, centers_hz, outdir: Path, device,
               denoiser=None, n_per_class=700, epochs=20, seed=0, verbose=True,
               gen_cfg: GenConfig | None = None):
    """Train one classifier per A window; save state dicts to outdir."""
    outdir.mkdir(parents=True, exist_ok=True)
    if gen_cfg is None:
        gen_cfg = GenConfig(n_pulses=cfg.n_pulses, noise_range=(0.02, 0.13))
    loss_fn = _loss_fn(cfg)
    accs = []
    for i, a_c in enumerate(centers_hz):
        t0 = time.time()
        rng = np.random.default_rng(seed + 1000 + i)
        sig_list, y = [], []
        tp = target_period(a_c, gen_cfg.b_repr, gen_cfg.b_field_g)
        for cls in (0, 1, 2):
            for _ in range(n_per_class):
                m_recs, _, _ = make_signals(rng, tau, a_c, cls, gen_cfg)
                sig_list.append(m_recs)
                y.append(cls)
        sigs = np.stack(sig_list)  # (n, C, T)
        if denoiser is not None:
            n_s, n_c, n_t = sigs.shape
            flat = sigs.reshape(n_s * n_c, n_t)
            den = []
            for j in range(0, len(flat), 2048):
                den.append(apply_denoiser(denoiser, flat[j : j + 2048], device))
            sigs = np.concatenate(den).reshape(n_s, n_c, n_t)
        X = np.stack([signals_to_image(tau, s, tp, gen_cfg) for s in sigs])
        y = np.array(y, dtype=np.int64)

        torch.manual_seed(seed + i)
        n = len(X)
        idx = torch.randperm(n)
        n_val = int(0.2 * n)
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        Xt = torch.from_numpy(X).to(device)
        yt = torch.from_numpy(y).to(device)
        model = _build_classifier(cfg, X.shape[1:], device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        best_acc, best_state = 0.0, None
        for ep in range(epochs):
            model.train()
            perm = tr_idx[torch.randperm(len(tr_idx))]
            for j in range(0, len(perm), 256):
                b = perm[j : j + 256]
                opt.zero_grad()
                loss = loss_fn(model(Xt[b]), yt[b])
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                pred = model(Xt[val_idx]).argmax(1)
                acc = (pred == yt[val_idx]).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        torch.save(best_state, outdir / f"w_{a_c/1e3:+08.1f}kHz.pt")
        accs.append(best_acc)
        if verbose and (i % 10 == 0 or i == len(centers_hz) - 1):
            print(f"    [{cfg.name}] window {i+1}/{len(centers_hz)} "
                  f"A={a_c/1e3:+.0f} kHz acc={best_acc:.3f} ({time.time()-t0:.1f}s)",
                  flush=True)
    return accs


def eval_bank(cfg: AblationConfig, tau, centers_hz, bankdir: Path, device,
              suite, denoiser=None, height=0.6, prominence=0.1,
              gen_cfg: GenConfig | None = None):
    """Run the bank on every suite dataset -> detections + P(spin) curves."""
    from scipy.signal import find_peaks

    if gen_cfg is None:
        gen_cfg = GenConfig(n_pulses=cfg.n_pulses)
    suite_pulses = suite[0].get("n_pulses", (8, 16, 20))
    sel = [list(suite_pulses).index(n) for n in cfg.n_pulses]

    # preload models
    models = []
    for a_c in centers_hz:
        f = bankdir / f"w_{a_c/1e3:+08.1f}kHz.pt"
        state = torch.load(f, map_location=device)
        # rebuild with correct input dim per window
        tp = target_period(a_c, gen_cfg.b_repr, gen_cfg.b_field_g)
        s = int(np.floor((tau[-1] - tau[0]) / tp))
        model = _build_classifier(cfg, (len(sel), s, gen_cfg.image_width), device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)

    results = []
    for ds in suite:
        m_recs = ds["m_recs"][sel]
        if denoiser is not None:
            m_recs = apply_denoiser(denoiser, m_recs, device)
        p_list = []
        for a_c, model in zip(centers_hz, models):
            tp = target_period(a_c, gen_cfg.b_repr, gen_cfg.b_field_g)
            img = signals_to_image(tau, m_recs, tp, gen_cfg)
            with torch.no_grad():
                logits = model(torch.from_numpy(img[None]).to(device))
                p_list.append(float(_p_spin(cfg, logits)[0]))
        p_arr = np.array(p_list)
        pk, _ = find_peaks(p_arr, height=height, prominence=prominence)
        results.append(dict(sigma=ds["sigma"], gt_a=ds["gt"][:, 0],
                            detected_a=np.array(centers_hz)[pk], p_curve=p_arr))
    return results


# ---------------------------------------------------------------- metrics

def match_score(detected_a, gt_a, tol=4e3):
    detected = list(detected_a)
    tp, errors = 0, []
    for g in sorted(gt_a):
        if not detected:
            break
        d = min(detected, key=lambda x: abs(x - g))
        if abs(d - g) <= tol:
            tp += 1
            errors.append(abs(d - g))
            detected.remove(d)
    return tp, len(detected_a) - tp, len(gt_a) - tp, errors


def summarize(results, tol=4e3):
    """Per-sigma precision/recall/F1."""
    out = {}
    sigmas = sorted({r["sigma"] for r in results})
    for s in sigmas:
        rows = [r for r in results if r["sigma"] == s]
        tp = fp = fn = 0
        errs = []
        for r in rows:
            a, b, c, e = match_score(r["detected_a"], r["gt_a"], tol)
            tp += a
            fp += b
            fn += c
            errs += e
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        out[s] = dict(tp=tp, fp=fp, fn=fn, precision=round(prec, 3),
                      recall=round(rec, 3), f1=round(f1, 3),
                      mean_err_khz=round(float(np.mean(errs)) / 1e3, 2) if errs else None)
    return out
