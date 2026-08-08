"""SpinDETR: end-to-end set prediction of nuclear spins from raw CPMG signals.

A clean architectural break from the 2021 pipeline (image dataset -> per-window
HPC classifiers -> denoiser):

  - input   : raw multi-channel envelope-normalized signals (C=3, T=700)
              -- no slice-stack images, no A-window grid, no separate denoiser
  - encoder : 1D conv stem + Transformer encoder (sees all dips and their
              periodic structure at once; noise robustness is learned end-to-end)
  - decoder : K learned queries cross-attend the signal (DETR, Carion 2020)
  - output  : a SET of spins {(existence, A, B)} in ONE forward pass,
              trained with Hungarian matching -- the spin count comes from
              existence probabilities, replacing 61 window models + peak finding

Conceptually this inverts old-cdetect's PhysicsDeepSets (spin set -> signal):
here, signal -> spin set.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from .datagen import GenConfig
from .physics import cpmg_M, stretched_exp

A_MAX = 60e3          # |A| range (Hz)
B_MIN, B_MAX = 8e3, 60e3
N_QUERIES = 10


# ----------------------------------------------------------------- data

def sample_random_scene(rng, tau, cfg: GenConfig, sigma=None,
                        max_targets=6, a_max=A_MAX, b_tgt=(15e3, 50e3),
                        min_sep=3e3):
    """Window-free training scene: detectable spins (targets) + weak bath."""
    n_t = int(rng.integers(0, max_targets + 1))
    while True:
        a = rng.uniform(-a_max, a_max, size=n_t)
        if n_t < 2 or np.min(np.diff(np.sort(a))) > min_sep:
            break
    b = rng.uniform(b_tgt[0], b_tgt[1], size=n_t)
    targets = np.column_stack([a, b]) if n_t else np.zeros((0, 2))

    n_bath = rng.poisson(cfg.bath_n_mean)
    bath = np.column_stack([
        rng.uniform(-cfg.bath_a_range, cfg.bath_a_range, size=n_bath),
        rng.uniform(1e3, cfg.bath_b_max, size=n_bath),
    ]) if n_bath else np.zeros((0, 2))
    ab_all = np.vstack([targets, bath])

    m_recs = []
    for n_pulse in cfg.n_pulses:
        m = cpmg_M(tau, ab_all, n_pulse, cfg.b_field_g)
        t2 = rng.uniform(*cfg.t2_range_us) * 1e-6
        st = rng.uniform(*cfg.stretch_range)
        a0 = rng.uniform(*cfg.contrast_range)
        sig = rng.uniform(*cfg.noise_range) if sigma is None else sigma
        env = stretched_exp(tau, t2, st)
        px = 0.5 + 0.5 * a0 * m * env + rng.normal(0.0, sig, len(tau))
        m_recs.append(np.clip((2 * px - 1) / np.maximum(a0 * env, 1e-3), -1.5, 1.5))
    return np.array(m_recs, dtype=np.float32), targets


def build_dataset(tau, n_samples, cfg: GenConfig, seed=0):
    rng = np.random.default_rng(seed)
    X = np.empty((n_samples, len(cfg.n_pulses), len(tau)), dtype=np.float32)
    T = np.full((n_samples, N_QUERIES, 2), np.nan, dtype=np.float32)
    n_gt = np.zeros(n_samples, dtype=np.int64)
    for i in range(n_samples):
        x, tg = sample_random_scene(rng, tau, cfg)
        X[i] = x
        n = min(len(tg), N_QUERIES)
        n_gt[i] = n
        if n:
            T[i, :n] = tg[:n]
    return X, T, n_gt


# ----------------------------------------------------------------- model

class _PosEnc(nn.Module):
    def __init__(self, d, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).float()[:, None]
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(1e4) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):  # (B, L, D)
        return x + self.pe[: x.shape[1]][None]


class SpinDETR(nn.Module):
    def __init__(self, in_ch=3, d=128, n_heads=4, n_enc=4, n_dec=4,
                 n_queries=N_QUERIES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 64, 7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(64, d, 5, stride=2, padding=2), nn.GELU(),
        )
        self.pos = _PosEnc(d)
        enc_layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d,
                                               batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(enc_layer, n_enc)
        dec_layer = nn.TransformerDecoderLayer(d, n_heads, 4 * d,
                                               batch_first=True, dropout=0.1)
        self.decoder = nn.TransformerDecoder(dec_layer, n_dec)
        self.queries = nn.Embedding(n_queries, d)
        self.head_exist = nn.Linear(d, 1)
        self.head_ab = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))

    def forward(self, x):  # x (B, C, T)
        f = self.stem(x).transpose(1, 2)  # (B, L, D)
        mem = self.encoder(self.pos(f))
        q = self.queries.weight[None].expand(x.shape[0], -1, -1)
        out = self.decoder(q, mem)  # (B, K, D)
        exist_logit = self.head_exist(out)[..., 0]        # (B, K)
        ab = self.head_ab(out)                            # (B, K, 2)
        a_hz = torch.tanh(ab[..., 0]) * A_MAX
        b_hz = B_MIN + torch.sigmoid(ab[..., 1]) * (B_MAX - B_MIN)
        return exist_logit, a_hz, b_hz


# ----------------------------------------------------------------- loss

def hungarian_loss(exist_logit, a_hz, b_hz, tgt, n_gt, eos_coef=0.15,
                   l1_weight=5.0):
    """DETR-style set loss. tgt (B, K, 2) with NaN padding, n_gt (B,)."""
    Bsz, K = exist_logit.shape
    device = exist_logit.device
    total_cls, total_reg, n_match = 0.0, 0.0, 0
    bce = nn.functional.binary_cross_entropy_with_logits

    for i in range(Bsz):
        n = int(n_gt[i])
        target_exist = torch.zeros(K, device=device)
        if n > 0:
            ta = tgt[i, :n, 0] / A_MAX                    # normalize
            tb = (tgt[i, :n, 1] - B_MIN) / (B_MAX - B_MIN)
            pa = a_hz[i] / A_MAX
            pb = (b_hz[i] - B_MIN) / (B_MAX - B_MIN)
            cost_reg = (pa[:, None] - ta[None]).abs() + (pb[:, None] - tb[None]).abs()
            cost = -torch.sigmoid(exist_logit[i])[:, None] + l1_weight * cost_reg
            row, col = linear_sum_assignment(cost.detach().cpu().numpy())
            row = torch.as_tensor(row, device=device)
            col = torch.as_tensor(col, device=device)
            target_exist[row] = 1.0
            reg = (pa[row] - ta[col]).abs().sum() + (pb[row] - tb[col]).abs().sum()
            total_reg = total_reg + reg
            n_match += n
        w = torch.where(target_exist > 0, torch.ones(1, device=device),
                        torch.full((1,), eos_coef, device=device))
        total_cls = total_cls + (bce(exist_logit[i], target_exist,
                                     reduction="none") * w).sum()
    loss = total_cls / (Bsz * K) + l1_weight * total_reg / max(n_match, 1)
    return loss


# ----------------------------------------------------------------- infer

@torch.no_grad()
def predict_spins(model, m_recs, device, thresh=0.5):
    """(C, T) -> list of (A, B, p) for queries above threshold."""
    x = torch.from_numpy(np.asarray(m_recs, dtype=np.float32))[None].to(device)
    exist_logit, a_hz, b_hz = model(x)
    p = torch.sigmoid(exist_logit)[0].cpu().numpy()
    a = a_hz[0].cpu().numpy()
    b = b_hz[0].cpu().numpy()
    keep = p > thresh
    order = np.argsort(-p[keep])
    return [(float(a[keep][j]), float(b[keep][j]), float(p[keep][j]))
            for j in order]
