"""PeriodFormer: overlapping period-window tokens + transformer attention.

The user-proposed middle ground between the 2021 pipeline and raw end-to-end:

  2021          : per-window IMAGE -> 61 independent classifiers ("does a
                  vertical line appear?"), no information shared across windows
  SpinDETR      : raw signal -> set decoder (no physics prior at all)
  PeriodFormer  : every candidate period window is EMBEDDED into a token by a
                  shared CNN; a transformer attends ACROSS windows, so evidence
                  from overlapping periods is combined before deciding.
                  Output: per-window spin-existence curve (dense over A).

The period-folding (slice-stack) physics prior is kept, but detection is a
single global model instead of 61 independent ones. Window token building is
implemented as a precomputed gather so the whole pipeline runs batched on GPU.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .datagen import GenConfig
from .physics import target_period

S_MAX = 13  # padded slice count
WIDTH = 53


class TokenBuilder:
    """Precomputes interpolation indices to turn (B, C, T) signals into
    (B, K, C, S_MAX, WIDTH) window-token images on GPU."""

    def __init__(self, tau: np.ndarray, centers_hz: np.ndarray,
                 cfg: GenConfig, device, s_max: int = S_MAX):
        self.centers_hz = centers_hz
        self.s_max = s_max
        t0, t_end = tau[0], tau[-1]
        dt = float(np.median(np.diff(tau)))
        idx0_list, frac_list, mask_list = [], [], []
        for a_c in centers_hz:
            tp = target_period(a_c, cfg.b_repr, cfg.b_field_g)
            n_s = int(np.floor((t_end - t0) / tp))
            cols = (np.arange(WIDTH) + 0.5) / WIDTH
            rows = np.arange(s_max)
            grid = t0 + (rows[:, None] + cols[None, :]) * tp  # (s_max, W)
            pos = (grid - t0) / dt
            idx0 = np.clip(np.floor(pos).astype(np.int64), 0, len(tau) - 2)
            frac = np.clip(pos - idx0, 0.0, 1.0)
            valid = (rows < n_s)[:, None] & (grid <= t_end)
            idx0_list.append(idx0)
            frac_list.append(frac)
            mask_list.append(valid)
        self.idx0 = torch.as_tensor(np.stack(idx0_list), device=device)          # (K,S,W)
        self.frac = torch.as_tensor(np.stack(frac_list), dtype=torch.float32,
                                    device=device)
        self.mask = torch.as_tensor(np.stack(mask_list), device=device)

    def __call__(self, signals: torch.Tensor) -> torch.Tensor:
        """(B, C, T) float32 -> (B, K, C, S_MAX, WIDTH)"""
        B, C, T = signals.shape
        K, S, W = self.idx0.shape
        flat_idx0 = self.idx0.reshape(-1)               # (K*S*W,)
        g0 = signals[..., flat_idx0].reshape(B, C, K, S, W)
        g1 = signals[..., flat_idx0 + 1].reshape(B, C, K, S, W)
        frac = self.frac[None, None]
        img = g0 * (1 - frac) + g1 * frac
        img = torch.where(self.mask[None, None], img, torch.ones_like(img))
        return img.permute(0, 2, 1, 3, 4).contiguous()  # (B, K, C, S, W)


class PeriodFormer(nn.Module):
    def __init__(self, in_ch=3, d=128, n_heads=4, n_layers=4, n_tokens=61,
                 s_max=S_MAX):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.GELU(), nn.MaxPool2d((1, 2)),
            nn.Conv2d(16, 32, 3, padding=1), nn.GELU(), nn.MaxPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(32 * (s_max // 2) * (WIDTH // 4), d), nn.GELU(),
        )
        pe = torch.zeros(n_tokens, d)
        pos = torch.arange(n_tokens).float()[:, None]
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(1e4) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, batch_first=True,
                                           dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens (B, K, C, S, W) -> existence logits (B, K)"""
        B, K = tokens.shape[:2]
        f = self.embed(tokens.flatten(0, 1)).reshape(B, K, -1)
        out = self.encoder(f + self.pe[None, :K])
        return self.head(out)[..., 0]


def soft_labels(gt_a_hz, centers_hz, width_hz=1500.0):
    """Gaussian soft existence labels over the window grid."""
    y = np.zeros(len(centers_hz), dtype=np.float32)
    for a in gt_a_hz:
        y = np.maximum(y, np.exp(-0.5 * ((centers_hz - a) / width_hz) ** 2))
    return y
