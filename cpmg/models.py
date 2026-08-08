"""Neural models for CPMG analysis (PyTorch 2.x)."""

from __future__ import annotations

import torch
import torch.nn as nn


class HPCNet(nn.Module):
    """Small CNN classifier on (C, S, W) slice-stack images.

    C = number of CPMG channels (N=8/16/20 -> 3), S = slices (~12-13),
    W = pixels per period (53). AdaptiveAvgPool makes it robust to the
    window-dependent slice count.
    """

    def __init__(self, in_ch: int = 3, n_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),  # keep slice axis, halve tau axis
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((3, 13)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 13, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class HPC_MLP2021(nn.Module):
    """Faithful re-implementation of the 2021 paper's HPC classifier
    (refer repo imports/models.py): flattened image -> Dense
    2048-1024-512-K with BatchNorm + LeakyReLU, sigmoid outputs
    (multi-label style, trained with BCE on one-hot targets).
    """

    def __init__(self, in_dim: int, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 2048), nn.BatchNorm1d(2048), nn.LeakyReLU(),
            nn.Linear(2048, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.LeakyReLU(),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):  # logits; apply sigmoid in the loss/score
        return self.net(x)


class DenoiseAE2021(nn.Module):
    """1D conv autoencoder denoiser after the 2021 paper (kernel 4, 64ch,
    2x conv+maxpool encoder, 2x transposed-conv decoder), single channel.
    Input (B, 1, T); T is padded internally to a multiple of 4.
    """

    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv1d(1, 64, 4, padding=2),
                                  nn.BatchNorm1d(64), nn.LeakyReLU(), nn.MaxPool1d(2))
        self.enc2 = nn.Sequential(nn.Conv1d(64, 64, 4, padding=2),
                                  nn.BatchNorm1d(64), nn.LeakyReLU(), nn.MaxPool1d(2))
        self.dec1 = nn.Sequential(nn.ConvTranspose1d(64, 64, 4, stride=2, padding=1),
                                  nn.BatchNorm1d(64), nn.LeakyReLU())
        self.dec2 = nn.ConvTranspose1d(64, 1, 4, stride=2, padding=1)

    def forward(self, x):
        t = x.shape[-1]
        pad = (-t) % 4
        if pad:
            x = nn.functional.pad(x, (0, pad), mode="replicate")
        y = self.dec2(self.dec1(self.enc2(self.enc1(x))))
        return y[..., :t]


class _SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class _NAFBlock1D(nn.Module):
    """NAFNet block (ECCV 2022) adapted to 1D."""

    def __init__(self, c: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, c)
        self.conv1 = nn.Conv1d(c, 2 * c, 1)
        self.dwconv = nn.Conv1d(2 * c, 2 * c, 3, padding=1, groups=2 * c)
        self.gate = _SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(c, c, 1))
        self.conv2 = nn.Conv1d(c, c, 1)
        self.norm2 = nn.GroupNorm(1, c)
        self.ffn1 = nn.Conv1d(c, 2 * c, 1)
        self.ffn2 = nn.Conv1d(c, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1))

    def forward(self, x):
        y = self.gate(self.dwconv(self.conv1(self.norm1(x))))
        y = self.conv2(y * self.sca(y))
        x = x + y * self.beta
        y = self.ffn2(self.gate(self.ffn1(self.norm2(x))))
        return x + y * self.gamma


class NAFNet1D(nn.Module):
    """Lightweight 1D NAFNet denoiser (~0.15M params): 2-scale U-Net of
    NAFBlocks with a global residual (predicts the clean signal).
    Input/output (B, 1, T).
    """

    def __init__(self, width: int = 32, blocks: int = 2):
        super().__init__()
        self.intro = nn.Conv1d(1, width, 3, padding=1)
        self.enc = nn.Sequential(*[_NAFBlock1D(width) for _ in range(blocks)])
        self.down = nn.Conv1d(width, 2 * width, 2, stride=2)
        self.mid = nn.Sequential(*[_NAFBlock1D(2 * width) for _ in range(blocks)])
        self.up = nn.ConvTranspose1d(2 * width, width, 2, stride=2)
        self.dec = nn.Sequential(*[_NAFBlock1D(width) for _ in range(blocks)])
        self.out = nn.Conv1d(width, 1, 3, padding=1)

    def forward(self, x):
        t = x.shape[-1]
        pad = (-t) % 2
        if pad:
            x = nn.functional.pad(x, (0, pad), mode="replicate")
        f = self.intro(x)
        e = self.enc(f)
        m = self.up(self.mid(self.down(e)))
        y = self.out(self.dec(e + m))
        return (x + y)[..., :t]
