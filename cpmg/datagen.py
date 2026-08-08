"""Training-set generator for the HPC classifier, following the
target/side/mid/far interference strategy of Jung et al. (2021),
re-tuned for our experimental conditions:

  B = 440.1 G, N = 8/16/20 (3 channels), 700 tau points (20 ns, 0.02-14 us).

Because our data has ~13 local periods (vs ~50 in the paper) the A-axis
granularity is coarser: classification windows are ~2 kHz wide (vs 200 Hz)
and interferer offsets are scaled up accordingly.

Classes (per A window, paper Fig. 2c):
  0 : no spin with the target period (interferers only)
  1 : one target spin in the window
  2 : two target spins with slightly dissimilar periods
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .physics import B_FIELD_G, cpmg_M, stretched_exp, target_period
from .represent import slice_stack


@dataclass
class GenConfig:
    b_field_g: float = B_FIELD_G
    n_pulses: tuple = (8, 16, 20)
    image_width: int = 53

    # target-spin (A, B) space  [Hz]
    a_half: float = 1e3          # half-width of one classification window
    b_tgt_min: float = 10e3
    b_tgt_max: float = 60e3
    b_repr: float = 30e3         # representative B for the window's TP

    # interferers [Hz offsets from window center]
    side_da: tuple = (3e3, 10e3)
    mid_da: tuple = (10e3, 30e3)
    far_da: tuple = (30e3, 100e3)
    side_n_max: int = 3
    mid_n_max: int = 4
    far_n_max: int = 6
    side_keep_p: float = 0.5     # paper spin_zero_scale 'side': 0.50
    mid_keep_p: float = 0.95     # 'mid'/'far': 0.05 zero-scale
    far_keep_p: float = 0.95
    b_int_min: float = 10e3
    b_int_max: float = 60e3

    # undetectable background bath (shallow dips, present in every class)
    bath_n_mean: float = 15.0
    bath_a_range: float = 100e3
    bath_b_max: float = 8e3

    # measurement model
    t2_range_us: tuple = (50.0, 600.0)
    stretch_range: tuple = (0.3, 1.2)
    contrast_range: tuple = (0.7, 1.0)
    noise_range: tuple = (0.02, 0.08)


def _sample_interferers(rng: np.random.Generator, a_center: float, cfg: GenConfig):
    """side / mid / far interfering spins around (but outside) the window."""
    spins = []
    groups = [
        (cfg.side_da, cfg.side_n_max, cfg.side_keep_p),
        (cfg.mid_da, cfg.mid_n_max, cfg.mid_keep_p),
        (cfg.far_da, cfg.far_n_max, cfg.far_keep_p),
    ]
    for (da_lo, da_hi), n_max, keep_p in groups:
        for _ in range(rng.integers(0, n_max + 1)):
            if rng.random() > keep_p:
                continue
            da = rng.uniform(da_lo, da_hi) * rng.choice([-1.0, 1.0])
            b = rng.uniform(cfg.b_int_min, cfg.b_int_max)
            spins.append((a_center + da, b))
    return spins


def _sample_bath(rng: np.random.Generator, cfg: GenConfig):
    """Weakly coupled background spins (B too small to classify)."""
    n = rng.poisson(cfg.bath_n_mean)
    a = rng.uniform(-cfg.bath_a_range, cfg.bath_a_range, size=n)
    b = rng.uniform(1e3, cfg.bath_b_max, size=n)
    return np.column_stack([a, b]) if n else np.zeros((0, 2))


def sample_spin_set(rng: np.random.Generator, a_center: float, cls: int, cfg: GenConfig):
    """Return (targets, others) as (K,2) arrays of (A, B) in Hz."""
    targets = []
    if cls >= 1:
        n_targets = cls  # class 1 -> 1 target, class 2 -> 2 targets
        for _ in range(n_targets):
            a = rng.uniform(a_center - cfg.a_half, a_center + cfg.a_half)
            b = rng.uniform(cfg.b_tgt_min, cfg.b_tgt_max)
            targets.append((a, b))
    others = _sample_interferers(rng, a_center, cfg)
    ab_others = np.array(others, dtype=np.float64).reshape(-1, 2)
    ab_others = np.vstack([ab_others, _sample_bath(rng, cfg)])
    return np.array(targets, dtype=np.float64).reshape(-1, 2), ab_others


def make_signals(
    rng: np.random.Generator,
    tau: np.ndarray,
    a_center: float,
    cls: int,
    cfg: GenConfig,
    sigma: float | None = None,
):
    """Envelope-normalized noisy signals for one sample.

    Returns (m_recs (C,T), m_cleans (C,T), info dict). All N channels share
    the same spin set (physically consistent); envelope/contrast/noise are
    drawn per channel (sigma overrides the random noise draw when given).
    """
    targets, others = sample_spin_set(rng, a_center, cls, cfg)
    ab_all = np.vstack([targets, others])
    m_recs, m_cleans, pxs = [], [], []
    for n_pulse in cfg.n_pulses:
        m = cpmg_M(tau, ab_all, n_pulse, cfg.b_field_g)
        t2 = rng.uniform(*cfg.t2_range_us) * 1e-6
        st = rng.uniform(*cfg.stretch_range)
        a0 = rng.uniform(*cfg.contrast_range)
        sig = rng.uniform(*cfg.noise_range) if sigma is None else sigma
        env = stretched_exp(tau, t2, st)
        px = 0.5 + 0.5 * a0 * m * env + rng.normal(0.0, sig, len(tau))
        # envelope-normalize with the KNOWN envelope (fast path; the
        # inference pipeline fits it, cf. represent.envelope_normalize)
        m_rec = np.clip((2.0 * px - 1.0) / np.maximum(a0 * env, 1e-3), -1.5, 1.5)
        m_recs.append(m_rec)
        m_cleans.append(m)
        pxs.append(px)
    info = {"targets": targets, "others": others,
            "tp": target_period(a_center, cfg.b_repr, cfg.b_field_g),
            "signals": np.array(pxs)}
    return np.array(m_recs), np.array(m_cleans), info


def signals_to_image(tau, m_recs, tp, cfg: GenConfig) -> np.ndarray:
    """(C, T) signals -> (C, S, W) slice-stack image stack."""
    imgs = [
        np.nan_to_num(
            slice_stack(tau, m, tp, width=cfg.image_width, tau_start=tau[0]),
            nan=1.0,
        )
        for m in m_recs
    ]
    return np.stack(imgs).astype(np.float32)


def make_sample(
    rng: np.random.Generator,
    tau: np.ndarray,
    a_center: float,
    cls: int,
    cfg: GenConfig,
    return_signals: bool = False,
):
    """One training sample: (C, S, W) slice-stack image stack + label."""
    m_recs, _, info = make_signals(rng, tau, a_center, cls, cfg)
    out = signals_to_image(tau, m_recs, info["tp"], cfg)
    if return_signals:
        return out, cls, info
    return out, cls


def gen_window_dataset(
    tau: np.ndarray,
    a_center: float,
    n_per_class: int,
    cfg: GenConfig | None = None,
    classes=(0, 1, 2),
    seed: int = 0,
):
    """Generate a balanced dataset for one classification window.

    Returns X (n, C, S, W) float32, y (n,) int64. Slice count S depends on
    TP(a_center), so different windows may have different heights.
    """
    cfg = cfg or GenConfig()
    rng = np.random.default_rng(seed)
    X, y = [], []
    for cls in classes:
        for _ in range(n_per_class):
            img, label = make_sample(rng, tau, a_center, cls, cfg)
            X.append(img)
            y.append(label)
    return np.stack(X), np.array(y, dtype=np.int64)
