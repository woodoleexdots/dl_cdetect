"""Native trans-dimensional RJMCMC baseline (reviewer action item 2).

A compact reimplementation of the old-cdetect native RJMCMC (PROGRESS 18.5)
on this repo's forward model: birth / death / perturb moves over a variable
number of spins, annealed Metropolis acceptance on the Gaussian likelihood,
BIC-like complexity prior, per-spin M caching.

Returns the MAP spin configuration (detections for benchmark scoring) and
the posterior over the spin count k.
"""

from __future__ import annotations

import numpy as np

from .physics import cpmg_M


class RJMCMC:
    def __init__(self, tau, m_data, n_pulses, b_field_g,
                 a_bounds=(-60e3, 60e3), b_bounds=(3e3, 65e3),
                 sigma=0.06, max_spins=20, complexity=1.0, seed=0,
                 regions=None):
        """regions: optional [(lo, hi), ...] — when given, birth samples A
        only inside these intervals and perturb proposals leaving them are
        rejected (hybrid v2: PF-region-constrained RJMCMC)."""
        self.tau = tau
        self.m_data = m_data              # (C, T)
        self.n_pulses = n_pulses
        self.b = b_field_g
        self.a_lo, self.a_hi = a_bounds
        self.b_lo, self.b_hi = b_bounds
        self.sigma = sigma
        self.max_spins = max_spins
        # BIC-like: each spin costs `complexity * 2 * ln(n)` in -2 log posterior
        self.k_penalty = complexity * 2.0 * np.log(m_data.size)
        self.rng = np.random.default_rng(seed)
        self.regions = regions
        if regions:
            lens = np.array([hi - lo for lo, hi in regions], dtype=float)
            self.region_w = lens / lens.sum()
        self.spins: list = []             # list of [A, B]
        self.cache: list = []             # per-spin (C, T) M arrays

    def _sample_a(self):
        if self.regions:
            j = self.rng.choice(len(self.regions), p=self.region_w)
            lo, hi = self.regions[j]
            return self.rng.uniform(lo, hi)
        return self.rng.uniform(self.a_lo, self.a_hi)

    def _a_allowed(self, a):
        if not self.regions:
            return self.a_lo <= a <= self.a_hi
        return any(lo <= a <= hi for lo, hi in self.regions)

    def _spin_m(self, a, b):
        return np.array([cpmg_M(self.tau, [[a, b]], n, self.b)
                         for n in self.n_pulses])

    def _sse(self, cache):
        if cache:
            prod = np.prod(np.stack(cache), axis=0)
        else:
            prod = np.ones_like(self.m_data)
        return float(np.sum((prod - self.m_data) ** 2))

    def _neg2logpost(self, cache, k):
        return self._sse(cache) / self.sigma ** 2 + self.k_penalty * k

    def run(self, n_iter=30000, t_start=20.0, burn_frac=0.4, verbose=False):
        cur = self._neg2logpost(self.cache, 0)
        best = dict(score=cur, spins=[])
        k_counts = np.zeros(self.max_spins + 1)
        n_burn = int(burn_frac * n_iter)
        accepts = 0

        for it in range(n_iter):
            temp = max(1.0, t_start * (1 - it / max(n_burn, 1))) \
                if it < n_burn else 1.0
            u = self.rng.random()
            k = len(self.spins)

            if u < 0.2 and k < self.max_spins:          # birth
                a = self._sample_a()
                bb = self.rng.uniform(self.b_lo, self.b_hi)
                new_cache = self.cache + [self._spin_m(a, bb)]
                new = self._neg2logpost(new_cache, k + 1)
                if self.rng.random() < np.exp(min(0, (cur - new) / (2 * temp))):
                    self.spins.append([a, bb])
                    self.cache = new_cache
                    cur = new
                    accepts += 1
            elif u < 0.4 and k > 0:                      # death
                j = self.rng.integers(k)
                new_cache = self.cache[:j] + self.cache[j + 1:]
                new = self._neg2logpost(new_cache, k - 1)
                if self.rng.random() < np.exp(min(0, (cur - new) / (2 * temp))):
                    self.spins.pop(j)
                    self.cache = new_cache
                    cur = new
                    accepts += 1
            elif k > 0:                                  # perturb
                j = self.rng.integers(k)
                a0, b0 = self.spins[j]
                a = np.clip(a0 + self.rng.normal(0, 2e3), self.a_lo, self.a_hi)
                bb = np.clip(b0 + self.rng.normal(0, 5e3), self.b_lo, self.b_hi)
                if not self._a_allowed(a):
                    continue
                new_m = self._spin_m(a, bb)
                new_cache = self.cache[:j] + [new_m] + self.cache[j + 1:]
                new = self._neg2logpost(new_cache, k)
                if self.rng.random() < np.exp(min(0, (cur - new) / (2 * temp))):
                    self.spins[j] = [a, bb]
                    self.cache = new_cache
                    cur = new
                    accepts += 1

            if cur < best["score"]:
                best = dict(score=cur, spins=[list(s) for s in self.spins])
            if it >= n_burn:
                k_counts[len(self.spins)] += 1
            if verbose and (it + 1) % 10000 == 0:
                print(f"  it {it+1}: k={len(self.spins)} cur={cur:.1f} "
                      f"best_k={len(best['spins'])} acc={accepts/(it+1):.2f}",
                      flush=True)

        post_k = k_counts / max(k_counts.sum(), 1)
        return dict(map_spins=best["spins"], map_score=best["score"],
                    posterior_k=post_k.tolist(),
                    modal_k=int(np.argmax(k_counts)),
                    accept_rate=accepts / n_iter)
