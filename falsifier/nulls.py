"""Null baselines.

A raw statistic answers "is this different from zero". Almost every dead
strategy clears that bar. The question that actually decides the case is
"is this different from something that knows nothing I claim to know", and
the answer depends entirely on what the null is allowed to keep.

Each generator below keeps a different part of the real signal and destroys
exactly one claim. Pick the one that matches what you are asserting -- and
match the null on every dimension that decided which names entered the
portfolio, not just the one that is convenient to match on.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


def _rng(seed) -> np.random.Generator:
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def cs_shuffle(signal: np.ndarray, mask: Optional[np.ndarray] = None, seed=0) -> np.ndarray:
    """Permute the signal inside each cross-section.

    Destroys everything. This is a harness sanity check, not evidence: if the
    real statistic does not clearly beat this, nothing else is worth running.
    """
    g = _rng(seed)
    out = np.asarray(signal, float).copy()
    for t in range(out.shape[0]):
        m = np.isfinite(out[t])
        if mask is not None:
            m = m & mask[t]
        idx = np.flatnonzero(m)
        if idx.size > 1:
            out[t, idx] = out[t, g.permutation(idx)]
    return out


def matched_permutation(signal: np.ndarray, covariates: Dict[str, np.ndarray],
                        mask: Optional[np.ndarray] = None, n_bucket: int = 5,
                        seed=0) -> np.ndarray:
    """Permute the signal *within* buckets of the given covariates.

    The null keeps every bit of information the covariates carry and destroys
    only the within-bucket ranking -- i.e. the part you are claiming is yours.
    If the real statistic sits inside this null, the covariates were the whole
    story and the signal is a repackaging of them.
    """
    g = _rng(seed)
    sig = np.asarray(signal, float)
    out = sig.copy()
    T = sig.shape[0]
    names = sorted(covariates)
    for t in range(T):
        m = np.isfinite(sig[t])
        if mask is not None:
            m = m & mask[t]
        for nm in names:
            m = m & np.isfinite(covariates[nm][t])
        idx = np.flatnonzero(m)
        if idx.size < n_bucket * 2:
            continue
        key = np.zeros(idx.size, dtype=np.int64)
        for nm in names:
            v = np.asarray(covariates[nm], float)[t, idx]
            r = np.argsort(np.argsort(v))
            b = np.minimum((r * n_bucket) // idx.size, n_bucket - 1)
            key = key * n_bucket + b
        for b in np.unique(key):
            sel = idx[key == b]
            if sel.size > 1:
                out[t, sel] = sig[t, g.permutation(sel)]
    return out


def identity_permutation(signal: np.ndarray, seed=0,
                         groups: Optional[np.ndarray] = None) -> np.ndarray:
    """Relabel which asset is which, once, for the whole sample.

    Preserves the cross-sectional distribution *and* the time-series
    persistence of the signal; destroys only the claim "it is this asset that
    is special". Pass ``groups`` to keep the permutation inside an index or
    sector so the null stays comparable.
    """
    g = _rng(seed)
    sig = np.asarray(signal, float)
    N = sig.shape[1]
    perm = np.arange(N)
    if groups is None:
        perm = g.permutation(N)
    else:
        for gid in np.unique(groups[~np.isnan(groups)] if groups.dtype.kind == "f" else groups):
            sel = np.flatnonzero(groups == gid)
            if sel.size > 1:
                perm[sel] = g.permutation(sel)
    return sig[:, perm]


def orthogonalize(signal: np.ndarray, controls: Sequence[np.ndarray],
                  mask: Optional[np.ndarray] = None, add_const: bool = True) -> np.ndarray:
    """Cross-sectional residual of the signal against known factors.

    Not a null in itself -- it is how you ask "is there anything left after
    the obvious explanation". Feed the residual back through the same nulls.
    """
    sig = np.asarray(signal, float)
    ctrls = [np.asarray(c, float) for c in controls]
    out = np.full_like(sig, np.nan)
    for t in range(sig.shape[0]):
        m = np.isfinite(sig[t])
        if mask is not None:
            m = m & mask[t]
        for c in ctrls:
            m = m & np.isfinite(c[t])
        idx = np.flatnonzero(m)
        if idx.size < len(ctrls) + 5:
            continue
        X = np.column_stack([c[t, idx] for c in ctrls])
        if add_const:
            X = np.column_stack([np.ones(idx.size), X])
        y = sig[t, idx]
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        out[t, idx] = y - X @ beta
    return out


def random_selection(mask: np.ndarray, n_per_date: np.ndarray, hold: int = 1,
                     seed=0) -> np.ndarray:
    """A random portfolio matched on breadth and holding period.

    ``n_per_date`` and ``hold`` must be taken from the real strategy: matching
    the count but not the holding period leaves the null with a different
    turnover, and turnover alone can manufacture the entire result.
    """
    g = _rng(seed)
    T, N = mask.shape
    out = np.zeros((T, N), bool)
    held_until = np.full(N, -1)
    for t in range(T):
        alive = np.flatnonzero(mask[t])
        keep = np.flatnonzero(held_until > t)
        want = int(n_per_date[t]) if np.isfinite(n_per_date[t]) else 0
        need = max(0, want - keep.size)
        pool = np.setdiff1d(alive, keep, assume_unique=False)
        pick = g.choice(pool, size=min(need, pool.size), replace=False) if pool.size and need else np.array([], int)
        held_until[pick] = t + hold
        sel = np.union1d(keep, pick).astype(int)
        out[t, sel] = True
    return out


def null_distribution(statistic: Callable[[int], float], n_draws: int = 200,
                      seed=0, progress: bool = False) -> np.ndarray:
    """Evaluate ``statistic(draw_seed)`` ``n_draws`` times."""
    g = _rng(seed)
    seeds = g.integers(0, 2 ** 31 - 1, size=n_draws)
    vals = np.empty(n_draws)
    for i, s in enumerate(seeds):
        vals[i] = statistic(int(s))
        if progress and (i + 1) % max(1, n_draws // 10) == 0:
            print(f"  null draw {i + 1}/{n_draws}", flush=True)
    return vals


def percentile_of(real: float, draws: np.ndarray) -> float:
    """Percentile of the real statistic inside the null distribution."""
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    if d.size == 0 or not np.isfinite(real):
        return float("nan")
    return float((d < real).mean() * 100.0)


def empirical_p(real: float, draws: np.ndarray, two_sided: bool = False) -> float:
    """One-sided (or two-sided) empirical p-value with the +1 correction."""
    d = np.asarray(draws, float)
    d = d[np.isfinite(d)]
    if d.size == 0 or not np.isfinite(real):
        return float("nan")
    if two_sided:
        k = (np.abs(d) >= abs(real)).sum()
    else:
        k = (d >= real).sum()
    return float((k + 1) / (d.size + 1))
