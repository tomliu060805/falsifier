"""Cross-sectional statistics with the overlap correction that most write-ups skip."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.stats import rankdata


def forward_returns(ret: np.ndarray, horizon: int, delay: int = 0) -> np.ndarray:
    """Return the compounded return from ``t+delay`` to ``t+delay+horizon``.

    ``delay`` exists for the label-shift audit: entering one day later must
    degrade a real signal smoothly, not fall off a cliff.
    """
    r = np.asarray(ret, float)
    T = r.shape[0]
    log = np.log1p(np.nan_to_num(r, nan=0.0))
    valid = np.isfinite(r)
    out = np.full_like(r, np.nan)
    for t in range(T):
        a, b = t + 1 + delay, t + 1 + delay + horizon
        if b > T:
            break
        w = log[a:b]
        ok = valid[a:b].all(axis=0)
        acc = np.expm1(w.sum(axis=0))
        out[t] = np.where(ok, acc, np.nan)
    return out


def rank_ic(signal: np.ndarray, fwd: np.ndarray, mask: Optional[np.ndarray] = None,
            min_n: int = 30) -> np.ndarray:
    """Per-date Spearman IC. NaN on dates with too few tradable names."""
    s_, f_ = np.asarray(signal, float), np.asarray(fwd, float)
    T = s_.shape[0]
    out = np.full(T, np.nan)
    for t in range(T):
        m = np.isfinite(s_[t]) & np.isfinite(f_[t])
        if mask is not None:
            m = m & mask[t]
        n = int(m.sum())
        if n < min_n:
            continue
        rs = rankdata(s_[t, m]).astype(float)
        rf = rankdata(f_[t, m]).astype(float)
        rs -= rs.mean()
        rf -= rf.mean()
        d = np.sqrt((rs @ rs) * (rf @ rf))
        if d <= 0:
            continue
        out[t] = (rs @ rf) / d
    return out


def newey_west_t(x: np.ndarray, lags: Optional[int] = None) -> float:
    """t-statistic of the mean, HAC-corrected.

    With overlapping forward windows the raw t-stat is inflated by roughly
    sqrt(horizon); pass ``lags >= horizon - 1``.
    """
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 10:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lags = max(0, min(lags, n - 2))
    e = v - v.mean()
    var = (e @ e) / n
    for L in range(1, lags + 1):
        var += 2.0 * (1.0 - L / (lags + 1.0)) * ((e[L:] @ e[:-L]) / n)
    if not np.isfinite(var) or var <= 0:
        return float("nan")
    return float(v.mean() / np.sqrt(var / n))


def sampling(ic: np.ndarray, horizon: int) -> Dict[str, float]:
    """Infer how the IC series was sampled, and how much of it is independent.

    A length-T series with NaN on the dates the signal was not published tells
    you its own cadence. That matters because the overlap correction depends on
    it: a signal evaluated every h-th step is already non-overlapping, and
    dividing its count by h a second time understates the sample by a factor of
    h. Getting this wrong in the safe direction is still getting it wrong -- it
    turned 83 independent observations into "effective n=4" in a real report.
    """
    raw = np.asarray(ic, float)
    idx = np.flatnonzero(np.isfinite(raw))
    if idx.size == 0:
        return {"n": 0.0, "spacing": 1.0, "overlap": 1.0, "n_eff": 0.0, "lags": 0.0, "period": 1.0}
    spacing = float(np.median(np.diff(idx))) if idx.size > 1 else 1.0
    spacing = max(spacing, 1.0)
    overlap = max(1.0, horizon / spacing)      # consecutive obs sharing a window
    return {"n": float(idx.size), "spacing": spacing, "overlap": overlap,
            "n_eff": idx.size / overlap,
            "lags": float(int(np.ceil(overlap)) - 1),
            "period": float(max(spacing, horizon))}


def ic_summary(ic: np.ndarray, horizon: int = 1, periods_per_year: int = 252) -> Dict[str, float]:
    sm = sampling(ic, horizon)
    v = np.asarray(ic, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "icir": np.nan, "t_nw": np.nan,
                "n_eff": 0.0, "spacing": 1.0}
    sd = float(v.std(ddof=1)) if v.size > 1 else np.nan
    lags = int(sm["lags"])
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "std": sd,
        # Annualise on independent periods, not on rows.
        "icir": float(v.mean() / sd * np.sqrt(periods_per_year / sm["period"]))
                if np.isfinite(sd) and sd > 0 else np.nan,
        "t_nw": newey_west_t(v, lags=lags if lags > 0 else None),
        "n_eff": float(sm["n_eff"]),
        "spacing": float(sm["spacing"]),
    }


def deflated_threshold(n_trials: int, alpha: float = 0.05) -> float:
    """Bonferroni-style |t| threshold after searching ``n_trials`` candidates.

    Report the number of candidates you actually looked at, including the ones
    you discarded early -- that is the number that governs the threshold.
    """
    from scipy.stats import norm

    n_trials = max(1, int(n_trials))
    return float(norm.ppf(1.0 - alpha / (2.0 * n_trials)))
