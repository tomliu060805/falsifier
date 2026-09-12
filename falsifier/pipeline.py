"""A rolling-window fit that can hand the referee its decisive audits.

`A0` (truncation rebuild) and `A1` (label-shuffle refit) are the two checks
that actually settle whether a signal leaked, and both are unavailable for a
frozen array of numbers: one needs the pipeline rebuilt from truncated
history, the other needs it refitted on destroyed labels. Most research
pipelines are the same shape underneath -- standardise a feature block, fit a
rolling regression on labels whose horizon has closed, predict the next
cross-section -- so wrapping that shape once makes the decisive audits
available to all of them.

The knobs that matter are `standardize` and `window`, because each has one
setting that is correct and one that reproduces a bug seen in the wild:

  standardize="cross-section"  per-date, uses nothing from other dates
  standardize="full-sample"    per-name over the whole sample -- A0 catches it

  window="causal"              trains only on labels whose horizon has closed
  window="contaminated"        training window reaches the predicted date
  window="full-sample"         one fit on everything -- A1 catches both
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.stats import norm, rankdata

import warnings

from .stats import forward_returns


def _xs_z(a: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score, clipped. Uses only the row it is given."""
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.nanmean(a, axis=0)
        s = np.nanstd(a, axis=0)
        return np.clip((a - m) / np.where(s > 0, s, np.nan), -5, 5)


@dataclass
class RollingFit:
    features: np.ndarray          # (T, N, K)
    ret: np.ndarray               # (T, N)
    mask: np.ndarray              # (T, N) tradable at t
    horizon: int = 1
    fitwin: int = 250
    ridge: float = 1e-2
    stride: int = 1
    min_n: int = 30
    standardize: str = "cross-section"
    window: str = "causal"

    def __post_init__(self) -> None:
        self.features = np.asarray(self.features, float)
        self.ret = np.asarray(self.ret, float)
        self.mask = np.asarray(self.mask, bool)
        self.T, self.N, self.K = self.features.shape
        self._fwd = forward_returns(self.ret, self.horizon)
        self._full_mu = np.nanmean(self.features, axis=0)   # (N, K) -- the leaky statistic
        self._full_sd = np.nanstd(self.features, axis=0)

    # ---- feature construction ---------------------------------------------
    def _feat(self, t: int, upto: Optional[int] = None) -> np.ndarray:
        """Features at date t. ``upto`` truncates the history any statistic may see."""
        f = self.features[t]
        if self.standardize == "full-sample":
            hi = self.T if upto is None else upto + 1
            with np.errstate(invalid="ignore", divide="ignore"):
                mu = np.nanmean(self.features[:hi], axis=0)
                sd = np.nanstd(self.features[:hi], axis=0)
                return np.clip((f - mu) / np.where(sd > 0, sd, np.nan), -5, 5)
        return _xs_z(f)

    def _rows(self, t: int, fwd: np.ndarray, upto: Optional[int] = None):
        m = self.mask[t] & np.isfinite(fwd[t])
        f = self._feat(t, upto)
        m = m & np.isfinite(f).all(axis=1)
        n = int(m.sum())
        if n < self.min_n:
            return None
        X = np.column_stack([np.ones(n), np.nan_to_num(f[m])])
        y = norm.ppf(rankdata(fwd[t, m]) / (n + 1))     # normal-score the label
        return m, X, y

    def _grams(self, fwd: np.ndarray, upto: Optional[int] = None):
        # All-NaN warm-up rows are expected; they are skipped, not fixed.
        hi_t = self.T if upto is None else upto + 1
        XX = np.zeros((self.T, self.K + 1, self.K + 1))
        XY = np.zeros((self.T, self.K + 1))
        NN = np.zeros(self.T)
        for t in range(hi_t):
            r = self._rows(t, fwd, upto)
            if r is None:
                continue
            _, X, y = r
            XX[t] = X.T @ X
            XY[t] = X.T @ y
            NN[t] = X.shape[0]
        return np.cumsum(XX, 0), np.cumsum(XY, 0), np.cumsum(NN, 0)

    def _solve(self, A: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
        A = A.copy()
        tr = np.trace(A[1:, 1:])
        if not np.isfinite(tr) or tr <= 0:
            return None
        A[1:, 1:] += self.ridge * tr / self.K * np.eye(self.K)
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None

    def _bounds(self, t: int):
        """Training window. `causal` stops far enough back that every training
        label's horizon has already closed by t."""
        if self.window == "full-sample":
            return 0, self.T - 1
        hi = t if self.window == "contaminated" else t - self.horizon - 1
        return max(0, hi - self.fitwin), hi

    # ---- the three things the referee wants --------------------------------
    def predict(self, fwd: Optional[np.ndarray] = None, upto: Optional[int] = None) -> np.ndarray:
        fwd = self._fwd if fwd is None else fwd
        cxx, cxy, cnn = self._grams(fwd, upto)
        out = np.full((self.T, self.N), np.nan)
        last = self.T if upto is None else upto + 1
        start = 0 if self.window == "full-sample" else self.fitwin + self.horizon + 2
        for t in range(start, last, self.stride):
            lo, hi = self._bounds(t)
            if hi <= lo or (cnn[hi] - cnn[lo]) < self.min_n * 5:
                continue
            w = self._solve(cxx[hi] - cxx[lo], cxy[hi] - cxy[lo])
            if w is None:
                continue
            r = self._rows(t, fwd, upto)
            if r is None:
                continue
            m, X, _ = r
            out[t, m] = X @ w
        return out

    def recompute_at(self, t: int) -> np.ndarray:
        """The value for date t rebuilt from history truncated at t.

        Everything -- the standardising constants, the training window, the
        fit -- is recomputed as it would have been on the evening of t.
        """
        return self.predict(upto=t)[t]

    def refit(self, fwd_override: np.ndarray, draw: int = 0) -> np.ndarray:
        """Rerun the whole pipeline against a different label panel."""
        return self.predict(fwd=np.asarray(fwd_override, float))

    def probe_dates(self, n: int = 10) -> np.ndarray:
        """Dates that actually carry a prediction.

        A probe on a date the pipeline never predicts compares NaN against NaN,
        which A0 reports as inconclusive -- correct, but useless. With a stride
        the prediction grid is sparse, so the probes have to be drawn from it
        rather than from the calendar.
        """
        start = 0 if self.window == "full-sample" else self.fitwin + self.horizon + 2
        grid = np.arange(start, self.T - self.horizon - 1, max(1, self.stride))
        if grid.size == 0:
            return np.array([], int)
        take = np.unique(np.linspace(0, grid.size - 1, min(n, grid.size)).astype(int))
        return grid[take]
