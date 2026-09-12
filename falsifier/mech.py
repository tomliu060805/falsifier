"""The mechanistic axis: is the signal the thing you say it is?

Statistical significance says the number is unlikely under "nothing". These
checks ask the harder question -- unlikely under *what you already had*. Each
one removes a different rival explanation:

  M0  is the harness even wired up            (destroy everything)
  M1  is it just the covariates               (permute within their buckets)
  M2  is it really about *this* asset         (relabel who is who)
  M3  is anything left after known factors    (orthogonalise, then retest)
  M4  is the machinery earning its keep       (beat the naive version)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .nulls import (cs_shuffle, empirical_p, identity_permutation,
                    matched_permutation, null_distribution, orthogonalize,
                    percentile_of)
from .stats import newey_west_t, rank_ic
from .verdict import FAIL, INCONCLUSIVE, NA, PASS, Check


def _mean_ic(signal, fwd, mask, min_n=30) -> float:
    v = rank_ic(signal, fwd, mask, min_n=min_n)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def orient(signal: np.ndarray, fwd: np.ndarray, mask: np.ndarray, min_n: int = 30) -> np.ndarray:
    """Flip the signal so a positive IC means 'works'. Direction is a
    convention; it must be fixed before nulls, not chosen after seeing them."""
    ic = _mean_ic(signal, fwd, mask, min_n)
    return signal * (1.0 if (np.isfinite(ic) and ic >= 0) else -1.0)


def m0_harness(signal, fwd, mask, n_draws: int = 60, seed: int = 0, min_n: int = 30) -> Check:
    real = _mean_ic(signal, fwd, mask, min_n)
    draws = null_distribution(lambda s: _mean_ic(cs_shuffle(signal, mask, seed=s), fwd, mask, min_n),
                              n_draws=n_draws, seed=seed)
    pct = percentile_of(real, draws)
    ok = pct >= 90.0
    return Check("M0", "mechanistic", "harness sanity (full shuffle)", PASS if ok else FAIL,
                 statistic=pct, threshold=90.0,
                 detail=(f"real IC {real:.4f} sits at the {pct:.0f}th pct of a fully shuffled signal"
                         if ok else
                         f"real IC {real:.4f} is indistinguishable from a shuffled signal "
                         f"({pct:.0f}th pct) -- there is nothing here to test further"),
                 evidence={"real": real, "null_mean": float(np.nanmean(draws)), "n_draws": n_draws})


def m1_matched_null(signal, fwd, mask, covariates: Dict[str, np.ndarray],
                    n_draws: int = 200, seed: int = 0, required_pct: float = 95.0,
                    n_bucket: int = 5, max_explained: float = 0.8, min_n: int = 30) -> Check:
    """Permute inside buckets of everything that decided who entered.

    The list of covariates is the whole check. Match on size when the real
    selector was liquidity and the null will happily confirm your result.
    """
    if not covariates:
        return Check("M1", "mechanistic", "matched null", NA, blocking=False,
                     detail="no covariates declared -- the strongest null did not run")
    real = _mean_ic(signal, fwd, mask, min_n)
    draws = null_distribution(
        lambda s: _mean_ic(matched_permutation(signal, covariates, mask, n_bucket=n_bucket, seed=s), fwd, mask, min_n),
        n_draws=n_draws, seed=seed)
    pct = percentile_of(real, draws)
    p = empirical_p(real, draws)
    null_mean = float(np.nanmean(draws))
    # Two ways to fail, and the second one matters more than it looks.
    #
    # Bucketing a continuous covariate leaves a little of its ordering inside
    # each cell, so a signal that simply *is* the covariate still edges past its
    # own matched null and lands at a high percentile. The percentile answers
    # "is the excess reliable"; it does not answer "is the excess worth having".
    # A null that knows nothing but the covariates and still reproduces most of
    # the effect has already explained the result, however small its error bars.
    explained = (null_mean / real) if (np.isfinite(real) and abs(real) > 1e-12) else np.nan
    ok_pct = pct >= required_pct
    ok_size = not (np.isfinite(explained) and explained > max_explained)
    ok = ok_pct and ok_size
    names = ", ".join(sorted(covariates))
    if ok:
        why = (f"beats a null matched on [{names}] at the {pct:.0f}th pct (p={p:.3f}); "
               f"the matched null reproduces {explained:.0%} of the effect")
    elif not ok_pct:
        why = (f"only the {pct:.0f}th pct of a null matched on [{names}] (p={p:.3f}): "
               "the covariates already explain this")
    else:
        why = (f"a null that knows only [{names}] already reproduces {explained:.0%} of the "
               f"effect (IC {null_mean:.4f} of {real:.4f}). The {pct:.0f}th pct says the "
               "remaining sliver is reliable, not that it is worth claiming")
    return Check("M1", "mechanistic", "matched null", PASS if ok else FAIL,
                 statistic=pct, threshold=required_pct, detail=why,
                 evidence={"real": real, "null_mean": null_mean, "percentile": pct,
                           "p_emp": p, "explained_by_null": explained,
                           "matched_on": sorted(covariates)})


def m2_identity_null(signal, fwd, mask, n_draws: int = 200, seed: int = 0,
                     required_pct: float = 95.0, groups: Optional[np.ndarray] = None,
                     min_n: int = 30) -> Check:
    """Relabel the assets. Keeps the distribution and the persistence, kills
    only the claim that a particular name was the one to hold.

    Advisory, and the reason is worth stating: once a real cross-sectional edge
    exists, permuting identity destroys it by construction, so this passes
    whenever M0 passes and fails only when the edge was already near zero --
    which M0 reaches first. It corroborates rather than decides. The question it
    reads as asking, whether the effect is really cross-sectional at all, is
    answered by M9.
    """
    real = _mean_ic(signal, fwd, mask, min_n)
    draws = null_distribution(
        lambda s: _mean_ic(identity_permutation(signal, seed=s, groups=groups), fwd, mask, min_n),
        n_draws=n_draws, seed=seed)
    pct = percentile_of(real, draws)
    ok = pct >= required_pct
    return Check("M2", "mechanistic", "identity null", PASS if ok else FAIL,
                 blocking=False, statistic=pct, threshold=required_pct,
                 detail=("the asset-level mapping carries the result"
                         if ok else
                         f"{pct:.0f}th pct against relabelled assets: the result comes from the "
                         "signal's shape and timing, not from which asset it points at"),
                 evidence={"real": real, "null_mean": float(np.nanmean(draws)), "percentile": pct})


def m3_orthogonalize(signal, fwd, mask, controls: Sequence[np.ndarray],
                     control_names: Optional[Sequence[str]] = None,
                     horizon: int = 1, min_t: float = 2.0, min_n: int = 30) -> Check:
    """Regress out the known factors and see what is left."""
    if not controls:
        return Check("M3", "mechanistic", "residual after known factors", NA, blocking=False,
                     detail="no controls declared")
    names = ", ".join(control_names or [f"c{i}" for i in range(len(controls))])
    base_ic = rank_ic(signal, fwd, mask, min_n=min_n)
    res = orthogonalize(signal, controls, mask)
    res_ic = rank_ic(res, fwd, mask, min_n=min_n)
    b, r = float(np.nanmean(base_ic)), float(np.nanmean(res_ic))
    t = newey_west_t(res_ic, lags=(horizon - 1) if horizon > 1 else None)
    retained = r / b if b not in (0.0,) and np.isfinite(b) else np.nan
    ok = abs(t) >= min_t
    return Check("M3", "mechanistic", "residual after known factors", PASS if ok else FAIL,
                 statistic=t, threshold=min_t,
                 detail=(f"IC {b:.4f} -> {r:.4f} ({retained:.0%} retained) after removing [{names}], "
                         f"residual t={t:.2f}"
                         + ("" if ok else " -- nothing independent survives")),
                 evidence={"ic_raw": b, "ic_residual": r, "retained": retained,
                           "t_residual": t, "controls": names})


def m4_beats_naive(signal, baseline, fwd, mask, horizon: int = 1, min_t: float = 2.0,
                   label: str = "naive baseline", min_n: int = 30) -> Check:
    """Paired test against the simple version of the same idea.

    Elaborate machinery has to be shown to add something over the obvious
    construction; a higher headline number is not that demonstration.
    """
    if baseline is None:
        return Check("M4", "mechanistic", "beats naive baseline", NA, blocking=False,
                     detail="no naive baseline declared -- complexity is unjustified by default")
    a = rank_ic(signal, fwd, mask, min_n=min_n)
    b = rank_ic(baseline, fwd, mask, min_n=min_n)
    d = a - b
    t = newey_west_t(d, lags=(horizon - 1) if horizon > 1 else None)
    ok = t >= min_t
    return Check("M4", "mechanistic", "beats naive baseline", PASS if ok else FAIL,
                 blocking=False, statistic=t, threshold=min_t,
                 detail=(f"IC {np.nanmean(a):.4f} vs {label} {np.nanmean(b):.4f}, "
                         f"paired t={t:.2f}"
                         + ("" if ok else " -- the extra machinery is not paying for itself")),
                 evidence={"ic_signal": float(np.nanmean(a)), "ic_baseline": float(np.nanmean(b)), "t_diff": t})
