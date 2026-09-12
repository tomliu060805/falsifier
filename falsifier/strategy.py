"""Portfolio-level claims.

A cross-sectional signal is judged by its IC; a strategy is judged by what a
book that traded it would have earned. The two need different nulls, and the
difference is not cosmetic. Shuffling a signal tells you nothing about a
rotation rule, because most of what a rotation rule does is decided by how
often it trades and how wide it holds -- and a random book with the same
turnover frequently earns the same return.

So the null here is not "random names". It is "random names, rebalanced on the
same dates, holding the same count, retaining the same number of positions from
one period to the next". That construction leaves turnover and cost identical
and isolates the only claim actually being made: that *these* names were the
ones to hold.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .econ import trade_metrics
from .nulls import empirical_p, null_distribution, percentile_of
from .prereg import Prereg
from .verdict import FAIL, INCONCLUSIVE, NA, PASS, Check, Report


def backtest(selection: np.ndarray, ret: np.ndarray, cost_bp: float) -> Dict[str, np.ndarray]:
    """Equal-weight book over ``selection``; weights drift between rebalances.

    ``selection[t]`` is what is held from t to t+1, so it earns ``ret[t+1]``.
    One-way turnover is charged on both legs of every change.
    """
    sel = np.asarray(selection, bool)
    r = np.nan_to_num(np.asarray(ret, float), nan=0.0)
    T, N = sel.shape
    w = np.zeros(N)
    gross, turn = np.zeros(T), np.zeros(T)
    prev = np.zeros(N, bool)
    for t in range(T - 1):
        cur = sel[t]
        if cur.any() and not np.array_equal(cur, prev):
            new = np.zeros(N)
            new[cur] = 1.0 / cur.sum()
            turn[t] = float(np.abs(new - w).sum() / 2.0)
            w = new
        prev = cur
        step = r[t + 1]
        gross[t] = float(w @ step)
        tot = float((w * (1.0 + step)).sum())
        if tot > 0:                      # let the weights drift, as a real book does
            w = w * (1.0 + step) / tot
    net = gross - turn * 2.0 * cost_bp / 1e4
    return {"gross": gross, "net": net, "turnover": turn}


def perf(net: np.ndarray, periods_per_year: int = 252) -> Dict[str, float]:
    x = np.asarray(net, float)
    x = x[np.isfinite(x)]
    if x.size < 20:
        return {"ann": np.nan, "sharpe": np.nan, "mdd": np.nan, "n": int(x.size)}
    ny = x.size / periods_per_year
    nav = np.cumprod(1.0 + x)
    return {"ann": float(nav[-1] ** (1 / ny) - 1.0),
            "sharpe": float(x.mean() / x.std(ddof=1) * np.sqrt(periods_per_year)) if x.std(ddof=1) > 0 else np.nan,
            "mdd": float((nav / np.maximum.accumulate(nav) - 1.0).min()),
            "n": int(x.size)}


def matched_selection_null(selection: np.ndarray, mask: np.ndarray, seed=0) -> np.ndarray:
    """A random book with the strategy's own turnover.

    On every date the strategy changes its holdings, the null changes its own by
    the same amount: it retains as many of its current positions as the strategy
    retained of hers, drawn at random from those still investable, and fills the
    remainder from the rest of the universe. Cost structure identical; only the
    identity of the names differs.
    """
    g = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    sel = np.asarray(selection, bool)
    T, N = sel.shape
    out = np.zeros_like(sel)
    prev_real = np.zeros(N, bool)
    prev_null = np.zeros(N, bool)
    for t in range(T):
        cur = sel[t]
        if not cur.any() or np.array_equal(cur, prev_real):
            out[t] = prev_null
        else:
            n = int(cur.sum())
            keep = int((cur & prev_real).sum())
            alive = np.flatnonzero(prev_null & mask[t])
            keep = min(keep, alive.size)
            kept = g.choice(alive, size=keep, replace=False) if keep > 0 else np.empty(0, int)
            pool = np.setdiff1d(np.flatnonzero(mask[t]), kept)
            take = min(n - keep, pool.size)
            new = g.choice(pool, size=take, replace=False) if take > 0 else np.empty(0, int)
            row = np.zeros(N, bool)
            row[kept] = True
            row[new] = True
            out[t] = row
        prev_real = cur
        prev_null = out[t]
    return out


def free_selection_null(selection: np.ndarray, mask: np.ndarray, seed=0) -> np.ndarray:
    """A random book that rebalances on the same dates but retains nothing.

    The null most studies run, and the reason so many rotation rules look good:
    it churns far harder than the strategy, pays far more cost, and therefore
    loses to almost anything. Reported here only as a contrast to the matched
    null -- clearing it is not evidence.
    """
    g = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    sel = np.asarray(selection, bool)
    T, N = sel.shape
    out = np.zeros_like(sel)
    prev_real = np.zeros(N, bool)
    prev_null = np.zeros(N, bool)
    for t in range(T):
        cur = sel[t]
        if not cur.any() or np.array_equal(cur, prev_real):
            out[t] = prev_null
        else:
            pool = np.flatnonzero(mask[t])
            take = min(int(cur.sum()), pool.size)
            row = np.zeros(N, bool)
            if take:
                row[g.choice(pool, size=take, replace=False)] = True
            out[t] = row
        prev_real = cur
        prev_null = out[t]
    return out


@dataclass
class StrategyStudy:
    claim: str
    selection: np.ndarray            # (T, N) bool: held from t to t+1
    ret: np.ndarray                  # (T, N)
    mask: np.ndarray                 # (T, N) investable at t
    benchmark: Optional[np.ndarray] = None   # (T,) per-period benchmark return
    benchmark_name: str = "benchmark"
    cost_bp: float = 0.0
    periods_per_year: int = 252
    n_candidates_searched: int = 1
    dates: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.selection = np.asarray(self.selection, bool)
        self.mask = np.asarray(self.mask, bool)
        self.ret = np.asarray(self.ret, float)


def run_strategy(study: StrategyStudy, prereg: Optional[Prereg] = None,
                 n_draws: int = 500, seed: int = 0, metric: str = "sharpe",
                 required_pct: float = 95.0, verbose: bool = True) -> Report:
    say = (lambda m: print(m, flush=True)) if verbose else (lambda m: None)
    rep = Report(claim=study.claim, prereg_id=prereg.id if prereg else None)

    if prereg is None:
        rep.add(Check("P0", "process", "pre-registration", INCONCLUSIVE,
                      detail="no pre-registration: the criterion cannot be shown to predate the result"))
    else:
        rep.add(Check("P0", "process", "pre-registration", PASS,
                      detail=f"{prereg.id} frozen {prereg.created_utc}; "
                             f"{prereg.n_candidates_searched} candidate(s) declared"))

    real = backtest(study.selection, study.ret, study.cost_bp)
    rp = perf(real["net"], study.periods_per_year)
    ann_turn = float(real["turnover"].sum() / (rp["n"] / study.periods_per_year))
    say(f"[1/3] strategy: ann {rp['ann']:+.2%} sharpe {rp['sharpe']:.2f} "
        f"mdd {rp['mdd']:.1%} turnover {ann_turn:.1f}x/yr")

    # ---- economic ----------------------------------------------------------
    if study.benchmark is not None:
        bp = perf(np.asarray(study.benchmark, float)[: len(real["net"])], study.periods_per_year)
        beat = rp["ann"] - bp["ann"]
        rep.add(Check("SE1", "economic", f"beats {study.benchmark_name} after cost",
                      PASS if beat > 0 else FAIL, statistic=beat * 100, threshold=0.0,
                      detail=(f"strategy {rp['ann']:+.2%} (sharpe {rp['sharpe']:.2f}, mdd {rp['mdd']:.1%}) "
                              f"vs {study.benchmark_name} {bp['ann']:+.2%} (sharpe {bp['sharpe']:.2f}, "
                              f"mdd {bp['mdd']:.1%}); turnover {ann_turn:.1f}x/yr at {study.cost_bp}bp"),
                      evidence={"strategy": rp, "benchmark": bp, "ann_turnover": ann_turn}))
    else:
        rep.add(Check("SE1", "economic", "beats benchmark after cost", NA, blocking=False,
                      detail="no benchmark supplied"))

    # ---- mechanistic -------------------------------------------------------
    def stat_of(sel):
        return perf(backtest(sel, study.ret, study.cost_bp)["net"], study.periods_per_year)[metric]

    say(f"[2/3] free-selection null ({n_draws} draws) ...")
    free = null_distribution(lambda s: stat_of(free_selection_null(study.selection, study.mask, seed=s)),
                             n_draws=n_draws, seed=seed)
    free_turn = float(backtest(free_selection_null(study.selection, study.mask, seed=seed),
                               study.ret, study.cost_bp)["turnover"].sum() / (rp["n"] / study.periods_per_year))
    pf = percentile_of(rp[metric], free)
    rep.add(Check("SM0", "mechanistic", "free-selection null", PASS if pf >= required_pct else FAIL,
                  blocking=False, statistic=pf, threshold=required_pct,
                  detail=(f"{pf:.0f}th pct of random books that retain nothing "
                          f"({free_turn:.1f}x/yr vs the strategy's {ann_turn:.1f}x). "
                          "Clearing this is not evidence: the null is paying for turnover the "
                          "strategy never spends"),
                  evidence={"percentile": pf, "null_median": float(np.nanmedian(free)),
                            "null_turnover": free_turn}))

    say(f"[3/3] turnover-matched null ({n_draws} draws) ...")
    matched = null_distribution(lambda s: stat_of(matched_selection_null(study.selection, study.mask, seed=s)),
                                n_draws=n_draws, seed=seed + 1)
    m_turn = float(backtest(matched_selection_null(study.selection, study.mask, seed=seed),
                            study.ret, study.cost_bp)["turnover"].sum() / (rp["n"] / study.periods_per_year))
    pm = percentile_of(rp[metric], matched)
    p_emp = empirical_p(rp[metric], matched)
    rep.add(Check("SM1", "mechanistic", "turnover-matched null", PASS if pm >= required_pct else FAIL,
                  statistic=pm, threshold=required_pct,
                  detail=(f"{pm:.0f}th pct of random books with the strategy's own turnover "
                          f"({m_turn:.1f}x/yr vs {ann_turn:.1f}x), p={p_emp:.3f}; "
                          f"null median {metric} {np.nanmedian(matched):.3f} vs strategy {rp[metric]:.3f}"
                          + ("" if pm >= required_pct else
                             ". Picking these names is not what produced the result")),
                  evidence={"percentile": pm, "p_emp": p_emp, "metric": metric,
                            "null_median": float(np.nanmedian(matched)),
                            "strategy": rp[metric], "null_turnover": m_turn,
                            "strategy_turnover": ann_turn}))

    blocks = np.add.reduceat(real["net"], np.r_[0, np.flatnonzero(real["turnover"] > 0)[1:]]) * 1e4 \
        if (real["turnover"] > 0).sum() > 1 else real["net"] * 1e4
    tm = trade_metrics(blocks)
    rep.add(Check("SE2", "economic", "per-trade block", PASS, blocking=False, statistic=tm["avg_bp"],
                  detail=(f"n={tm['n_trades']} win={tm['win_rate']:.1%} avg={tm['avg_bp']:+.1f}bp "
                          f"payoff={tm['payoff_ratio']:.2f} PF={tm['profit_factor']:.2f} "
                          f"worst={tm['worst_bp']:.0f}bp"),
                  evidence=tm))
    return rep
