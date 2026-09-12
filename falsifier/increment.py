"""What adding one thing did to the curve, and to each year.

The question "should this factor go in" is not the question "is this factor
real", and it is answered with different evidence. A candidate can have a
perfectly good IC and still be worth nothing to a book that already holds
something correlated with it; it can raise the headline and take the bad years
with it; and it can add three points that all come from one year nobody will
see again.

So the unit here is the pair -- the book without it and the book with it -- and
the output is the year-by-year delta rather than a single number. That is the
form these decisions are actually made in, and it is the form that makes the
two failure modes visible: an improvement concentrated in one year, and an
improvement that buys the good years by worsening the bad ones.

Accounting follows the compounding convention throughout: excess is
(1+strategy)/(1+benchmark)-1 chained, never a difference of arithmetic means,
and never an intraday-only segment presented as a curve.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .econ import trade_metrics
from .nulls import null_distribution, percentile_of
from .verdict import FAIL, INCONCLUSIVE, NA, PASS, Check


def _years(dates: np.ndarray) -> np.ndarray:
    d = np.asarray(dates)
    if d.dtype.kind in "MU":
        return np.array([str(x)[:4] for x in d])
    v = d.astype(np.int64)
    return (v // 10000).astype(str) if v.max() > 10000 else v.astype(str)


def excess(strategy: np.ndarray, benchmark: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-period geometric excess. Without a benchmark, the strategy itself."""
    s = np.nan_to_num(np.asarray(strategy, float), nan=0.0)
    if benchmark is None:
        return s
    b = np.nan_to_num(np.asarray(benchmark, float), nan=0.0)
    return (1.0 + s) / (1.0 + b) - 1.0


def _compound(x: np.ndarray) -> float:
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    return float(np.prod(1.0 + v) - 1.0) if v.size else float("nan")


@dataclass
class Increment:
    """A book without the candidate, and the same book with it."""

    dates: np.ndarray
    baseline: np.ndarray
    combined: np.ndarray
    benchmark: Optional[np.ndarray] = None
    label_baseline: str = "baseline"
    label_combined: str = "with candidate"
    periods_per_year: int = 252
    trades: Optional[np.ndarray] = None
    """Rebalance indices, so the per-trade block counts holding periods rather
    than calendar steps."""

    def __post_init__(self) -> None:
        n = min(len(self.dates), len(self.baseline), len(self.combined))
        self.dates = np.asarray(self.dates)[:n]
        self.baseline = np.asarray(self.baseline, float)[:n]
        self.combined = np.asarray(self.combined, float)[:n]
        if self.benchmark is not None:
            self.benchmark = np.asarray(self.benchmark, float)[:n]

    # ---- series -----------------------------------------------------------
    def curves(self) -> Dict[str, np.ndarray]:
        eb = excess(self.baseline, self.benchmark)
        ec = excess(self.combined, self.benchmark)
        nav_b = np.cumprod(1.0 + np.nan_to_num(eb))
        nav_c = np.cumprod(1.0 + np.nan_to_num(ec))
        return {"excess_baseline": eb, "excess_combined": ec,
                "nav_baseline": nav_b, "nav_combined": nav_c,
                "nav_delta": nav_c / np.where(nav_b == 0, np.nan, nav_b)}

    def yearly(self) -> List[Dict[str, float]]:
        eb = excess(self.baseline, self.benchmark)
        ec = excess(self.combined, self.benchmark)
        yrs = _years(self.dates)
        rows = []
        for y in sorted(set(yrs.tolist())):
            m = yrs == y
            b, c = _compound(eb[m]), _compound(ec[m])
            rows.append({"year": y, "n": int(m.sum()), "baseline": b, "combined": c,
                         "delta": c - b})
        return rows

    def per_trade(self) -> Dict[str, Dict[str, float]]:
        """The block that must accompany any reported result."""
        out = {}
        for name, series in ((self.label_baseline, self.baseline),
                             (self.label_combined, self.combined)):
            ex = excess(series, self.benchmark)
            if self.trades is not None and len(self.trades) > 1:
                edges = list(np.asarray(self.trades, int)) + [len(ex)]
                blocks = [_compound(ex[edges[i]:edges[i + 1]]) * 1e4
                          for i in range(len(edges) - 1)]
            else:
                blocks = (ex * 1e4).tolist()
            out[name] = trade_metrics(np.array(blocks, float))
        return out

    def summary(self) -> Dict[str, float]:
        rows = self.yearly()
        eb = excess(self.baseline, self.benchmark)
        ec = excess(self.combined, self.benchmark)
        ny = max(len(eb) / self.periods_per_year, 1e-9)
        ann = lambda x: (1.0 + _compound(x)) ** (1 / ny) - 1.0
        neg_b = sum(1 for r in rows if r["baseline"] < 0)
        neg_c = sum(1 for r in rows if r["combined"] < 0)
        up = sum(1 for r in rows if r["delta"] > 0)
        return {"ann_baseline": ann(eb), "ann_combined": ann(ec),
                "ann_delta": ann(ec) - ann(eb),
                "total_delta": _compound(ec) - _compound(eb),
                "years": len(rows), "years_improved": up,
                "negative_years_baseline": neg_b, "negative_years_combined": neg_c}


def render_yearly(inc: Increment, width: int = 72) -> str:
    rows = inc.yearly()
    s = inc.summary()
    out = ["=" * width,
           f"{'year':<8}{inc.label_baseline:>14}{inc.label_combined:>16}{'delta':>12}",
           "-" * width]
    for r in rows:
        mark = "" if r["delta"] >= 0 else "  <-"
        out.append(f"{r['year']:<8}{r['baseline']:>13.2%}{r['combined']:>16.2%}"
                   f"{r['delta']:>12.2%}{mark}")
    out += ["-" * width,
            f"{'annualised':<8}{s['ann_baseline']:>13.2%}{s['ann_combined']:>16.2%}"
            f"{s['ann_delta']:>12.2%}",
            f"years improved {s['years_improved']}/{s['years']}   "
            f"negative years {s['negative_years_baseline']} -> {s['negative_years_combined']}"]
    for name, tm in inc.per_trade().items():
        out.append(f"{name:<16}n={tm['n_trades']} win={tm['win_rate']:.1%} "
                   f"avg={tm['avg_bp']:+.1f}bp payoff={tm['payoff_ratio']:.2f} "
                   f"PF={tm['profit_factor']:.2f} worst={tm['worst_bp']:.0f}bp")
    out.append("=" * width)
    return "\n".join(out)


def i1_incremental_contribution(inc: Increment, max_year_share: float = 0.8,
                                allow_worse_years: int = 0) -> Check:
    """I1 -- did adding it improve the curve without taking the bad years with it?

    Three ways an addition fails that a headline number hides. It can be
    negative overall. It can be positive overall while turning a year that was
    merely flat into a losing one, which is a different book from the one that
    was approved. And it can be positive because of a single year -- the same
    concentration problem as a strategy whose record is four days, one level up.
    """
    rows = inc.yearly()
    s = inc.summary()
    if len(rows) < 3:
        return Check("I1", "economic", "incremental contribution", INCONCLUSIVE,
                     detail=f"only {len(rows)} year(s); nothing to read year by year")
    deltas = np.array([r["delta"] for r in rows])
    pos = deltas[deltas > 0].sum()
    top_share = float(deltas.max() / pos) if pos > 0 else np.nan
    worse_years = s["negative_years_combined"] - s["negative_years_baseline"]

    reasons = []
    if s["ann_delta"] <= 0:
        reasons.append(f"annualised excess moves {s['ann_delta']:+.2%}")
    if worse_years > allow_worse_years:
        reasons.append(f"losing years go from {s['negative_years_baseline']} to "
                       f"{s['negative_years_combined']}")
    if np.isfinite(top_share) and top_share > max_year_share:
        worst = rows[int(np.argmax(deltas))]["year"]
        reasons.append(f"{top_share:.0%} of the gain is {worst} alone")

    ev = {"summary": s, "yearly": rows, "top_year_share": top_share,
          "per_trade": inc.per_trade()}
    if reasons:
        return Check("I1", "economic", "incremental contribution", FAIL,
                     statistic=s["ann_delta"] * 100, threshold=0.0,
                     detail="; ".join(reasons), evidence=ev)
    return Check("I1", "economic", "incremental contribution", PASS,
                 statistic=s["ann_delta"] * 100, threshold=0.0,
                 detail=(f"annualised excess {s['ann_baseline']:+.2%} -> "
                         f"{s['ann_combined']:+.2%} ({s['ann_delta']:+.2%}); "
                         f"{s['years_improved']}/{s['years']} years improved, losing years "
                         f"{s['negative_years_baseline']} -> {s['negative_years_combined']}, "
                         f"largest single year is {top_share:.0%} of the gain"),
                 evidence=ev)


def i2_increment_null(inc: Increment, combined_of_seed: Callable[[int], np.ndarray],
                      n_draws: int = 200, seed: int = 0,
                      required_pct: float = 95.0) -> Check:
    """I2 -- compare the addition against adding something that knows nothing.

    Adding almost anything to a book changes its curve, and a change is not a
    contribution. The comparison that settles it is a candidate with the same
    construction and the same turnover, carrying no information -- a random
    signal, a shuffled version of the real one, a random weight vector. If the
    real candidate lands inside that distribution, what was measured was the act
    of adding, not the thing added.
    """
    real = inc.summary()["ann_delta"]

    def stat(s: int) -> float:
        alt = Increment(dates=inc.dates, baseline=inc.baseline,
                        combined=combined_of_seed(s), benchmark=inc.benchmark,
                        periods_per_year=inc.periods_per_year)
        return alt.summary()["ann_delta"]

    draws = null_distribution(stat, n_draws=n_draws, seed=seed)
    pct = percentile_of(real, draws)
    ok = pct >= required_pct
    return Check("I2", "economic", "increment against a null candidate",
                 PASS if ok else FAIL, statistic=pct, threshold=required_pct,
                 detail=(f"the addition sits at the {pct:.0f}th pct of adding an "
                         f"uninformative candidate ({real:+.2%} vs null median "
                         f"{np.nanmedian(draws):+.2%})"
                         if ok else
                         f"adding an uninformative candidate does about as well "
                         f"({pct:.0f}th pct: {real:+.2%} against a null median of "
                         f"{np.nanmedian(draws):+.2%}) -- what was measured is the act of "
                         "adding, not the thing added"),
                 evidence={"real": real, "percentile": pct,
                           "null_median": float(np.nanmedian(draws)), "n_draws": n_draws})


def run_increment(inc: Increment, combined_of_seed: Optional[Callable[[int], np.ndarray]] = None,
                  n_draws: int = 200, seed: int = 0, chart: Optional[str] = None,
                  title: Optional[str] = None, verbose: bool = True):
    """Judge an addition, and draw what it did.

    Deliberately a separate entry point from `run`. "Is this factor real" and
    "should this factor go in" are different questions with different evidence:
    a candidate can be perfectly real and worth nothing to a book that already
    holds something correlated with it, and a candidate can improve the headline
    while taking the bad years with it. Running the signal gauntlet answers the
    first; this answers the second.
    """
    from .verdict import Report

    rep = Report(claim=f"Adding {inc.label_combined} improves {inc.label_baseline}")
    say = (lambda m: print(m, flush=True)) if verbose else (lambda m: None)

    say("[1/2] year by year ...")
    rep.add(i1_incremental_contribution(inc))

    if combined_of_seed is not None:
        say(f"[2/2] null candidate ({n_draws} draws) ...")
        rep.add(i2_increment_null(inc, combined_of_seed, n_draws=n_draws, seed=seed))
    else:
        rep.add(Check("I2", "economic", "increment against a null candidate", NA,
                      blocking=False,
                      detail="no null candidate supplied. Adding almost anything moves a "
                             "curve, and a move is not a contribution -- pass a candidate "
                             "with the same construction and turnover that knows nothing"))

    if chart:
        from .charts import plot_increment
        path = plot_increment(inc, chart, title=title)
        rep.notes.append(f"chart written to {path}")
    rep.notes.append("compounded excess throughout; a difference of arithmetic means "
                     "would not be the number a book earns")
    return rep
