#!/usr/bin/env python3
"""Self-check: five synthetic claims whose correct verdict is known in advance.

A referee that has never been shown to be right about a case with a known
answer is just an opinion with a table. So the repository ships targets built
to die in one specific way each, and the suite asserts both the verdict and
the cause of death -- getting REJECTED for the wrong reason teaches the wrong
lesson and is scored as a failure here.

  survivor    honest trailing-window momentum        -> SURVIVES
  noise       pure random cross-section              -> REJECTED by M0 (harness)
  leaky       same feature, window off by one step   -> REJECTED by A2 (boundary)
  size_proxy  a static characteristic in disguise    -> REJECTED by M1 (matched null)
  costly      the survivor traded ten times faster   -> REJECTED by E1 (cost)

and two portfolio-level targets, which need a different null entirely:

  skilled_book  a book holding the latent state       -> SURVIVES
  blind_book    same cadence, random names            -> REJECTED by SM1 (matched null)

and four pipeline targets, the only ones that can exercise the decisive audits,
since A0 needs a pipeline to rebuild and A1 needs one to refit:

  increment_real         a steady addition                 -> SURVIVES
  increment_is_noise     adding anything would have done it-> REJECTED by I2
  increment_one_year     the whole gain is one year        -> REJECTED by I1

  mechanism_falsified    the number holds, the story does not -> REJECTED by P6
  filtered_events        a trigger list that is the answer -> REJECTED by M7
  same_bar_fill          filled on the bar it was formed on-> REJECTED by P4
  external_fact_wrong    disagrees with the public record  -> REJECTED by P5
  spread_twice           the spread charged twice          -> REJECTED by E4
  no_capacity            more money than the names hold    -> REJECTED by E5

  bad_prints             moves that never happened         -> REJECTED by M5
  stale_cache            an input three weeks behind       -> REJECTED by M6
  sticky_label           a label that never reorders       -> REJECTED by S9

  universe_predates_index  an index that did not exist yet-> REJECTED by M14
  panel_saw_the_test     rows it was not allowed to see   -> REJECTED by P9
  bought_the_locked_board  a real edge you could not enter-> REJECTED by E6
  oracle_proposed_it     the idea source knew the future  -> REJECTED by P8
  guard_cannot_fire      a gate that could never go red   -> REJECTED by P7
  unidentified_fit       a ridge, not a point             -> REJECTED by S10
  leaky_control          orthogonalised against the answer-> REJECTED by M12

  frozen_stale           right about the first half only  -> REJECTED by M10
  frequency_flip         pays at 1x, reverses at 4x       -> REJECTED by M11

  stale_index            a real IC on untradable prints   -> REJECTED by A2
  bounce                 the spread coming back            -> REJECTED by E3
  overfit_knob           train up, valid down              -> REJECTED by S8
  config_drifted         the frozen file no longer binds   -> REJECTED by P3

  weak_after_search      real, and far too weak for 5000  -> REJECTED by S4
  frozen_covariate       an input that stopped updating   -> REJECTED by S7
  dead_panel             nothing there, and no power       -> INCONCLUSIVE (not REJECTED)
  seed_lucky             best of thirty runs, reported     -> REJECTED by S5

  clean                  causal window, per-date scaling  -> SURVIVES
  full-sample scaling    scaled on the whole sample       -> REJECTED by A0
  contaminated window    trains on the predicted date     -> REJECTED by A1
  one fit over history   coefficients from everything     -> REJECTED by A0

Run: python examples/selfcheck.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import falsifier as F

T, N = 1400, 240
RHO = 0.88          # persistence of the latent state
KAPPA = 0.0030      # how much of it reaches returns
SD_IDIO = 0.018
MU_SIZE = 0.0007    # a genuine size premium, so a size proxy scores above zero
SD_SIZE = 0.005
WINDOW = 5
SEED = 20260910


def build_panel(seed: int = SEED):
    """Returns driven by a persistent latent state plus a priced size factor.

    Only the latent state is forecastable from a trailing window; the size term
    exists so that a characteristic which knows nothing else still produces a
    positive IC -- which is the trap the matched null is there to catch.
    """
    g = np.random.default_rng(seed)
    a = np.zeros((T, N))
    eps = g.standard_normal((T, N)) * np.sqrt(1 - RHO ** 2)
    for t in range(1, T):
        a[t] = RHO * a[t - 1] + eps[t]
    z_size = g.standard_normal(N)
    f_size = g.normal(MU_SIZE, SD_SIZE, size=T)
    idio = g.standard_normal((T, N)) * SD_IDIO
    ret = np.empty((T, N))
    ret[0] = idio[0]
    ret[1:] = z_size[None, :] * f_size[1:, None] + KAPPA * a[:-1] + idio[1:]
    mask = np.ones((T, N), bool)
    mask[:WINDOW + 2] = False           # warm-up for the trailing window
    return ret, mask, z_size, a


def trailing_mean(ret: np.ndarray, window: int, end_offset: int = 0) -> np.ndarray:
    """Mean of returns over a window ending at ``t + end_offset``.

    ``end_offset=0`` is the honest construction. ``end_offset=1`` is the
    off-by-one that reads one step past the timestamp it is published under --
    the most common real leak, and the one A2 is built to locate.
    """
    T_, N_ = ret.shape
    out = np.full((T_, N_), np.nan)
    for t in range(T_):
        hi = t + end_offset
        lo = hi - window + 1
        if lo < 0 or hi >= T_:
            continue
        out[t] = np.nanmean(ret[lo:hi + 1], axis=0)
    return out


SURVIVOR_PREREG = None


def survivor_prereg() -> "F.Prereg":
    """The one target allowed to survive must arrive with its criterion already
    written. A claim with no pre-registered criterion is not judged harshly here
    -- it is not judged at all, which is what INCONCLUSIVE means."""
    return F.Prereg(
        claim="Trailing 5-step momentum forecasts the next 5 steps",
        mechanism="A persistent latent state leaks into realised returns, so its "
                  "trailing average carries information about the next window.",
        implications=["the edge must decay smoothly as entry is delayed",
                      "it must survive a null matched on size and volatility",
                      "it must not be reproducible by last step's return alone"],
        primary_metric="rank_ic_mean", threshold=0.0, horizon=5,
        cost_bp=5.0, n_candidates_searched=1)


def make_targets():
    ret, mask, z_size, _ = build_panel()
    size_panel = np.repeat(z_size[None, :], T, axis=0)
    g = np.random.default_rng(SEED + 1)
    covars = {"size": size_panel, "vol": np.abs(ret)}

    common = dict(ret=ret, mask=mask, covariates=covars,
                  controls=[size_panel], control_names=["size"])

    honest = trailing_mean(ret, WINDOW, end_offset=0)

    return [
        # The implications this study pre-registered are things the gauntlet
        # itself tests -- delay decay is A3, the matched null is M1, the naive
        # baseline is M4 -- so the results are recorded from those checks. That
        # is the intended workflow: declaring implications and never testing
        # them leaves the claim where it was, and P6 reads it as inconclusive
        # rather than passing.
        ("survivor", "SURVIVES", None, F.Study(
            claim="Trailing 5-step momentum forecasts the next 5 steps",
            signal=honest, horizon=5, cost_bp=5.0, n_candidates_searched=1,
            naive_baseline=ret, naive_label="last step's return",
            implication_results={
                "the edge must decay smoothly as entry is delayed": "held",
                "it must survive a null matched on size and volatility": "held",
                "it must not be reproducible by last step's return alone": "held"},
            **common)),

        ("noise", "REJECTED", "M0", F.Study(
            claim="A random cross-section forecasts returns",
            signal=g.standard_normal((T, N)), horizon=5, cost_bp=5.0, **common)),

        ("leaky", "REJECTED", "A2", F.Study(
            claim="Trailing momentum forecasts returns (window ends one step late)",
            signal=trailing_mean(ret, WINDOW, end_offset=1), horizon=5, cost_bp=5.0, **common)),

        ("size_proxy", "REJECTED", "M1", F.Study(
            claim="This characteristic forecasts returns",
            signal=size_panel + g.standard_normal((T, N)) * 0.01,
            horizon=5, cost_bp=5.0, window_based=False, **common)),

        ("costly", "REJECTED", "E1", F.Study(
            claim="The same momentum signal, rebalanced every step at 20bp",
            signal=honest, horizon=1, cost_bp=20.0, **common)),
    ]


def robustness_targets():
    """Targets for the three checks that ask questions the rest cannot.

    `dead_panel` is the one that matters most. It is the only target whose
    correct answer is INCONCLUSIVE rather than REJECTED: the signal really is
    nothing, but so is a known effect on the same panel, so nothing has been
    established either way. A referee that returns REJECTED there is claiming
    evidence of absence from an apparatus with no demonstrated power -- and that
    mistake closes lines of work that were never actually tested.
    """
    ret, mask, z_size, a = build_panel()
    T_, N_ = ret.shape
    size_panel = np.repeat(z_size[None, :], T_, axis=0)
    honest = trailing_mean(ret, WINDOW, end_offset=0)
    g = np.random.default_rng(SEED + 21)

    frozen_vol = np.abs(ret).copy()
    frozen_vol[T_ // 2:] = frozen_vol[T_ // 2 - 1]      # a cache that stopped updating

    dead = g.standard_normal((T_, N_)) * 0.02           # no structure to find at all

    # A real, noisy quantity: the signal's IC measured on a random subsample of
    # dates. Weak enough that the answer depends on which dates you drew.
    weak = honest + g.standard_normal((T_, N_)) * np.nanstd(honest) * 12.0
    fwd5 = F.forward_returns(ret, 5)

    def subsample_ic(sd: int) -> float:
        r = np.random.default_rng(sd)
        idx = r.choice(np.arange(WINDOW + 2, T_ - 6), size=40, replace=False)
        sub = np.full_like(weak, np.nan)
        sub[idx] = weak[idx]
        v = F.rank_ic(sub, fwd5, mask, min_n=30)
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float("nan")

    # The max of thirty runs sits near the 97th percentile of its own distribution
    # by construction. Twelve put it at the 85th, which is genuinely borderline --
    # a target that only just fails tests the threshold, not the check.
    best = max(subsample_ic(s) for s in range(30))

    # An honest signal, genuinely there, and far too weak for the number of
    # candidates it came out of. Its own generator: every draw above feeds
    # `seed_lucky`, which sits at a deliberately chosen percentile of its own
    # distribution, and inserting a draw into that stream would move it.
    thin = honest + np.random.default_rng(SEED + 57).standard_normal(honest.shape) \
        * float(np.nanstd(honest)) * 11.0

    return [
        ("weak_after_search", "REJECTED", "S4", F.Study(
            claim="A weak but real signal, found after searching five thousand candidates",
            signal=thin, ret=ret, mask=mask, horizon=5, cost_bp=1.0,
            covariates={"size": size_panel}, n_candidates_searched=5000)),

        ("frozen_covariate", "REJECTED", "S7", F.Study(
            claim="Momentum forecasts returns (one covariate quietly stopped updating)",
            signal=honest, ret=ret, mask=mask, horizon=5, cost_bp=5.0,
            covariates={"size": size_panel, "vol": frozen_vol})),

        ("dead_panel", "INCONCLUSIVE", None, F.Study(
            claim="A random signal forecasts returns on a structureless panel",
            signal=g.standard_normal((T_, N_)), ret=dead, mask=mask, horizon=5,
            cost_bp=5.0, covariates={"vol": np.abs(dead)})),

        ("seed_lucky", "REJECTED", "S5", F.Study(
            claim="A weak signal works (reported from the best of thirty runs)",
            signal=honest, ret=ret, mask=mask, horizon=5, cost_bp=5.0,
            covariates={"size": size_panel, "vol": np.abs(ret)},
            seed_metric=subsample_ic, reported_metric=best, n_seeds=20)),
    ]


def increment_targets():
    """Targets for the question "should this go in", which is not the question
    "is this real".

    `increment_is_noise` is the one worth looking at. Its yearly table passes
    every reading a person would give it -- positive overall, most years up,
    losing years unchanged -- and adding a candidate that knows nothing does
    about as well. The table alone would have let it through.
    """
    g = np.random.default_rng(SEED + 61)
    T_ = 252 * 9
    dates = np.array([int(f"{y}{1 + i // 21:02d}{1 + i % 21:02d}")
                      for y in range(2016, 2025) for i in range(252)][:T_])
    yr = np.array([int(str(d)[:4]) for d in dates])
    bench = g.standard_normal(T_) * 0.011
    base = bench + g.standard_normal(T_) * 0.006 + 0.00018
    sd = 0.004

    real = base + g.standard_normal(T_) * sd * 0.4 + 0.00030      # steady, everywhere
    one_year = base + np.where(yr == 2020, 0.0030, -0.00002)       # all of it is 2020

    def null_of(s):
        return base + np.random.default_rng(s).standard_normal(T_) * sd

    def inc(combined):
        return F.Increment(dates=dates, baseline=base, combined=combined, benchmark=bench,
                           label_baseline="baseline", label_combined="+candidate")

    # The noise target has to be a draw whose yearly table *passes*, because
    # that is the case it exists to exhibit: an addition that reads well by
    # every year-by-year measure and is still no better than adding nothing.
    # Roughly half of all noise draws come out positive, so taking the first one
    # that does is constructing the fixture, not selecting a result -- a draw
    # that happened to be negative would be killed by I1 and would demonstrate
    # nothing about I2.
    noise = None
    for s_ in range(200):
        cand = null_of(SEED + 900 + s_)
        if F.i1_incremental_contribution(inc(cand)).outcome == "PASS":
            noise = cand
            break
    if noise is None:
        raise RuntimeError("no noise draw produced a passing yearly table")

    return [("increment_real", "SURVIVES", None, inc(real), null_of),
            ("increment_is_noise", "REJECTED", "I2", inc(noise), null_of),
            ("increment_one_year", "REJECTED", "I1", inc(one_year), null_of)]


def declaration_targets():
    """Targets for the checks that catch what lives outside the study's code.

    Three of these fail on a declaration rather than on a number, which is the
    only mechanical form available: a backtester that matches on the bar the
    signal was formed on produces a clean result from a correct script, and a
    spread charged twice reads as conservatism. Neither is visible in the data.
    What can be enforced is that somebody wrote down the answer -- and NA here
    is not a pass, it means nobody has.
    """
    ret, mask, z_size, _ = build_panel()
    T_, N_ = ret.shape
    g = np.random.default_rng(SEED + 51)
    honest = trailing_mean(ret, WINDOW, end_offset=0)
    size_panel = np.repeat(z_size[None, :], T_, axis=0)
    common = dict(ret=ret, mask=mask, horizon=5, covariates={"size": size_panel})

    # Events chosen by how they turned out: the trigger list is the answer key.
    fwd5 = F.forward_returns(ret, 5)
    triggers = np.zeros((T_, N_), bool)
    for t in range(WINDOW + 2, T_ - 6):
        ok = mask[t] & np.isfinite(fwd5[t]) & (fwd5[t] > 0)
        idx = np.flatnonzero(ok)
        if idx.size >= 5:
            triggers[t, g.choice(idx, size=5, replace=False)] = True

    tiny_volume = np.full((T_, N_), 1e6)

    return [
        ("filtered_events", "REJECTED", "M7", F.Study(
            claim="This detector finds moves before they happen",
            signal=honest, cost_bp=5.0, triggers=triggers, **common)),

        ("same_bar_fill", "REJECTED", "P4", F.Study(
            claim="Momentum forecasts returns (filled on the bar it was formed on)",
            signal=honest, cost_bp=5.0, fill_convention="same-close", **common)),

        ("external_fact_wrong", "REJECTED", "P5", F.Study(
            claim="Momentum forecasts returns (and the panel disagrees with the record)",
            signal=honest, cost_bp=5.0,
            external_facts=[{"what": "names at the limit on the crash day",
                             "expected": 3000, "observed": 964, "tol": 0.05}], **common)),

        ("spread_twice", "REJECTED", "E4", F.Study(
            claim="Momentum survives costs (with the spread charged twice)",
            signal=honest, cost_bp=20.0, price_convention="touch",
            cost_components=["commission", "spread"], **common)),

        ("mechanism_falsified", "REJECTED", "P6", F.Study(
            claim="Momentum forecasts returns because slow information diffuses",
            signal=honest, cost_bp=5.0,
            implication_results={"it must be stronger where coverage is thinner": "failed",
                                 "it must decay as the horizon lengthens": "held"}, **common)),

        ("no_capacity", "REJECTED", "E5", F.Study(
            claim="Momentum survives costs at ten billion",
            signal=honest, cost_bp=5.0, price_convention="trade",
            cost_components=["commission", "spread"],
            dollar_volume=tiny_volume, capital=1e10, **common)),
    ]


def integrity_targets():
    """Targets for the three input-integrity checks.

    These run before anything is measured, because a series that carries moves
    which never happened, or stops three weeks short of the panel it is joined
    to, or barely reorders between observations, makes every statistic below it
    arithmetic rather than evidence. The threshold on M5 sits between what real
    prints do (0.049% on clean one-minute index data) and what the incident it
    was written from did (about 1%).
    """
    ret, mask, z_size, a = build_panel()
    T_, N_ = ret.shape
    g = np.random.default_rng(SEED + 41)
    honest = trailing_mean(ret, WINDOW, end_offset=0)
    size_panel = np.repeat(z_size[None, :], T_, axis=0)

    # One percent of steps jump and are undone on the next -- the shape a
    # stitched or synthesised series produces, and nothing a market does.
    spiked = ret.copy()
    sd = np.nanstd(ret)
    for i in g.choice(np.arange(1, T_ - 1), size=int(T_ * 0.01), replace=False):
        j = g.choice(np.arange(N_), size=max(1, N_ // 3), replace=False)
        jump = sd * g.uniform(8, 40) * g.choice([-1.0, 1.0])
        spiked[i, j] = jump
        spiked[i + 1, j] = -jump * 0.95

    ends_early = np.abs(ret).copy()
    ends_early[-40:] = np.nan                       # a cache that was never rebuilt

    # A label that is mostly a static per-name drift: it ranks the same way
    # every day, so any ratio built on its IC series is inflated.
    theta = g.standard_normal(N_) * 0.004
    sticky = theta[None, :] + g.standard_normal((T_, N_)) * 0.004

    return [
        ("bad_prints", "REJECTED", "M5", F.Study(
            claim="Momentum forecasts returns (on a series with prints that never happened)",
            signal=trailing_mean(spiked, WINDOW, end_offset=0), ret=spiked, mask=mask,
            horizon=5, cost_bp=5.0, covariates={"size": size_panel})),

        ("stale_cache", "REJECTED", "M6", F.Study(
            claim="Momentum forecasts returns (one input stops three weeks short)",
            signal=honest, ret=ret, mask=mask, horizon=5, cost_bp=5.0,
            covariates={"size": size_panel, "vol": ends_early})),

        ("sticky_label", "REJECTED", "S9", F.Study(
            claim="A static characteristic forecasts a label that never reorders",
            signal=theta[None, :] + g.standard_normal((T_, N_)) * 0.001,
            ret=sticky, mask=mask, horizon=5, cost_bp=5.0, window_based=False,
            covariates={"size": size_panel})),
    ]


def identification_targets():
    """Targets for S10 and M12 -- both closing modes the prior records opened.

    `unidentified_fit` is a two-parameter objective with one observation per
    unit, so its solution set is a ridge rather than a point. One parameter is
    pinned by the data and the other slides along the ridge, which is the usual
    shape: something is estimated, something is along for the ride, and the
    table reports both the same way. The fit here is deliberately honest --
    it converges, it reports a surface, and every number in it is reproducible.
    Only running it from different starting points shows which half is real.

    `leaky_control` is the one that was found the expensive way. A control is
    added to orthogonalise against, and it happens to be measured over a window
    that overlaps the label. Residualising against it then *raises* the residual
    IC, which reads as the signal surviving a hard test. Nothing else objects:
    the signal is honest, the panel is fine, and M3 will report a healthy
    residual. The contamination is in the control, so the boundary locator has
    to be run on the control as well.
    """
    ret, mask, z_size, a = build_panel()
    T_, N_ = ret.shape
    g = np.random.default_rng(SEED + 83)
    honest = trailing_mean(ret, WINDOW, end_offset=0)
    size_panel = np.repeat(z_size[None, :], T_, axis=0)

    # A ridge: y = (alpha + beta) * x, so only the sum is identified.
    x = g.standard_normal(400)
    y = 1.7 * x + g.standard_normal(400) * 0.05

    def fit_from_start(start):
        al, be = float(start["alpha"]), float(start["beta"])
        # Coordinate descent on a ridge-shaped objective: it converges, it is
        # deterministic, and where it lands depends on where it began.
        for _ in range(200):
            al = float(np.mean((y - be * x) * x) / np.mean(x ** 2))
            be = float(np.mean((y - al * x) * x) / np.mean(x ** 2))
        return {"alpha": al, "beta": be, "sum": al + be}

    starts = [{"alpha": v, "beta": 1.7 - v} for v in (-2.0, -0.5, 0.85, 2.0, 3.5)]

    # A control that reads one step past its own timestamp -- the same off-by-one
    # A2 finds in a signal, but sitting in the thing being controlled *for*.
    leaky_ctrl = trailing_mean(ret, WINDOW, end_offset=1)

    return [
        ("unidentified_fit", "REJECTED", "S10", F.Study(
            claim="The calibrated parameter surface describes the data",
            signal=honest, ret=ret, mask=mask, horizon=5, cost_bp=1.0,
            covariates={"size": size_panel},
            fit_from_start=fit_from_start, starts=starts)),

        ("leaky_control", "REJECTED", "M12", F.Study(
            claim="Momentum survives orthogonalisation against a related feature",
            signal=honest, ret=ret, mask=mask, horizon=5, cost_bp=1.0,
            covariates={"size": size_panel},
            controls=[leaky_ctrl], control_names=["overlapping_window"])),
    ]


def data_targets():
    """A target for M14 -- the universe did not exist over the period tested.

    Nothing is wrong with the signal, the panel or the arithmetic. The index
    whose membership this panel claims to be was published years after the
    panel starts, so the earlier stretch is the vendor applying today's
    methodology backwards -- nine years of it for one index on this machine,
    with nothing in the store to say so. Nobody could have held that universe,
    and no one else's result on it is comparable, which is the part that bites:
    a finding that looks like it contradicts the literature may be contradicting
    the universe instead.

    M13 has no target of its own and does not need one. It is advisory by
    construction -- a legitimately huge number is possible on a small universe
    over a short window -- and every deliberately-broken target here dies of
    something sharper first. It is exercised by the unit tests instead.
    """
    ret, mask, z_size, a = build_panel()
    T_, N_ = ret.shape
    honest = trailing_mean(ret, WINDOW, end_offset=0)
    size_panel = np.repeat(z_size[None, :], T_, axis=0)
    # A panel that starts in 2014 and calls itself CSI 2000, published 2023-08.
    days = np.array([f"2014-08-{1 + (i % 28):02d}" if i < 28 else
                     f"{2014 + i // 250}-{1 + (i // 21) % 12:02d}-{1 + i % 28:02d}"
                     for i in range(T_)])

    return [
        ("universe_predates_index", "REJECTED", "M14", F.Study(
            claim="A trailing-window signal forecasts returns across the CSI 2000",
            signal=honest, ret=ret, mask=mask, dates=days, horizon=5, cost_bp=1.0,
            universe="csi_2000", covariates={"size": size_panel})),
    ]


def seal_targets():
    """A target for P9 -- the panel included rows it was not allowed to see.

    Every peek in this record arrived this way, and none of them was an unseal:
    a yearly alignment diagnostic printed without excluding the test rows, and a
    panel handed to this referee that quietly contained them. In the second case
    every check ran and every number was computed correctly; all of them were
    about a panel that should not have existed.

    So the study underneath is the honest survivor again. Nothing is wrong with
    the signal. The only thing wrong is which rows it was allowed to be built
    from, and that is invisible to every other check in the battery -- P2 reports
    whether the seal is intact and never looks at the data.
    """
    ret, mask, z_size, a = build_panel()
    T_, N_ = ret.shape
    honest = trailing_mean(ret, WINDOW, end_offset=0)
    size_panel = np.repeat(z_size[None, :], T_, axis=0)
    dates = np.arange(T_)
    split = F.SealedSplit(train_end=int(T_ * 0.6), valid_end=int(T_ * 0.8),
                          ledger=str(Path(tempfile.mkdtemp()) / "unseal_ledger.json"))

    return [
        ("panel_saw_the_test", "REJECTED", "P9", F.Study(
            claim="Momentum forecasts returns (on a panel that runs through the sealed period)",
            signal=honest, ret=ret, mask=mask, dates=dates, horizon=5, cost_bp=1.0,
            covariates={"size": size_panel})),
    ], split


def constraint_targets():
    """A target for E6 -- the book bought what was locked.

    The signal is real. It fires the day a name goes limit-up and the name does
    keep going the next step, so the IC is genuine, the harness passes, the
    matched null passes, and the turnover is cheap enough that the cost gate has
    nothing to say. Every number in the battery is correct.

    It is also unavailable. The names it picks are sitting on a locked board on
    the day it picks them, and a locked board cannot be bought. Removing exactly
    those entries -- not charging more for them, removing them -- takes the whole
    edge, because the edge *was* those entries.

    This is the shape that the cost axis structurally cannot see: it is not a
    price, it is an absence, and a backtest that fills the order simply never
    finds out.
    """
    ret, mask, z_size, a = build_panel()
    T_, N_ = ret.shape
    g = np.random.default_rng(SEED + 113)
    size_panel = np.repeat(z_size[None, :], T_, axis=0)

    # Locked boards with a one-step continuation, injected on the honest panel so
    # the positive control still has something to find. Both directions, and
    # that symmetry is the point: with only limit-ups, the signal keeps a real
    # residual edge after the constraint -- "do not short what just jumped" --
    # because nothing stops you shorting a board that is locked *up*. Injecting
    # only one side produces a target that looks like it has surviving alpha and
    # does not, which is the mistake this whole check is about.
    locked_ret = ret.copy()
    sig = g.standard_normal((T_, N_)) * 0.01           # a weak, honest ranking
    for t in np.arange(WINDOW + 2, T_ - 2):
        pick = g.choice(N_, size=max(4, N_ // 10), replace=False)
        up, down = pick[: pick.size // 2], pick[pick.size // 2:]
        locked_ret[t, up] = 0.100                       # locked up: cannot be bought
        locked_ret[t + 1, up] += 0.020                  # and keeps going
        sig[t, up] += 3.0
        locked_ret[t, down] = -0.100                    # locked down: cannot be shorted
        locked_ret[t + 1, down] -= 0.020
        sig[t, down] -= 3.0

    return [
        ("bought_the_locked_board", "REJECTED", "E6", F.Study(
            claim="A signal that fires on locked boards earns the continuation",
            signal=sig, ret=locked_ret, mask=mask, horizon=1, cost_bp=1.0,
            window_based=False, covariates={"size": size_panel})),
    ]


def hindsight_targets():
    """A target for P8 -- the idea came from something that already knew.

    The study underneath is the honest survivor: a real trailing-window signal,
    causally built, that passes every audit in the battery. That is the point.
    The contamination is not in the pipeline and no downstream check can reach
    it, because the pipeline was pointed in the right direction by a source that
    had already seen the outcome.

    The probes follow Jump's example: a question whose answer became knowable
    only after the declared boundary, asked of a source that answers it anyway.
    Control probes from before the boundary are supplied as well, because a
    source that answers nothing looks exactly like a source with an honest
    boundary -- that is S6's argument, and it applies in this direction too.
    """
    ret, mask, z_size, a = build_panel()
    T_, N_ = ret.shape
    honest = trailing_mean(ret, WINDOW, end_offset=0)
    size_panel = np.repeat(z_size[None, :], T_, axis=0)

    # Answers that only exist on the far side of the boundary, and answers that
    # any source reasoning from before it should still have.
    after = {"the first-day close of the listing on 2026-06-12": 160.95,
             "the index level at the close of 2026-06-12": 21550.0}
    before = {"the number of US market-wide halts in March 2020": 4,
              "the ticker of the CSI 300 index": "000300"}

    def leaks(question):
        for k, v in list(after.items()) + list(before.items()):
            if k in question:
                return f"about {v}"
        return "I cannot know that"

    return [
        ("oracle_proposed_it", "REJECTED", "P8", F.Study(
            claim="A trailing-window signal forecasts returns (proposed by a source "
                  "reasoning, it says, from before 2026-06-05)",
            signal=honest, ret=ret, mask=mask, horizon=5, cost_bp=1.0,
            covariates={"size": size_panel},
            ask=leaks, information_boundary="2026-06-05",
            hindsight_probes=[{"question": q, "answer": v, "tol": 0.01}
                              for q, v in after.items()],
            hindsight_controls=[{"question": q, "answer": v} for q, v in before.items()])),
    ]


def guard_targets():
    """A target for P7 -- the guard that could never have failed.

    Taken from the incident rather than invented. A release gate compares the
    previous published panel against the new one and reports any value that
    moved, and it does it by intersecting the two sides' non-missing masks:
    `old.notna() & new.notna()`. That intersection excludes exactly the rows
    where a value appeared or vanished between releases, which is the class of
    change the gate exists to find. It had been green on every release since it
    was written.

    What makes it worth a target is that nothing else can see it. The gate's
    output on real data is indistinguishable from the output of a correct gate
    that has nothing to report, and no amount of running the pipeline separates
    them. Only injecting the defect does -- which is S6's argument about panels,
    pointed at a check instead.

    The injected pair is deliberately asymmetric: the gate does catch a value
    that *changed*, and only misses one that *vanished*. A target that missed
    everything would pass a P7 that merely counted, rather than one that reads
    which defects got through.
    """
    ret, mask, z_size, a = build_panel()
    T_, N_ = ret.shape
    honest = trailing_mean(ret, WINDOW, end_offset=0)
    size_panel = np.repeat(z_size[None, :], T_, axis=0)

    g = np.random.default_rng(SEED + 61)
    published = g.standard_normal(600)
    release = {"old": published, "new": published.copy()}

    def gate(d):
        """The real one: compare where both sides have a value."""
        both = np.isfinite(d["old"]) & np.isfinite(d["new"])
        return bool(np.allclose(d["old"][both], d["new"][both]))

    def a_value_vanished(d):
        new = d["new"].copy(); new[17] = np.nan
        return {"old": d["old"], "new": new}

    def a_value_changed(d):
        new = d["new"].copy(); new[23] += 1.0
        return {"old": d["old"], "new": new}

    return [
        ("guard_cannot_fire", "REJECTED", "P7", F.Study(
            claim="The release gate guarantees no value changed silently between releases",
            signal=honest, ret=ret, mask=mask, horizon=5, cost_bp=1.0,
            covariates={"size": size_panel},
            validator=gate, validator_data=release,
            corruptions={"a value vanished": a_value_vanished,
                         "a value changed": a_value_changed})),
    ]


def stationarity_targets():
    """Targets for M10 and M11 -- the two modes the taxonomy carried as holes.

    Both are built the way they actually happen rather than by injecting a bug.

    `frozen_stale` is a representation that is genuinely right about the first
    half of the sample and genuinely wrong about the second, and is shipped
    frozen. Nothing upstream can see it: the pooled IC is comfortably positive,
    the leak audits are clean, the matched nulls pass. It is only when the
    sample is split in time that the claim turns out to be about a window
    rather than about the market.

    `frequency_flip` is a one-step continuation followed by a slower reversal --
    the shape that makes a signal pay at its native cadence and lose money held
    four times longer. The construction is unchanged at every frequency, which
    is exactly why the conclusion gets carried across them. It is declared to
    hold at 4x, and that declaration is what makes M11 blocking.
    """
    g = np.random.default_rng(SEED + 71)

    # ---- a relationship that moves under a frozen fit ----------------------
    a = np.zeros((T, N))
    eps = g.standard_normal((T, N)) * np.sqrt(1 - RHO ** 2)
    for t in range(1, T):
        a[t] = RHO * a[t - 1] + eps[t]
    z_size = g.standard_normal(N)
    f_size = g.normal(MU_SIZE, SD_SIZE, size=T)
    idio = g.standard_normal((T, N)) * SD_IDIO
    # The channel the latent state reaches returns through weakens and turns
    # over: positive for the first half of the sample, negative for the second.
    kappa = np.where(np.arange(T) < T // 2, KAPPA, -0.4 * KAPPA)
    drift = np.empty((T, N))
    drift[0] = idio[0]
    drift[1:] = z_size[None, :] * f_size[1:, None] + kappa[1:, None] * a[:-1] + idio[1:]
    drift_mask = np.ones((T, N), bool)
    drift_mask[:WINDOW + 2] = False
    size_panel = np.repeat(z_size[None, :], T, axis=0)

    # ---- one-step continuation, multi-step reversal ------------------------
    sig = g.standard_normal((T, N))
    flip = g.standard_normal((T, N)) * SD_IDIO
    c = 0.0060
    for lag, w in ((1, 1.0), (2, -0.55), (3, -0.55), (4, -0.35)):
        flip[lag:] += w * c * sig[:-lag]
    flip_mask = np.ones((T, N), bool)
    flip_mask[:WINDOW + 2] = False
    flip_size = np.repeat(g.standard_normal(N)[None, :], T, axis=0)

    return [
        ("frozen_stale", "REJECTED", "M10", F.Study(
            claim="A frozen representation of the latent state forecasts returns",
            signal=a, ret=drift, mask=drift_mask, horizon=1, cost_bp=5.0,
            window_based=False, retrained=False,
            covariates={"size": size_panel})),

        ("frequency_flip", "REJECTED", "M11", F.Study(
            claim="The signal forecasts returns, at this frequency and four times slower",
            signal=sig, ret=flip, mask=flip_mask, horizon=1, cost_bp=5.0,
            window_based=False, claimed_strides=(4,),
            covariates={"size": flip_size})),
    ]


def artefact_targets():
    """Targets for E3, S8 and P3.

    `stale_index` is built the way the real thing happens rather than by
    injecting a bug: the tradable series has nothing forecastable in it at all,
    and the index simply prints half of yesterday because half its names have
    not traded yet. A signal formed on the last tradable return then predicts
    the next index print with a perfectly real, perfectly untradable IC. Every
    other check passes it -- there is no leak, no null it fails, no cost it
    cannot carry -- which is exactly why the artefact is worth its own check.
    """
    ret, mask, _, a = build_panel()
    T_, N_ = ret.shape
    g = np.random.default_rng(SEED + 31)

    # A tradable series with no forecastable structure, and an index that lags it.
    tradable = g.standard_normal((T_, N_)) * 0.018
    index = 0.5 * tradable + 0.5 * np.vstack([np.zeros((1, N_)), tradable[:-1]])
    # The return realised over period t is observed at t's close, so using it to
    # forecast t+1 is ordinary momentum, not look-ahead. The index print at t+1
    # happens to contain half of it, which is where the fake edge comes from.
    last_seen = tradable.copy()

    honest = trailing_mean(ret, WINDOW, end_offset=0)
    fwd5 = F.forward_returns(ret, 5)

    # A knob that fits training-period noise: at k=0 the signal is honest, and
    # every increment mixes in a component that matches the label on train dates
    # and is noise everywhere else.
    cut = T_ // 2
    fitted = g.standard_normal((T_, N_))
    fitted[:cut] = np.nan_to_num(fwd5[:cut])
    seg_mask = {"train": mask & (np.arange(T_) < cut)[:, None],
                "valid": mask & (np.arange(T_) >= cut)[:, None]}

    def knob(k: float, seg: str) -> float:
        sd = np.nanstd(honest)
        sig = honest + k * sd * (fitted / (np.nanstd(fitted) or 1.0))
        v = F.rank_ic(sig, fwd5, seg_mask[seg], min_n=30)
        v = v[np.isfinite(v)]
        return float(v.mean()) if v.size else float("nan")

    # Bid-ask bounce: the observed price alternates between the two sides of a
    # spread, so consecutive observed returns are negatively autocorrelated for
    # reasons that have nothing to do with forecasting. A reversal signal picks
    # that up mechanically. Unlike the stale print above, the signal here does
    # *not* overlap its own label -- A2 passes it -- which is why E3 exists.
    f = g.standard_normal((T_, N_)) * 0.012
    side = g.choice([-1.0, 1.0], size=(T_, N_))
    spread = 0.004
    obs = f + spread * (side - np.vstack([side[:1], side[:-1]]))
    bounce_sig = -obs

    frozen_dir = tempfile.mkdtemp(prefix="falsifier_frozen_")
    frozen_path = os.path.join(frozen_dir, "config.json")
    with open(frozen_path, "w", encoding="utf-8") as fh:
        json.dump({"q_low": 0.3300, "q_high": 0.6700, "window": 5}, fh)

    return [
        ("stale_index", "REJECTED", "A2", F.Study(
            claim="The last tradable return forecasts the next index print",
            signal=last_seen, ret=index, mask=mask, horizon=1, cost_bp=0.0,
            tradable_ret=tradable, price_source="index",
            covariates={"vol": np.abs(index)})),

        ("bounce", "REJECTED", "E3", F.Study(
            claim="Reversal on observed prices forecasts the next observed return",
            signal=bounce_sig, ret=obs, mask=mask, horizon=1, cost_bp=0.0,
            tradable_ret=f, price_source="synthetic",
            covariates={"vol": np.abs(obs)})),

        ("overfit_knob", "REJECTED", "S8", F.Study(
            claim="Turning the knob up improves the signal",
            signal=honest, ret=ret, mask=mask, horizon=5, cost_bp=5.0,
            covariates={"vol": np.abs(ret)},
            knob_metric=knob, knob_params=(0.0, 0.5, 1.0, 2.0, 4.0, 8.0))),

        ("config_drifted", "REJECTED", "P3", F.Study(
            claim="The production config still reproduces (it does not)",
            signal=honest, ret=ret, mask=mask, horizon=5, cost_bp=5.0,
            covariates={"vol": np.abs(ret)},
            frozen_config={"q_low": 0.3412, "q_high": 0.6700, "window": 5},
            frozen_config_path=frozen_path)),
    ]


def strategy_targets():
    """Portfolio-level targets. The strategy module needs the same standard as
    the rest: a book built on real foresight must clear its turnover-matched
    null, and a book that churns identically on no information must not.

    One caveat that belongs in the open. ``blind_book`` is a single draw from
    the very distribution SM1 compares against, so by construction it clears
    the 95th percentile about one time in twenty; a seed exists for which this
    target 'fails'. That is the check behaving correctly, not a defect, and it
    is why the false-positive rate is measured separately in the test suite
    rather than inferred from one book getting rejected here."""
    ret, mask, z_size, a = build_panel()
    T_, N_ = ret.shape
    invest = mask.copy()
    g = np.random.default_rng(SEED + 7)

    def book_from(score, n_hold=25, every=20):
        sel = np.zeros((T_, N_), bool)
        cur = None
        for t in range(T_):
            if t % every == 0 and invest[t].sum() > n_hold:
                idx = np.flatnonzero(invest[t] & np.isfinite(score[t]))
                if idx.size > n_hold:
                    cur = idx[np.argsort(score[t, idx])[-n_hold:]]
            if cur is not None:
                sel[t, cur] = True
        return sel

    skilled = book_from(a)                                   # holds the latent state
    blind = book_from(g.standard_normal((T_, N_)))           # same cadence, no information
    return [
        ("skilled_book", "SURVIVES", None,
         F.StrategyStudy(claim="A book that can see the latent state beats a matched random book",
                         selection=skilled, ret=ret, mask=invest, cost_bp=2.0,
                         n_candidates_searched=1)),
        ("blind_book", "REJECTED", "SM1",
         F.StrategyStudy(claim="A book that picks at random beats a matched random book",
                         selection=blind, ret=ret, mask=invest, cost_bp=2.0,
                         n_candidates_searched=1)),
    ]


def pipeline_targets():
    """Targets for the two decisive audits.

    A0 and A1 need a pipeline, not an array, so these wrap `RollingFit` with one
    deliberate defect each. They also show the division of labour between the
    two: A0 asks whether a value could have been produced on its own date, so it
    catches anything computed from data that did not exist yet -- a full-sample
    scaling constant, or coefficients fitted over the whole history. A1 asks
    whether the fit can score on labels that carry no information, so it catches
    a training window that reaches into the period being predicted. Neither
    subsumes the other, and a clean pipeline has to pass both.
    """
    ret, mask, _, _ = build_panel()
    T_, N_ = ret.shape
    r0 = np.nan_to_num(ret, nan=0.0)
    windows = (1, 2, 3, 5, 10, 20)
    feat = np.full((T_, N_, len(windows)), np.nan)
    for k, w in enumerate(windows):
        c = np.cumsum(np.vstack([np.zeros((1, N_)), r0]), 0)
        feat[w - 1:, :, k] = (c[w:] - c[:-w]) / w
    m = mask & np.isfinite(feat).all(axis=2)

    def study_for(name, standardize, window, expect, killer):
        rf = F.RollingFit(feat, ret, m, horizon=5, fitwin=250, stride=10, min_n=50,
                          standardize=standardize, window=window)
        return (name, expect, killer, F.Study(
            claim=f"A rolling fit on trailing-return features [{name}]",
            signal=rf.predict(), ret=ret, mask=m, horizon=5, min_n=50,
            recompute_at=rf.recompute_at, refit=rf.refit,
            probe_dates=rf.probe_dates(4), n_shuffle=6,
            covariates={"vol": np.abs(ret)}, cost_bp=2.0, n_candidates_searched=1))

    return [
        study_for("clean", "cross-section", "causal", "SURVIVES", None),
        study_for("full-sample scaling", "full-sample", "causal", "REJECTED", "A0"),
        study_for("training window reaches the predicted date",
                  "cross-section", "contaminated", "REJECTED", "A1"),
        study_for("one fit over the whole history", "cross-section", "full-sample",
                  "REJECTED", "A0"),
    ]


def all_target_specs():
    return (make_targets() + declaration_targets() + integrity_targets()
            + stationarity_targets() + identification_targets() + guard_targets()
            + hindsight_targets() + constraint_targets() + seal_targets()[0]
            + data_targets() + artefact_targets()
            + robustness_targets() + pipeline_targets() + strategy_targets())


def declared_killers() -> set:
    """Which checks some target is built to be killed by.

    Used by the test suite: a check claimed to catch a failure mode, with no
    target that trips it, has never been shown to work.
    """
    return ({k for _, _, k, _ in all_target_specs() if k}
            | {k for _, _, k, _, _ in increment_targets() if k})


def main(n_draws: int = 120, include_pipeline: bool = True) -> int:
    rows, failures, reports = [], [], []
    for name, want_outcome, want_killer, study in make_targets():
        print(f"\n{'#' * 96}\n### target: {name}   (expected {want_outcome}"
              + (f" by {want_killer}" if want_killer else "") + ")\n" + "#" * 96)
        pre = survivor_prereg() if name == "survivor" else None
        rep = F.run(study, prereg=pre, n_draws=n_draws, seed=SEED, verbose=True)
        print(rep.render())
        reports.append(rep)
        killers = [c.id for c in rep.killers]
        got_outcome = rep.outcome
        ok_outcome = got_outcome == want_outcome
        ok_killer = want_killer is None or (killers and killers[0] == want_killer)
        rows.append((name, want_outcome, got_outcome, want_killer or "-", ",".join(killers) or "-",
                     ok_outcome and ok_killer))
        if not (ok_outcome and ok_killer):
            failures.append(name)

    for name, want_outcome, want_killer, inc, null_of in increment_targets():
        print(f"\n{'#' * 96}\n### target: {name}   (expected {want_outcome}"
              + (f" by {want_killer}" if want_killer else "") + ")\n" + "#" * 96)
        rep = F.run_increment(inc, combined_of_seed=null_of, n_draws=max(60, n_draws),
                              seed=SEED, verbose=False)
        print(F.render_yearly(inc))
        print(rep.render())
        reports.append(rep)
        killers = [c.id for c in rep.killers]
        ok_o = rep.outcome == want_outcome
        ok_k = want_killer is None or (killers and killers[0] == want_killer)
        rows.append((name, want_outcome, rep.outcome, want_killer or "-",
                     ",".join(killers) or "-", ok_o and ok_k))
        if not (ok_o and ok_k):
            failures.append(name)

    for name, want_outcome, want_killer, study in declaration_targets():
        print(f"\n{'#' * 96}\n### target: {name}   (expected {want_outcome}"
              + (f" by {want_killer}" if want_killer else "") + ")\n" + "#" * 96)
        # A mechanism can only be falsified if it was written down, so this
        # target needs the pre-registration that declared it.
        pre = (F.Prereg(claim=study.claim,
                        mechanism="slow information diffusion",
                        implications=list(study.implication_results),
                        primary_metric="rank_ic_mean", horizon=5)
               if study.implication_results else None)
        rep = F.run(study, prereg=pre, n_draws=max(40, n_draws // 2), seed=SEED, verbose=True)
        print(rep.render())
        reports.append(rep)
        killers = [c.id for c in rep.killers]
        ok_o = rep.outcome == want_outcome
        ok_k = want_killer is None or (killers and killers[0] == want_killer)
        rows.append((name, want_outcome, rep.outcome, want_killer or "-",
                     ",".join(killers) or "-", ok_o and ok_k))
        if not (ok_o and ok_k):
            failures.append(name)

    for name, want_outcome, want_killer, study in integrity_targets():
        print(f"\n{'#' * 96}\n### target: {name}   (expected {want_outcome}"
              + (f" by {want_killer}" if want_killer else "") + ")\n" + "#" * 96)
        rep = F.run(study, n_draws=max(40, n_draws // 2), seed=SEED, verbose=True)
        print(rep.render())
        reports.append(rep)
        killers = [c.id for c in rep.killers]
        ok_o = rep.outcome == want_outcome
        ok_k = want_killer is None or (killers and killers[0] == want_killer)
        rows.append((name, want_outcome, rep.outcome, want_killer or "-",
                     ",".join(killers) or "-", ok_o and ok_k))
        if not (ok_o and ok_k):
            failures.append(name)

    for name, want_outcome, want_killer, study in data_targets():
        print(f"\n{'#' * 96}\n### target: {name}   (expected {want_outcome}"
              + (f" by {want_killer}" if want_killer else "") + ")\n" + "#" * 96)
        rep = F.run(study, n_draws=max(40, n_draws // 2), seed=SEED, verbose=True)
        print(rep.render())
        reports.append(rep)
        killers = [c.id for c in rep.killers]
        ok_o = rep.outcome == want_outcome
        ok_k = want_killer is None or (killers and killers[0] == want_killer)
        rows.append((name, want_outcome, rep.outcome, want_killer or "-",
                     ",".join(killers) or "-", ok_o and ok_k))
        if not (ok_o and ok_k):
            failures.append(name)

    _seal_specs, _seal_split = seal_targets()
    for name, want_outcome, want_killer, study in _seal_specs:
        print(f"\n{'#' * 96}\n### target: {name}   (expected {want_outcome}"
              + (f" by {want_killer}" if want_killer else "") + ")\n" + "#" * 96)
        rep = F.run(study, seal=_seal_split, n_draws=max(40, n_draws // 2), seed=SEED, verbose=True)
        print(rep.render())
        reports.append(rep)
        killers = [c.id for c in rep.killers]
        ok_o = rep.outcome == want_outcome
        ok_k = want_killer is None or (killers and killers[0] == want_killer)
        rows.append((name, want_outcome, rep.outcome, want_killer or "-",
                     ",".join(killers) or "-", ok_o and ok_k))
        if not (ok_o and ok_k):
            failures.append(name)

    for name, want_outcome, want_killer, study in constraint_targets():
        print(f"\n{'#' * 96}\n### target: {name}   (expected {want_outcome}"
              + (f" by {want_killer}" if want_killer else "") + ")\n" + "#" * 96)
        rep = F.run(study, n_draws=max(40, n_draws // 2), seed=SEED, verbose=True)
        print(rep.render())
        reports.append(rep)
        killers = [c.id for c in rep.killers]
        ok_o = rep.outcome == want_outcome
        ok_k = want_killer is None or (killers and killers[0] == want_killer)
        rows.append((name, want_outcome, rep.outcome, want_killer or "-",
                     ",".join(killers) or "-", ok_o and ok_k))
        if not (ok_o and ok_k):
            failures.append(name)

    for name, want_outcome, want_killer, study in hindsight_targets():
        print(f"\n{'#' * 96}\n### target: {name}   (expected {want_outcome}"
              + (f" by {want_killer}" if want_killer else "") + ")\n" + "#" * 96)
        rep = F.run(study, n_draws=max(40, n_draws // 2), seed=SEED, verbose=True)
        print(rep.render())
        reports.append(rep)
        killers = [c.id for c in rep.killers]
        ok_o = rep.outcome == want_outcome
        ok_k = want_killer is None or (killers and killers[0] == want_killer)
        rows.append((name, want_outcome, rep.outcome, want_killer or "-",
                     ",".join(killers) or "-", ok_o and ok_k))
        if not (ok_o and ok_k):
            failures.append(name)

    for name, want_outcome, want_killer, study in guard_targets():
        print(f"\n{'#' * 96}\n### target: {name}   (expected {want_outcome}"
              + (f" by {want_killer}" if want_killer else "") + ")\n" + "#" * 96)
        rep = F.run(study, n_draws=max(40, n_draws // 2), seed=SEED, verbose=True)
        print(rep.render())
        reports.append(rep)
        killers = [c.id for c in rep.killers]
        ok_o = rep.outcome == want_outcome
        ok_k = want_killer is None or (killers and killers[0] == want_killer)
        rows.append((name, want_outcome, rep.outcome, want_killer or "-",
                     ",".join(killers) or "-", ok_o and ok_k))
        if not (ok_o and ok_k):
            failures.append(name)

    for name, want_outcome, want_killer, study in identification_targets():
        print(f"\n{'#' * 96}\n### target: {name}   (expected {want_outcome}"
              + (f" by {want_killer}" if want_killer else "") + ")\n" + "#" * 96)
        rep = F.run(study, n_draws=max(40, n_draws // 2), seed=SEED, verbose=True)
        print(rep.render())
        reports.append(rep)
        killers = [c.id for c in rep.killers]
        ok_o = rep.outcome == want_outcome
        ok_k = want_killer is None or (killers and killers[0] == want_killer)
        rows.append((name, want_outcome, rep.outcome, want_killer or "-",
                     ",".join(killers) or "-", ok_o and ok_k))
        if not (ok_o and ok_k):
            failures.append(name)

    for name, want_outcome, want_killer, study in stationarity_targets():
        print(f"\n{'#' * 96}\n### target: {name}   (expected {want_outcome}"
              + (f" by {want_killer}" if want_killer else "") + ")\n" + "#" * 96)
        rep = F.run(study, n_draws=max(40, n_draws // 2), seed=SEED, verbose=True)
        print(rep.render())
        reports.append(rep)
        killers = [c.id for c in rep.killers]
        ok_o = rep.outcome == want_outcome
        ok_k = want_killer is None or (killers and killers[0] == want_killer)
        rows.append((name, want_outcome, rep.outcome, want_killer or "-",
                     ",".join(killers) or "-", ok_o and ok_k))
        if not (ok_o and ok_k):
            failures.append(name)

    for name, want_outcome, want_killer, study in artefact_targets():
        print(f"\n{'#' * 96}\n### target: {name}   (expected {want_outcome}"
              + (f" by {want_killer}" if want_killer else "") + ")\n" + "#" * 96)
        rep = F.run(study, n_draws=max(40, n_draws // 2), seed=SEED, verbose=True)
        print(rep.render())
        reports.append(rep)
        killers = [c.id for c in rep.killers]
        ok_o = rep.outcome == want_outcome
        ok_k = want_killer is None or (killers and killers[0] == want_killer)
        rows.append((name, want_outcome, rep.outcome, want_killer or "-",
                     ",".join(killers) or "-", ok_o and ok_k))
        if not (ok_o and ok_k):
            failures.append(name)

    for name, want_outcome, want_killer, study in robustness_targets():
        print(f"\n{'#' * 96}\n### target: {name}   (expected {want_outcome}"
              + (f" by {want_killer}" if want_killer else "") + ")\n" + "#" * 96)
        rep = F.run(study, n_draws=max(40, n_draws // 2), seed=SEED, verbose=True)
        print(rep.render())
        reports.append(rep)
        killers = [c.id for c in rep.killers]
        ok_o = rep.outcome == want_outcome
        ok_k = want_killer is None or (killers and killers[0] == want_killer)
        rows.append((name, want_outcome, rep.outcome, want_killer or "-",
                     ",".join(killers) or "-", ok_o and ok_k))
        if not (ok_o and ok_k):
            failures.append(name)

    for name, want_outcome, want_killer, study in (pipeline_targets() if include_pipeline else []):
        print(f"\n{'#' * 96}\n### target: {name}   (expected {want_outcome}"
              + (f" by {want_killer}" if want_killer else "") + ")\n" + "#" * 96)
        pre = F.Prereg(claim=study.claim, mechanism="synthetic: a persistent latent state "
                       "leaks into returns and a trailing window recovers it",
                       primary_metric="rank_ic_mean", horizon=5) if name == "clean" else None
        rep = F.run(study, prereg=pre, n_draws=max(40, n_draws // 2), seed=SEED, verbose=True)
        print(rep.render())
        reports.append(rep)
        killers = [c.id for c in rep.killers]
        ok_o = rep.outcome == want_outcome
        ok_k = want_killer is None or (killers and killers[0] == want_killer)
        rows.append((name, want_outcome, rep.outcome, want_killer or "-",
                     ",".join(killers) or "-", ok_o and ok_k))
        if not (ok_o and ok_k):
            failures.append(name)

    for name, want_outcome, want_killer, study in strategy_targets():
        print(f"\n{'#' * 96}\n### target: {name}   (expected {want_outcome}"
              + (f" by {want_killer}" if want_killer else "") + ")\n" + "#" * 96)
        pre = F.Prereg(claim=study.claim,
                       mechanism="synthetic: the latent state is observable by construction",
                       primary_metric="sharpe") if name == "skilled_book" else None
        rep = F.run_strategy(study, prereg=pre, n_draws=n_draws * 2, seed=SEED, verbose=True)
        print(rep.render())
        reports.append(rep)
        killers = [c.id for c in rep.killers]
        ok_o = rep.outcome == want_outcome
        ok_k = want_killer is None or (killers and killers[0] == want_killer)
        rows.append((name, want_outcome, rep.outcome, want_killer or "-",
                     ",".join(killers) or "-", ok_o and ok_k))
        if not (ok_o and ok_k):
            failures.append(name)

    # Which veto-carrying checks actually fired somewhere, measured rather than
    # declared. A check nobody has seen reject anything has not been shown to
    # work, whatever its coverage entry says.
    # A blocking check has been watched doing its job when it has been seen to
    # stop something -- FAIL, or INCONCLUSIVE while carrying a veto, which means
    # the claim was not judged and is not survival either. Counting only FAIL
    # listed P0 on every run: by construction it reports PASS or INCONCLUSIVE
    # and never FAILs, so the line was reporting a fact about the check's
    # vocabulary rather than about the coverage, which is how a warning that is
    # always on stops being read.
    tripped, blocking_seen = set(), set()
    for r in reports:
        for c in r.checks:
            if c.blocking:
                blocking_seen.add(c.id)
                if c.outcome in ("FAIL", "INCONCLUSIVE"):
                    tripped.add(c.id)
    never = sorted(blocking_seen - tripped)

    print("\n" + "=" * 96)
    print(f"{'target':<42}{'expected':<14}{'got':<14}{'want':<7}{'got killer':<14}")
    print("-" * 96)
    for name, we, ge, wk, gk, ok in rows:
        print(f"{name:<42}{we:<14}{ge:<14}{wk:<7}{gk:<14}{'ok' if ok else 'MISMATCH'}")
    print("=" * 96)
    if never:
        print(f"\nveto-carrying checks never seen to reject anything: {', '.join(never)}")
        print("Each of those is a check nobody has watched do its job.")
    if failures:
        print(f"SELF-CHECK FAILED on: {', '.join(failures)}")
        print("The referee is not calibrated. Fix it before trusting it on a real claim.")
        return 1
    print("SELF-CHECK PASSED: every target died of the cause it was built to die of.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 120,
                          include_pipeline="--fast" not in sys.argv))
