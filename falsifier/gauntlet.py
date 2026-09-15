"""The gauntlet: run every axis and return one verdict.

Order matters. Process first, because an unregistered claim cannot be judged
at all. Then point-in-time, because a leak makes every downstream number
meaningless. Then the nulls, then the cost. Stopping early is the point --
there is no reason to price a signal that turned out to be tomorrow's return.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Sequence

import numpy as np

from . import econ, mech, pit, robust
from .prereg import (Prereg, p3_frozen_config, p4_fill_convention,
                     p5_external_facts, p6_mechanism_implications,
                     p7_validator_control, p8_hindsight_control)
from .seal import SealedSplit
from .stats import deflated_threshold, forward_returns, ic_summary, rank_ic
from .verdict import FAIL, INCONCLUSIVE, NA, PASS, Check, Report


@dataclass
class Study:
    """Everything the referee needs, and nothing it should have to guess."""

    claim: str
    signal: np.ndarray                      # (T, N), the value known at t
    ret: np.ndarray                         # (T, N), simple one-period returns
    mask: np.ndarray                        # (T, N), tradable at t
    dates: Optional[np.ndarray] = None
    horizon: int = 1

    # --- what makes the audits possible -------------------------------------
    window_based: bool = True
    recompute_at: Optional[Callable[[int], np.ndarray]] = None
    refit: Optional[Callable[[np.ndarray, int], np.ndarray]] = None
    probe_dates: Optional[Sequence[int]] = None
    n_shuffle: int = 20
    """Both decisive audits are expensive -- A0 rebuilds history per probe date,
    A1 refits the whole pipeline per draw. Lower these to make them affordable;
    a handful of probes still settles A0, which is deterministic."""

    # --- what the nulls must be matched on ----------------------------------
    covariates: Dict[str, np.ndarray] = field(default_factory=dict)
    controls: List[np.ndarray] = field(default_factory=list)
    control_names: List[str] = field(default_factory=list)
    naive_baseline: Optional[np.ndarray] = None
    naive_label: str = "naive baseline"
    groups: Optional[np.ndarray] = None

    # --- economics ----------------------------------------------------------
    cost_bp: float = 0.0
    price_convention: Optional[str] = None
    """'mid', 'touch', 'trade' or 'vwap' -- which price the returns are measured at."""
    cost_components: Optional[Sequence[str]] = None
    """What `cost_bp` contains: 'commission', 'spread', 'impact', 'tax', 'borrow'."""
    dollar_volume: Optional[np.ndarray] = None
    capital: Optional[float] = None
    max_participation: float = 0.1
    tradable_ret: Optional[np.ndarray] = None
    """Returns of the instrument a book would actually hold -- an ETF, a future,
    the contract itself. Supplying it turns E3 from a hint into a verdict."""
    price_source: str = "tradable"
    """'tradable', 'index' or 'synthetic'. Declaring 'index' switches on E3."""
    min_dollar_volume: Optional[float] = None
    """A liquidity floor the book would actually enforce. E6 prices it."""
    price_limit: Optional[float] = 0.0995
    """Daily price limit as a simple return. Names locked at it cannot be entered.
    Set to None for a market without limits -- and say so, because E6 then has
    nothing to price and reports that the book has not been shown to be enterable."""

    quantile: float = 0.1
    long_short: bool = True

    # --- honesty about the search -------------------------------------------
    n_candidates_searched: int = 1

    # --- robustness ---------------------------------------------------------
    seed_metric: Optional[Callable[[int], float]] = None
    """metric_of_seed(seed) -> float. Supply it and S5 reruns the study under
    different randomness; without it a one-run result is taken on faith."""
    reported_metric: Optional[float] = None
    """The number actually being claimed, checked against its own seed spread."""
    n_seeds: int = 20
    knob_metric: Optional[Callable[[float, str], float]] = None
    """metric_of(param, segment) -> float, over a parameter sweep. S8 needs the
    whole sweep, not the setting that was chosen."""
    knob_params: Sequence[float] = ()
    fit_from_start: Optional[Callable[[Dict[str, float]], Dict[str, float]]] = None
    """fit(starting_values) -> fitted_values. Supply it when the claim rests on a
    calibrated parameter, and S10 checks whether the parameter was estimated or
    is simply where the optimiser stopped."""
    starts: Sequence[Dict[str, float]] = ()
    """The starting points to run that fit from -- three or more, spread wide
    enough that a parameter following its start is visible."""

    frozen_config: Optional[Dict[str, Any]] = None
    """The cut points, thresholds and constants the current run recomputed."""
    frozen_config_path: Optional[str] = None
    triggers: Optional[np.ndarray] = None
    """(T, N) bool: which events the rule fired on, for an event study."""
    fill_convention: Optional[str] = None
    bar_includes_signal_period: Optional[bool] = None
    implication_results: Optional[Dict[str, str]] = None
    """Each pre-registered implication mapped to "held", "failed" or "untested"."""
    ask: Optional[Callable[[str], str]] = None
    """Query the source that proposed this hypothesis -- a model, a search tool,
    a knowledge base. P8 uses it to find out whether that source already knew
    what happened after the date the study claims to reason from."""
    hindsight_probes: Sequence[Dict[str, Any]] = ()
    """Questions whose answers became knowable only after `information_boundary`."""
    hindsight_controls: Sequence[Dict[str, Any]] = ()
    """Questions from before it, which the source ought to answer. Without these
    a silent source looks identical to a clean one."""
    information_boundary: str = ""

    validator: Optional[Callable[[Any], Any]] = None
    """A data check this study relies on: validator(data) is truthy when the data
    is acceptable. P7 requires it to be capable of failing."""
    corruptions: Dict[str, Callable[[Any], Any]] = field(default_factory=dict)
    """name -> a function that injects one defect `validator` claims to catch."""
    validator_data: Any = None
    """The data believed clean, which the corruptions are applied to."""

    external_facts: Optional[List[Dict[str, Any]]] = None
    """Facts from outside the pipeline it can be checked against -- the only
    thing that finds an error a value-by-value reproduction shares."""
    """The file they were frozen into. P3 refuses to continue if they drifted."""
    retrained: Optional[bool] = None
    """Does the production pipeline refit on a rolling window, or ship a frozen
    fit? M10 needs it: a relationship that moves is the reason a rolling
    pipeline exists and a defect in a frozen one, and the panel cannot tell
    which is being shipped."""
    claimed_strides: Sequence[int] = ()
    """The coarser cadences the claim is asserted to hold at, as multiples of
    the panel's own. Empty means the claim is about this frequency only, and
    M11 reports the profile without enforcing it."""
    frequency_label: str = "the panel's own frequency"

    positive_control: Optional[np.ndarray] = None
    """A signal known to work on this panel. Defaults to short-horizon reversal,
    which needs no data the study does not already have."""
    control_label: str = "short-horizon reversal"

    # Minimum names per cross-section. The default suits an equity universe;
    # a study on a handful of ETFs or futures must lower it deliberately, and
    # should expect wider null distributions as a consequence.
    min_n: int = 30

    def __post_init__(self) -> None:
        for name in ("signal", "ret", "mask"):
            arr = np.asarray(getattr(self, name))
            if arr.shape != np.asarray(self.signal).shape:
                raise ValueError(f"{name} shape {arr.shape} != signal shape {np.asarray(self.signal).shape}")
            setattr(self, name, arr)
        self.mask = self.mask.astype(bool)


def _retrained(study: "Study") -> Optional[bool]:
    """Supplying a `refit` callable is itself a declaration that the pipeline
    retrains -- A1 could not run against it otherwise. An explicit `retrained`
    always wins, so a study that refits only for the audit can say so."""
    if study.retrained is not None:
        return study.retrained
    return True if study.refit is not None else None


def run(study: Study, prereg: Optional[Prereg] = None, seal: Optional[SealedSplit] = None,
        n_draws: int = 200, seed: int = 0, verbose: bool = True) -> Report:
    rep = Report(claim=study.claim, prereg_id=prereg.id if prereg else None)
    say = (lambda m: print(m, flush=True)) if verbose else (lambda m: None)

    fwd = forward_returns(study.ret, study.horizon)
    sig = mech.orient(study.signal, fwd, study.mask, min_n=study.min_n)
    if not np.array_equal(np.nan_to_num(sig), np.nan_to_num(np.asarray(study.signal, float))):
        rep.notes.append("signal was sign-flipped so that positive IC means 'works'")

    # ---- process -----------------------------------------------------------
    if prereg is None:
        rep.add(Check("P0", "process", "pre-registration", INCONCLUSIVE,
                      detail="no pre-registration: the criterion cannot be shown to predate the result"))
    else:
        rep.add(Check("P0", "process", "pre-registration", PASS,
                      detail=f"{prereg.id} frozen {prereg.created_utc}; "
                             f"{prereg.n_candidates_searched} candidate(s) declared"))
        if not prereg.mechanism.strip():
            # Inconclusive rather than rejected, and the distinction is the whole
            # point. An unexplained anomaly is not a refuted one -- "we do not
            # know why this works" is not evidence that it does not. But it is
            # not a pass either, because the mechanistic axis has nothing to
            # test and the claim has therefore not been judged on it.
            #
            # What it buys is a different downstream treatment, not a verdict:
            # a human has to look at it, the honest candidate count is the whole
            # space that could have been searched rather than the few that were,
            # and it gets reported as an anomaly. What it must never become is a
            # result sized as though it were understood.
            rep.add(Check("P1", "process", "stated mechanism", INCONCLUSIVE,
                          detail="no economic mechanism stated, so nothing on the mechanistic "
                                 "axis could be tested. This is not a rejection -- an anomaly "
                                 "with no explanation can still be real -- but it is not a pass: "
                                 "it needs a human to screen it, it needs the candidate count to "
                                 "reflect the whole space that could have been searched rather "
                                 "than the handful that were, and it must be reported as an "
                                 "anomaly. Do not size it as though it were understood"))
        else:
            rep.add(Check("P1", "process", "stated mechanism", PASS, blocking=False,
                          detail=prereg.mechanism.strip()[:160]))
    if study.frozen_config is not None or study.frozen_config_path:
        rep.add(p3_frozen_config(study.frozen_config or {}, study.frozen_config_path or ""))
    rep.add(p6_mechanism_implications(prereg, study.implication_results))
    rep.add(p4_fill_convention(study.fill_convention, study.bar_includes_signal_period))
    rep.add(p5_external_facts(study.external_facts))
    rep.add(p7_validator_control(study.validator, study.corruptions, study.validator_data))
    rep.add(p8_hindsight_control(study.ask, study.hindsight_probes,
                                 study.hindsight_controls, study.information_boundary))
    if seal is not None:
        st = seal.status()
        rep.add(Check("P2", "process", "test seal", PASS if st is None else INCONCLUSIVE,
                      blocking=False,
                      detail="test period still sealed" if st is None
                      else f"unsealed {st['when']} (config {st['config_hash']}, reads={st['reads']})"))

    # ---- are the inputs alive? ---------------------------------------------
    # Before anything is measured. A frozen input makes every number below it
    # look entirely normal and mean nothing, so this runs first and stops.
    inputs = {"signal": study.signal, "ret": study.ret}
    inputs.update({f"covariate:{k}": v for k, v in study.covariates.items()})
    inputs.update({f"control:{n}": c for n, c in
                   zip(study.control_names or [f"c{i}" for i in range(len(study.controls))],
                       study.controls)})
    rep.add(robust.s7_input_staleness(inputs, dates=study.dates))
    rep.add(robust.m6_input_freshness(inputs, dates=study.dates, horizon=study.horizon))
    rep.add(robust.m5_print_quality(study.ret, dates=study.dates))
    if rep.killers:
        rep.notes.append("stopped at the input checks: a frozen, stale or spike-ridden input "
                         "makes every downstream number meaningless")
        return rep

    # ---- is there anything here at all? ------------------------------------
    # Run first. Every audit below interprets a number; on a signal that is
    # indistinguishable from noise those numbers are themselves noise, and a
    # verdict attributed to the wrong killer teaches the wrong lesson.
    say("[1/5] harness sanity + positive control ...")
    m0 = mech.m0_harness(sig, fwd, study.mask, n_draws=max(40, n_draws // 4), seed=seed,
                         min_n=study.min_n)
    control = (study.positive_control if study.positive_control is not None
               else robust.default_control(study.ret))
    s6 = robust.s6_positive_control(control, fwd, study.mask, horizon=study.horizon,
                                    min_n=study.min_n, label=study.control_label)
    # A measurement is interpretable only if the apparatus could have produced a
    # different one. When a known effect comes back flat on this panel, neither
    # direction survives: failing to find something proves nothing, and finding
    # something at the 98th percentile of a shuffle is what noise does two times
    # in a hundred. So this stops here rather than downgrading one direction and
    # letting the other through.
    #
    # The way out is actionable, which is why the verdict is inconclusive rather
    # than a rejection: short-horizon reversal is only a guess at what this panel
    # should contain, and a study that knows better should say so.
    if s6.outcome == FAIL:
        rep.add(Check(m0.id, m0.axis, m0.name, INCONCLUSIVE, statistic=m0.statistic,
                      threshold=m0.threshold, evidence=m0.evidence,
                      detail=m0.detail + "  -- but see S6: nothing measured on this panel is "
                                         "interpretable in either direction"))
        rep.add(s6)
        rep.notes.append("stopped at the positive control: a known effect is flat on this panel, "
                         "so nothing here could have been established either way. If "
                         "short-horizon reversal is not the right known effect for this universe "
                         "and horizon, supply one that is as `positive_control` and run again.")
        return rep
    rep.add(m0)
    rep.add(s6)
    if rep.killers:
        rep.notes.append("stopped after the harness check: nothing here to attack further")
        return rep

    # ---- statistical -------------------------------------------------------
    say("[2/5] point-in-time audits ...")
    for c in pit.audit(sig, study.ret, study.mask, study.horizon, fwd=fwd,
                       window_based=study.window_based,
                       recompute_at=study.recompute_at, refit=study.refit, seed=seed,
                       min_n=study.min_n, probe_dates=study.probe_dates,
                       n_shuffle=study.n_shuffle):
        rep.add(c)

    ic = rank_ic(sig, fwd, study.mask, min_n=study.min_n)
    summ = ic_summary(ic, horizon=study.horizon)
    thr = deflated_threshold(study.n_candidates_searched)
    ok = np.isfinite(summ["t_nw"]) and abs(summ["t_nw"]) >= thr
    s4 = Check("S4", "statistical", "significance after search", PASS if ok else FAIL,
                  statistic=summ["t_nw"], threshold=thr,
                  detail=(f"IC {summ['mean']:.4f}, ICIR {summ['icir']:.2f}, NW t={summ['t_nw']:.2f}; "
                          f"{study.n_candidates_searched} candidate(s) searched requires |t|>={thr:.2f}; "
                          f"effective n={summ['n_eff']:.0f} (overlap-corrected)"),
                  evidence=summ | {"threshold": thr})
    rep.add(s4)

    if study.seed_metric is not None:
        say("[2b/5] seed stability ...")
        rep.add(robust.s5_seed_stability(study.seed_metric, n_seeds=study.n_seeds,
                                         seed=seed, reported=study.reported_metric))
    else:
        rep.add(Check("S5", "statistical", "seed stability", NA, blocking=False,
                      detail="no seed_metric supplied -- a single run is being taken on faith"))

    rep.add(robust.s9_label_persistence(fwd, study.mask, horizon=study.horizon,
                                       min_n=study.min_n))

    if study.fit_from_start is not None and len(study.starts) >= 3:
        say("[2d/5] identification ...")
        rep.add(robust.s10_identification(study.fit_from_start, study.starts))
    else:
        rep.add(Check("S10", "statistical", "identification", NA, blocking=False,
                      detail="no fit supplied -- if the claim rests on a calibrated "
                             "parameter, it has not been shown to be estimated rather "
                             "than to be where the optimiser stopped"))

    if study.knob_metric is not None and len(study.knob_params) >= 4:
        say("[2c/5] knob sweep ...")
        rep.add(robust.s8_knob_monotonicity(study.knob_metric, study.knob_params))
    else:
        rep.add(Check("S8", "statistical", "knob monotonicity", NA, blocking=False,
                      detail="no parameter sweep supplied -- a chosen setting cannot be "
                             "distinguished from a fitted one"))

    # ---- mechanistic -------------------------------------------------------
    if rep.killers:
        rep.notes.append("stopped after the point-in-time audits: a leak makes every "
                         "downstream number meaningless, so the nulls and the cost gate were not run")
        return rep
    say("[3/5] identity null ...")
    rep.add(mech.m2_identity_null(sig, fwd, study.mask, n_draws=n_draws, seed=seed + 1,
                                  groups=study.groups, min_n=study.min_n))
    say("[4/5] matched null + orthogonalisation ...")
    # Before either of them. A control that reads past its own timestamp does not
    # make M1 and M3 wrong-looking, it makes them look fine and be about
    # something else, so this runs first and carries a veto.
    rep.add(robust.m12_control_integrity(study.controls, study.control_names,
                                         sig, fwd, study.mask, horizon=study.horizon,
                                         min_n=study.min_n))
    rep.add(mech.m1_matched_null(sig, fwd, study.mask, study.covariates,
                                 n_draws=n_draws, seed=seed + 2, min_n=study.min_n))
    rep.add(mech.m3_orthogonalize(sig, fwd, study.mask, study.controls,
                                  study.control_names, horizon=study.horizon, min_n=study.min_n))
    rep.add(mech.m4_beats_naive(sig, study.naive_baseline, fwd, study.mask,
                                horizon=study.horizon, label=study.naive_label, min_n=study.min_n))
    if study.triggers is not None:
        say("[4b/5] event integrity ...")
        rep.add(robust.m7_event_integrity(study.triggers, fwd, study.mask,
                                          n_draws=max(50, n_draws // 2), seed=seed + 3))
    else:
        rep.add(Check("M7", "mechanistic", "event integrity", NA, blocking=False,
                      detail="not an event study -- pass `triggers` if it is"))
    rep.add(robust.m8_threshold_or_slope(sig, fwd, study.mask, min_n=study.min_n))
    rep.add(robust.m9_cross_sectional_independence(sig, fwd, study.mask,
                                                   min_n=max(20, study.min_n // 4)))
    rep.add(robust.m10_stationarity(sig, fwd, study.mask, retrained=_retrained(study),
                                    horizon=study.horizon, min_n=study.min_n))
    rep.add(robust.m11_frequency_transfer(sig, study.ret, study.mask, horizon=study.horizon,
                                          claimed_strides=study.claimed_strides,
                                          min_n=study.min_n))

    # ---- economic ----------------------------------------------------------
    say("[5/5] cost gate ...")
    rep.add(econ.e4_cost_convention(study.price_convention, study.cost_components))
    if study.dollar_volume is not None:
        rep.add(econ.e5_capacity(sig, study.mask, study.dollar_volume, q=study.quantile,
                                 hold=max(1, study.horizon), capital=study.capital,
                                 max_participation=study.max_participation, min_n=study.min_n))
    else:
        rep.add(Check("E5", "economic", "capacity", NA, blocking=False,
                      detail="no volume supplied -- an edge in basis points says nothing about "
                             "how much money fits behind it"))
    pf = econ.quantile_portfolio(sig, study.ret, study.mask, q=study.quantile,
                                 hold=study.horizon, long_short=study.long_short,
                                 min_n=study.min_n)
    rep.add(econ.cost_gate(pf["gross"], pf["turnover"], study.cost_bp))
    rep.add(econ.trade_block_check(pf["gross"], pf["turnover"], study.cost_bp,
                                   hold=study.horizon, rebalances=pf["rebalances"]))
    rep.add(econ.e6_entry_constraints(sig, study.ret, study.mask, q=study.quantile,
                                      hold=study.horizon, long_short=study.long_short,
                                      cost_bp=study.cost_bp,
                                      dollar_volume=study.dollar_volume,
                                      min_dollar_volume=study.min_dollar_volume,
                                      price_limit=study.price_limit, min_n=study.min_n))
    rep.add(econ.e3_execution_delay(sig, study.ret, study.mask, horizon=study.horizon,
                                    q=study.quantile, cost_bp=study.cost_bp,
                                    tradable_ret=study.tradable_ret,
                                    price_source=study.price_source, min_n=study.min_n))
    return rep
