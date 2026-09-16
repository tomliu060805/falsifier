"""Point-in-time audits.

The premise: do not read the code and conclude it looks fine. Perturb the
pipeline and check that the output moves in the direction the pipeline's own
promises require. The judgement comes from a declared invariant, never from
the result looking plausible.

Four audits, in decreasing order of power and increasing order of
applicability. Run the strongest one your study design can support, and let
the report say plainly when only the weak ones applied.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .nulls import cs_shuffle
from .stats import forward_returns, ic_summary, rank_ic
from .verdict import FAIL, INCONCLUSIVE, NA, PASS, Check


def _mean_ic(signal, fwd, mask, min_n=30) -> float:
    v = rank_ic(signal, fwd, mask, min_n=min_n)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def a0_truncation(production_signal: np.ndarray,
                  recompute_at: Callable[[int], np.ndarray],
                  probe_dates: Sequence[int],
                  tol: float = 1e-8) -> Check:
    """A0 -- rebuild the signal for date t using only data up to t.

    The decisive audit. If a value computed from a truncated history differs
    from the value the production pipeline published for that same date, the
    production pipeline saw something it should not have. Nothing else needs
    to be argued.
    """
    diffs: List[float] = []
    for t in probe_dates:
        a = np.asarray(production_signal[t], float)
        b = np.asarray(recompute_at(int(t)), float)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() == 0:
            continue
        scale = max(np.nanstd(a[m]), 1e-12)
        diffs.append(float(np.max(np.abs(a[m] - b[m])) / scale))
    if not diffs:
        return Check("A0", "statistical", "truncation rebuild", INCONCLUSIVE,
                     detail="no probe date produced comparable values")
    worst = max(diffs)
    ok = worst <= tol
    return Check("A0", "statistical", "truncation rebuild", PASS if ok else FAIL,
                 statistic=worst, threshold=tol,
                 detail=("production signal reproduces from truncated history"
                         if ok else
                         f"{sum(d > tol for d in diffs)}/{len(diffs)} probe dates differ; "
                         "the published value used data from after its own timestamp"),
                 evidence={"n_probe": len(diffs), "worst_rel_diff": worst})


def a1_label_shuffle_refit(refit: Callable[[np.ndarray, int], np.ndarray],
                           fwd: np.ndarray, mask: np.ndarray,
                           n_draws: int = 20, seed: int = 0,
                           min_n: int = 30) -> Check:
    """A1 -- refit the whole pipeline on shuffled labels.

    A pipeline that fits anything (feature selection, hyper-parameters,
    normalisation constants) on the full sample will still score above zero
    when the labels are pure noise. Only a refit exposes that; permuting the
    labels against a frozen signal cannot.
    """
    g = np.random.default_rng(seed)
    vals = []
    for i in range(n_draws):
        sh = cs_shuffle(fwd, mask, seed=int(g.integers(0, 2 ** 31 - 1)))
        sig = refit(sh, i)
        vals.append(_mean_ic(sig, sh, mask, min_n))
    v = np.array([x for x in vals if np.isfinite(x)])
    if v.size < 3:
        return Check("A1", "statistical", "label-shuffle refit", INCONCLUSIVE,
                     detail="refit returned too few usable draws")
    t = float(v.mean() / (v.std(ddof=1) / np.sqrt(v.size))) if v.std(ddof=1) > 0 else 0.0
    ok = abs(t) < 3.0
    return Check("A1", "statistical", "label-shuffle refit", PASS if ok else FAIL,
                 statistic=float(v.mean()), threshold=0.0,
                 detail=("IC collapses to noise when labels are destroyed"
                         if ok else
                         f"pipeline still scores IC={v.mean():.4f} (t={t:.1f}) on shuffled labels: "
                         "something is fitted on the full sample"),
                 evidence={"n_draws": int(v.size), "t_of_mean": t})


def a2_feature_shift(signal: np.ndarray, fwd: np.ndarray, mask: np.ndarray,
                     window_based: bool = True, min_n: int = 30,
                     span: int = 4) -> Check:
    """A2 -- slide the feature through time and locate the label boundary.

    Score the feature at shifts -span..+span. A trailing-window feature scores
    flat while its window sits entirely in the past, then steps up the moment
    the window swallows its first label day, and keeps stepping up as it
    swallows more. The interesting quantity is not how big any step is but
    *where the first one happens*: that position is the boundary between legal
    and illegal information, and for an honestly timestamped signal it must lie
    between shift 0 and shift +1 -- exactly one step in front of publication.

    A boundary at -1 -> 0 says the value published for t already contains the
    label. That is what an off-by-one looks like from outside the code, and
    unlike a code review it does not depend on anyone spotting the line.

    Two things this deliberately does not do. It does not compare shift 0
    against shift +1 alone: past one step of horizon a leaked feature keeps
    gaining as it slides forward for the same reason an honest one does, so
    that comparison clears both. And it does not take the largest step as the
    boundary: with a multi-step horizon the second and third label days are
    worth about as much as the first, and which of them wins is noise.
    """
    sig = np.asarray(signal, float)
    ic0 = _mean_ic(sig, fwd, mask, min_n)
    sign = 1.0 if (np.isfinite(ic0) and ic0 >= 0) else -1.0
    sig = sig * sign

    # Only shifts up to +1 are needed to locate the boundary, and stopping there
    # avoids a second artefact: once the window slides clear of the label period
    # entirely the increments turn negative, and those have nothing to do with
    # where legality ends.
    shifts = list(range(-span, 2))
    ics = {}
    for k in shifts:
        if k == 0:
            v = sig
        else:
            v = np.full_like(sig, np.nan)
            if k > 0:
                v[:-k] = sig[k:]
            else:
                v[-k:] = sig[:k]
        ics[k] = abs(_mean_ic(v, fwd, mask, min_n))

    steps = [(k, ics[k + 1] - ics[k]) for k in shifts[:-1]
             if np.isfinite(ics[k]) and np.isfinite(ics[k + 1])]
    ev = {"abs_ic_by_shift": {str(k): float(ics[k]) for k in shifts},
          "increments": {f"{k:+d}->{k+1:+d}": float(d) for k, d in steps}}

    if not window_based:
        return Check("A2", "statistical", "feature time-shift", NA, blocking=False,
                     detail="signal declared not trailing-window; the boundary carries no invariant",
                     evidence=ev)
    if len(steps) < 4:
        return Check("A2", "statistical", "feature time-shift", INCONCLUSIVE,
                     detail="too few usable shifts to locate a boundary", evidence=ev)

    # Split "quiet" steps from "the window just ate a label day" at the widest
    # gap in the sorted increments, and judge that gap against the median of the
    # remaining gaps. A rule anchored on a fixed fraction of the increments
    # breaks on the case that matters most -- when the leak is early, most
    # increments are already contaminated.
    #
    # Increments are floored at zero first. A boundary is a *rise*, so a fall
    # carries no information about where legality ends -- and left unclipped a
    # large fall wrecks the split. That is not hypothetical: a same-day feature
    # scored against a same-day label produces a spike rather than a step, and
    # the collapse on the far side of the spike opens a gap at the bottom of the
    # sorted increments wider than the real one at the top.
    d = np.sort(np.maximum(np.array([x[1] for x in steps]), 0.0))
    gaps = np.diff(d)
    gi = int(np.argmax(gaps))
    gap = float(gaps[gi])
    others = np.delete(gaps, gi)
    scale = float(np.median(others)) if others.size else 0.0
    ratio = (gap / scale) if scale > 0 else (np.inf if gap > 0 else 0.0)
    ev |= {"gap": gap, "other_gap_scale": scale, "gap_ratio": float(ratio),
           "n_quiet": int(gi + 1)}

    # Every step looks like every other step: the score does not respond to
    # crossing any particular date, so there is no boundary to find. Reporting
    # that beats naming one at random.
    if ratio < 5.0:
        return Check("A2", "statistical", "feature time-shift", INCONCLUSIVE,
                     statistic=float(ratio), threshold=5.0,
                     detail="no step stands out from the rest; the score does not respond to "
                            "crossing any particular date, so no label boundary could be located",
                     evidence=ev)

    thr = float(d[gi] + gap / 2.0)
    hit = next(((k, inc) for k, inc in steps if max(inc, 0.0) > thr), None)
    if hit is None:  # every increment identical -- nothing varies with time
        return Check("A2", "statistical", "feature time-shift", INCONCLUSIVE,
                     detail="the score does not change when the feature is shifted at all",
                     evidence=ev)
    boundary, jump = hit

    # A gap that is wide only relative to other gaps can still be smaller than
    # the sampling error of the scores it separates. Require the jump to clear a
    # multiple of the standard error of the mean IC, so a profile made entirely
    # of noise cannot nominate one of its wiggles as a leak.
    v0 = rank_ic(sig, fwd, mask, min_n=min_n)
    v0 = v0[np.isfinite(v0)]
    se = float(v0.std(ddof=1) / np.sqrt(v0.size)) if v0.size > 1 else np.nan
    floor = 5.0 * np.sqrt(2.0) * se if np.isfinite(se) else 0.0
    ev |= {"jump": float(jump), "ic_stderr": se, "jump_floor": floor}
    if np.isfinite(floor) and jump <= floor:
        return Check("A2", "statistical", "feature time-shift", INCONCLUSIVE,
                     statistic=float(jump), threshold=floor,
                     detail=f"the largest rise ({jump:.4g}) is within the sampling error of the "
                            f"scores it separates ({floor:.4g}); no boundary can be located",
                     evidence=ev)
    ev |= {"boundary_at": f"{boundary:+d}->{boundary+1:+d}", "step_threshold": thr}
    ok = boundary == 0
    trace = ", ".join(f"{k:+d}:{ics[k]:.4f}" for k in shifts if np.isfinite(ics[k]))
    return Check("A2", "statistical", "feature time-shift", PASS if ok else FAIL,
                 statistic=float(boundary), threshold=0.0,
                 detail=(f"label boundary at shift 0->+1, one step ahead of publication, "
                         f"as an honest trailing window requires  [{trace}]"
                         if ok else
                         f"label boundary at shift {boundary:+d}->{boundary+1:+d}, not 0->+1: the signal "
                         f"published for t overlaps the period it claims to forecast  [{trace}]. Two "
                         "causes look identical from here -- the signal peeked, or the label is a "
                         "stale print that still contains period t. Both make the score not a "
                         "forecast; which one it is decides what to fix"),
                 evidence=ev)


def a3_label_delay(signal: np.ndarray, ret: np.ndarray, mask: np.ndarray,
                   horizon: int, max_delay: int = 3, min_n: int = 30) -> Check:
    """A3 -- enter one day late and check the decay against the overlap floor.

    Two windows [t+1, t+h] and [t+1+d, t+h+d] share (h-d)/h of their length,
    so even a signal with no persistence at all keeps roughly that fraction of
    its IC. Decay far below the overlap floor means the signal was never about
    the window -- it was about one specific day, which is what an off-by-one
    alignment error looks like from the outside.
    """
    if horizon < 3:
        return Check("A3", "statistical", "label delay decay", NA, blocking=False,
                     detail=f"horizon={horizon}: overlap floor is uninformative below h=3")
    ics = []
    for d in range(0, max_delay + 1):
        f = forward_returns(ret, horizon, delay=d)
        ics.append(_mean_ic(signal, f, mask, min_n))
    base = ics[0]
    if not np.isfinite(base) or abs(base) < 1e-6:
        return Check("A3", "statistical", "label delay decay", INCONCLUSIVE,
                     detail="baseline IC is ~0, nothing to decay")
    ratio = float(ics[1] / base)
    floor = 0.4 * (horizon - 1) / horizon
    ok = ratio >= floor
    return Check("A3", "statistical", "label delay decay", PASS if ok else FAIL,
                 blocking=False, statistic=ratio, threshold=floor,
                 detail=("IC decays smoothly with entry delay"
                         if ok else
                         f"IC collapses to {ratio:.0%} of baseline after a one-step delay, "
                         f"far below the {floor:.0%} the window overlap alone guarantees. Either the "
                         "signal is misaligned with the label, or it lives on a single step and its "
                         f"{horizon}-step IC is really a one-step effect -- both change what may be claimed"),
                 evidence={"ic_by_delay": [float(x) for x in ics], "overlap_floor": floor})


def audit(signal: np.ndarray, ret: np.ndarray, mask: np.ndarray, horizon: int,
          fwd: Optional[np.ndarray] = None,
          window_based: bool = True,
          recompute_at: Optional[Callable[[int], np.ndarray]] = None,
          probe_dates: Optional[Sequence[int]] = None,
          refit: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
          n_shuffle: int = 20, seed: int = 0, min_n: int = 30) -> List[Check]:
    """Run every audit the study design supports."""
    fwd = forward_returns(ret, horizon) if fwd is None else fwd
    out: List[Check] = []
    if recompute_at is not None:
        if probe_dates is None:
            T = signal.shape[0]
            probe_dates = np.linspace(int(T * 0.2), T - horizon - 2, 12).astype(int)
        out.append(a0_truncation(signal, recompute_at, probe_dates))
    else:
        out.append(Check("A0", "statistical", "truncation rebuild", NA, blocking=False,
                         detail="no recompute_at supplied -- the decisive audit did not run"))
    if refit is not None:
        out.append(a1_label_shuffle_refit(refit, fwd, mask, n_draws=n_shuffle, seed=seed, min_n=min_n))
    else:
        out.append(Check("A1", "statistical", "label-shuffle refit", NA, blocking=False,
                         detail="no refit supplied; frozen signal cannot be tested for in-sample fitting"))
    a4 = a4_shift_power(signal, ret, mask, horizon=horizon, fwd=fwd, min_n=min_n)
    a2 = a2_feature_shift(signal, fwd, mask, window_based=window_based, min_n=min_n)
    if a4.outcome == INCONCLUSIVE and a4.blocking:
        # A2's number is not about the signal, so do not let it read as though it
        # were -- in either direction. A4 carries the veto and the explanation.
        a2 = replace(a2, outcome=INCONCLUSIVE, blocking=False,
                     detail=a2.detail + "  -- but see A4: the step this check needs is below "
                                        "the panel's sampling error, so this verdict is about "
                                        "the detector, not the signal")
    out.append(a4)
    a3 = a3_label_delay(signal, ret, mask, horizon, min_n=min_n)

    # A2 exists as a stand-in for A0: it infers from a signal's behaviour what A0
    # establishes directly by rebuilding it. When A0 has actually run and
    # returned a verdict, letting A2 keep its veto means the weaker check can
    # overrule the stronger one -- and it does, because a fitted model's output
    # has no trailing-window boundary for A2 to find, so A2 goes INCONCLUSIVE
    # and blocks a claim that A0 just cleared. With A0 decided, A2 corroborates.
    a0 = next(c for c in out if c.id == "A0")
    if a0.outcome in (PASS, FAIL):
        a2 = replace(a2, blocking=False)
        if a0.outcome == PASS and a2.outcome == FAIL:
            a2 = replace(a2, detail=a2.detail + "  [!] A0 rebuilt this signal exactly from "
                                                "truncated history, so the two audits disagree "
                                                "-- resolve before trusting either")
    out.append(a2)
    out.append(a3)
    return out

def a4_shift_power(signal: np.ndarray, ret: np.ndarray, mask: np.ndarray,
                   horizon: int = 1, fwd: Optional[np.ndarray] = None,
                   min_n: int = 30, weak: float = 3.0, span: int = 4) -> Check:
    """A4 -- is the step A2 needs to see bigger than this panel's noise?

    A2 finds a leak by sliding the feature across the label and locating where
    the score first steps up. The premise holds for a raw trailing window, where
    one step of slide swallows one whole label day. It fails for a smoothed
    signal, and fails quietly: when the score is an average over a span of L,
    sliding one step changes only about 1/L of it, so the step shrinks roughly
    as 1/L while the sampling error of the IC series does not shrink at all.
    Below some amount of smoothing the step disappears into the noise, A2
    returns INCONCLUSIVE, and **that reads as a suspicion of look-ahead when it
    is a statement about the instrument.**

    So measure the instrument. The quantity is the one the failure is about:

        step  = |IC| at shift +1  minus  |IC| at shift 0
        floor = A2's own detection floor, 5 * sqrt(2) * SE of the IC series

    and the ratio between them is how much power A2 has on *this* signal. On a
    five-day window it is about 10; the same signal put through a ten-day EMA
    gives about 3, and a forty-day EMA about 1.5. Past that the detector is
    blind and says INCONCLUSIVE regardless of what is true.

    Same shape as S6 and P7, one axis over. S6 asks whether the panel can detect
    an effect it is known to contain; P7 whether a guard can be made to fail;
    this asks whether the leak detector can see a step at all. All three exist
    because a null result from an apparatus with no demonstrated power is not
    evidence of absence -- and this one because it is routinely read as the
    opposite.

    When it comes back weak the answer is not a different statistic on the same
    array. It is `A0` and `A1`: rebuilding from truncated history and refitting
    on destroyed labels are unaffected by smoothing, and they settle it. That
    costs wiring the pipeline in, and the cost is the point -- without it the
    decisive audits read NA forever.
    """
    fwd = forward_returns(ret, horizon) if fwd is None else fwd
    m = np.asarray(mask, bool)
    sig = np.asarray(signal, float)

    def _abs_ic(x):
        v = rank_ic(x, fwd, m, min_n=min_n)
        v = v[np.isfinite(v)]
        if v.size < 4:
            return np.nan, np.nan
        return float(abs(v.mean())), float(v.std(ddof=1) / np.sqrt(v.size))

    # Across the same shifts A2 scans, not at one chosen position. A first
    # version measured the step at 0 -> +1 only, which is where the boundary of
    # an *honest* signal sits -- so on a leaked signal, whose step is somewhere
    # else, it measured a place with no step and declared A2 blind. It then
    # downgraded an A2 that was successfully rejecting. The power of a detector
    # is not a property of one location, and this must never be able to say "no
    # power" about a signal on which A2 found a boundary.
    def _shift(x, k):
        o = np.full_like(x, np.nan)
        if k > 0:
            o[:-k] = x[k:]
        elif k < 0:
            o[-k:] = x[:k]
        else:
            o = np.array(x, float)
        return o

    scores, se = {}, np.nan
    for k in range(-span, 2):
        v, s_ = _abs_ic(_shift(sig, k))
        scores[k] = v
        if k == 0:
            se = s_
    vals = np.array([scores[k] for k in sorted(scores)], float)
    if not (np.isfinite(vals).all() and np.isfinite(se) and se > 0):
        return Check("A4", "statistical", "shift-detector power", INCONCLUSIVE, blocking=False,
                     detail="too few usable cross-sections to size A2's steps against its noise")

    floor = 5.0 * np.sqrt(2.0) * se
    incs = np.diff(vals)
    step = float(incs.max()) if incs.size else 0.0
    where = int(sorted(scores)[int(np.argmax(incs))]) if incs.size else 0
    ratio = step / floor if floor > 0 else np.inf
    ev = {"profile": {str(k): scores[k] for k in sorted(scores)}, "largest_step": step,
          "largest_step_at": f"{where}->{where + 1}", "floor": floor, "ratio": ratio}
    body = (f"the largest one-step rise anywhere in the shift profile is {step:+.4f} "
            f"(at {where}->{where + 1}) against a detection floor of {floor:.4f} -- {ratio:.2f}x")

    if ratio >= weak:
        return Check("A4", "statistical", "shift-detector power", PASS,
                     statistic=ratio, threshold=weak, evidence=ev,
                     detail=f"{body}. A2 has room to see a misalignment on this signal, so its "
                            f"verdict can be read as being about the signal")
    if ratio >= 1.0:
        return Check("A4", "statistical", "shift-detector power", FAIL, blocking=False,
                     statistic=ratio, threshold=weak, evidence=ev,
                     detail=(f"{body}, which is thin. A2 will catch a wholesale one-step error "
                             f"here and may well miss a partial one -- a composite with a single "
                             f"early input, or a shift inside a smoothed component. Treat its "
                             f"verdict as covering the coarse case; `recompute_at` and `refit` "
                             f"settle the rest, and are unaffected by smoothing"))
    # Inconclusive rather than failed, and the word matters: nothing here says the
    # signal is wrong. It says the instrument cannot tell, which is the claim not
    # having been judged on this axis -- and blocking, because if A0 and A1 were
    # not wired either, then nothing has looked at the leak question at all.
    return Check("A4", "statistical", "shift-detector power", INCONCLUSIVE,
                 statistic=ratio, threshold=weak, evidence=ev,
                 detail=(f"{body}, below the floor. **A2 cannot decide anything about this "
                         f"signal**, and an INCONCLUSIVE from it here is a fact about A2, not a "
                         f"suspicion about the signal -- reading it the other way is how a clean "
                         f"result gets treated as a leak. This is what smoothing does: the step "
                         f"shrinks as 1/span while the sampling error does not. Wire "
                         f"`recompute_at` and `refit` -- A0 and A1 rebuild and refit rather than "
                         f"slide, so smoothing does not touch them"))
