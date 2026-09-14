"""Three checks for failure modes the rest of the battery does not see.

Each one closes a hole in `taxonomy` that shows up repeatedly in real
post-mortems, and each answers a question the other checks structurally cannot:

  S5  would this survive a different random seed?
  S6  can this panel detect an effect it is known to contain?
  S7  is any input still updating?

S6 is the one worth reading twice. Every other check here can reject a claim;
S6 is the only one that can stop a rejection from meaning anything. A null
result on a pipeline that cannot detect a known effect is not evidence of
absence, it is evidence of nothing, and the difference decides whether a line
of work closes or stays open.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np

from .nulls import percentile_of
from .stats import newey_west_t, rank_ic
from .verdict import FAIL, INCONCLUSIVE, NA, PASS, Check


def s5_seed_stability(metric_of_seed: Callable[[int], float], n_seeds: int = 20,
                      seed: int = 0, min_positive: float = 0.8,
                      reported: Optional[float] = None) -> Check:
    """S5 -- rerun under different randomness and look at the distribution.

    A configuration that works once has been observed once. Where the signal is
    weak relative to the noise in training -- which is where most of this work
    happens -- the spread across seeds routinely exceeds the effect, and the
    share of seeds that come out positive is a far better summary than any
    single run or than the mean.

    Pass ``reported`` to check the number actually being claimed against the
    distribution it came from. A result sitting in the top decile of its own
    seed distribution was chosen, whether or not anyone meant to choose it.
    """
    g = np.random.default_rng(seed)
    seeds = g.integers(0, 2 ** 31 - 1, size=n_seeds)
    vals = np.array([metric_of_seed(int(s)) for s in seeds], float)
    v = vals[np.isfinite(vals)]
    if v.size < 5:
        return Check("S5", "statistical", "seed stability", INCONCLUSIVE,
                     detail=f"only {v.size} usable runs")
    pos = float((v > 0).mean())
    ev = {"n_seeds": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
          "positive_fraction": pos, "min": float(v.min()), "max": float(v.max())}
    detail = (f"{v.size} seeds: mean {v.mean():+.4g} ± {v.std(ddof=1):.4g}, "
              f"{pos:.0%} positive, range [{v.min():+.4g}, {v.max():+.4g}]")
    ok = pos >= min_positive
    if reported is not None and np.isfinite(reported):
        pct = float((v < reported).mean() * 100)
        ev["reported"] = float(reported)
        ev["reported_percentile"] = pct
        detail += f"; the reported {reported:+.4g} sits at the {pct:.0f}th pct of that spread"
        if pct >= 90.0:
            ok = False
            detail += " -- that number was picked, deliberately or not"
    return Check("S5", "statistical", "seed stability", PASS if ok else FAIL,
                 statistic=pos, threshold=min_positive, detail=detail, evidence=ev)


def s6_positive_control(control: np.ndarray, fwd: np.ndarray, mask: np.ndarray,
                        horizon: int = 1, min_t: float = 3.0, min_n: int = 30,
                        label: str = "positive control") -> Check:
    """S6 -- make the panel find something it is known to contain.

    Reporting a zero is a claim about the world only if the apparatus could have
    reported a non-zero. Run a known effect through the identical panel, mask,
    horizon and scoring code; if that comes back flat too, the zero describes the
    pipeline rather than the market, and the honest verdict is inconclusive.
    """
    ic = rank_ic(control, fwd, mask, min_n=min_n)
    t = newey_west_t(ic, lags=(horizon - 1) if horizon > 1 else None)
    m = float(np.nanmean(ic))
    ok = np.isfinite(t) and abs(t) >= min_t
    return Check("S6", "statistical", "positive control", PASS if ok else FAIL,
                 blocking=False, statistic=t, threshold=min_t,
                 detail=(f"{label} scores IC {m:+.4f} (t={t:.2f}) on this panel, so a null "
                         "result here would mean something"
                         if ok else
                         f"{label} scores only IC {m:+.4f} (t={t:.2f}): this panel has not been "
                         "shown able to detect anything, so a null result from it is "
                         "uninterpretable"),
                 evidence={"ic": m, "t": t, "label": label})


def default_control(ret: np.ndarray, lookback: int = 5) -> np.ndarray:
    """Short-horizon reversal, built from the return panel itself.

    Not the strongest control, but it needs no data the study does not already
    have, which is the difference between a check that runs and one that is
    always skipped.
    """
    r = np.nan_to_num(np.asarray(ret, float), nan=0.0)
    c = np.cumsum(np.vstack([np.zeros((1, r.shape[1])), r]), axis=0)
    out = np.full_like(r, np.nan)
    out[lookback - 1:] = -(c[lookback:] - c[:-lookback])
    return out


def _frozen_runs(a: np.ndarray, share: float = 0.98) -> Dict[str, float]:
    """Longest stretch over which almost the whole panel stopped changing.

    Deliberately panel-level. A single series holding one value for months is
    ordinary -- a halted name, a rate that did not move -- while an entire panel
    doing it at once is a cache that stopped updating or a filter that started
    matching nothing. Only the second is worth an alarm.
    """
    x = np.asarray(a, float)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] < 3:
        return {"run": 0.0, "start": -1.0, "end": -1.0}
    prev, cur = x[:-1], x[1:]
    same = (prev == cur) | (~np.isfinite(prev) & ~np.isfinite(cur))
    live = np.isfinite(prev) | np.isfinite(cur)
    n_live = live.sum(axis=1)
    frac = np.divide(np.where(live, same, False).sum(axis=1), np.maximum(n_live, 1),
                     out=np.zeros(len(n_live)), where=n_live > 0)
    flat = (frac >= share) & (n_live > 0)
    # A series that never moves is a static characteristic -- size at a point,
    # an industry code, a sector dummy -- not a feed that died. What this is
    # looking for is something that *was* moving and stopped, so anything flat
    # essentially throughout is reported as static and left alone.
    flat_share = float(flat.mean()) if flat.size else 1.0
    if flat_share >= 0.95:
        return {"run": 0.0, "start": -1.0, "end": -1.0, "flat_share": flat_share,
                "static": 1.0}

    best = run = 0
    start = end = -1
    for i, f in enumerate(flat):
        if f:
            run += 1
            if run > best:
                best, start, end = run, i - run + 1, i + 1
        else:
            run = 0
    return {"run": float(best), "start": float(start), "end": float(end),
            "flat_share": flat_share, "static": 0.0}


def s7_input_staleness(inputs: Dict[str, np.ndarray], min_run: int = 20,
                       dates: Optional[np.ndarray] = None) -> Check:
    """S7 -- check every input is still moving.

    The judgement is the last date a series changed, not whether its values look
    plausible. A frozen cache looks entirely normal: the numbers are real, they
    are simply the same numbers as last month, and every downstream statistic is
    computed without complaint.
    """
    if not inputs:
        return Check("S7", "mechanistic", "input staleness", NA, blocking=False,
                     detail="no inputs supplied")
    # -1 so that a genuinely moving input (longest flat run 0) still registers
    # as checked. Starting at 0 conflates "nothing is frozen" with "there was
    # nothing to check", and reports a healthy panel as time-invariant.
    worst_name, worst = None, {"run": -1.0, "start": -1.0, "end": -1.0}
    found, static = {}, []
    for name, arr in inputs.items():
        a = np.asarray(arr)
        if a.dtype == bool or a.size == 0:
            continue
        r = _frozen_runs(a)
        if r.get("static"):
            static.append(name)     # surfaced, not silently dropped
            continue
        found[name] = r["run"]
        if r["run"] > worst["run"]:
            worst_name, worst = name, r
    if worst_name is None:
        return Check("S7", "mechanistic", "input staleness", NA, blocking=False,
                     detail=("every numeric input is time-invariant"
                             + (f" ({', '.join(static)})" if static else "")),
                     evidence={"static": static})
    run = worst["run"]
    ok = run < min_run
    lo, hi = int(worst["start"]), int(worst["end"])
    when = ""
    if dates is not None and 0 <= lo < len(dates) and 0 <= hi < len(dates):
        when = f" ({dates[lo]} to {dates[hi]})"
    return Check("S7", "mechanistic", "input staleness", PASS if ok else FAIL,
                 statistic=run, threshold=min_run,
                 detail=(f"every input keeps moving (longest panel-wide flat stretch "
                         f"{int(run)} steps, in '{worst_name}')"
                         if ok else
                         f"'{worst_name}' stops changing for {int(run)} consecutive steps{when} "
                         "-- the whole panel at once, which is a cache that stopped updating or "
                         "a filter that began matching nothing, not a quiet market"),
                 evidence={"longest_flat_run": found, "worst": worst_name,
                           "treated_as_static": static})


def s8_knob_monotonicity(metric_of: Callable[[float, str], float],
                         params: Sequence[float],
                         segments: Sequence[str] = ("train", "valid"),
                         rise: float = 0.7, fall: float = -0.4) -> Check:
    """S8 -- turn the knob and watch both segments at once.

    Overfitting has a shape, and it is visible long before any single
    configuration is chosen: the training score climbs monotonically as the knob
    is turned while the validation score slides monotonically the other way.
    Looking at one segment hides it, and looking at the best configuration hides
    it too -- by then the knob has been set and the evidence discarded.

    This is also why a swept parameter beats a picked one as evidence. A single
    setting that works can always be defended; a whole family moving the wrong
    way cannot, which makes the sweep a rejection of the region rather than of
    one point.
    """
    from scipy.stats import spearmanr

    x = np.asarray(list(params), float)
    if x.size < 4:
        return Check("S8", "statistical", "knob monotonicity", INCONCLUSIVE,
                     detail=f"only {x.size} settings; a sweep needs at least four")
    curves = {}
    for seg in segments:
        curves[seg] = np.array([metric_of(float(p), seg) for p in x], float)
    a, b = curves[segments[0]], curves[segments[1]]
    ok_mask = np.isfinite(a) & np.isfinite(b)
    if ok_mask.sum() < 4:
        return Check("S8", "statistical", "knob monotonicity", INCONCLUSIVE,
                     detail="too few settings returned usable values")
    rho_a = float(spearmanr(x[ok_mask], a[ok_mask]).statistic)
    rho_b = float(spearmanr(x[ok_mask], b[ok_mask]).statistic)
    ev = {"params": x.tolist(),
          segments[0]: [float(v) for v in a], segments[1]: [float(v) for v in b],
          f"rho_{segments[0]}": rho_a, f"rho_{segments[1]}": rho_b}
    bad = rho_a >= rise and rho_b <= fall
    return Check("S8", "statistical", "knob monotonicity", FAIL if bad else PASS,
                 statistic=rho_a - rho_b, threshold=rise - fall,
                 detail=(f"the knob moves {segments[0]} rho {rho_a:+.2f} and {segments[1]} "
                         f"rho {rho_b:+.2f} -- no systematic divergence"
                         if not bad else
                         f"{segments[0]} rises monotonically with the knob (rho {rho_a:+.2f}) "
                         f"while {segments[1]} falls (rho {rho_b:+.2f}): the textbook overfitting "
                         "signature, and it rejects the whole region rather than one setting"),
                 evidence=ev)


def m5_print_quality(ret: np.ndarray, dates: Optional[np.ndarray] = None,
                     k: float = 6.0, revert: float = 0.6,
                     max_share: float = 0.002) -> Check:
    """M5 -- look for prints that jump and immediately come back.

    A constructed series -- an index stitched from constituents, a continuous
    contract spliced across rolls, a bar built from ticks -- can carry moves
    that never happened: one bad print, then the next bar undoing it. They are
    rare enough to survive a glance at a chart and common enough to dominate a
    tail statistic, and a defensive overlay trained on them looks superb,
    because the crashes it learns to avoid are arithmetic rather than market.

    The signature is specific: a move many robust deviations wide, reversed by
    most of its own size on the very next step. Genuine dislocations are not
    usually undone that exactly, that fast.
    """
    r = np.asarray(ret, float)
    if r.ndim == 1:
        r = r[:, None]
    if r.shape[0] < 10:
        return Check("M5", "mechanistic", "print quality", INCONCLUSIVE,
                     detail="series too short to characterise")
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(r, axis=0)
        scale = 1.4826 * np.nanmedian(np.abs(r - med), axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, np.nan)

    cur, nxt = r[:-1], r[1:]
    big = np.abs(cur) > k * scale
    undone = (np.sign(nxt) != np.sign(cur)) & (np.abs(nxt) > revert * np.abs(cur))
    bad = big & undone & np.isfinite(cur) & np.isfinite(nxt)
    n_obs = int((np.isfinite(cur) & np.isfinite(nxt)).sum())
    if n_obs < 100:
        return Check("M5", "mechanistic", "print quality", INCONCLUSIVE,
                     detail="too few usable observations")
    share = float(bad.sum() / n_obs)
    per_row = bad.sum(axis=1)
    worst = np.argsort(-per_row)[:3]
    when = ""
    if dates is not None and len(dates) > int(worst.max()):
        when = "; worst at " + ", ".join(str(dates[int(i)]) for i in worst if per_row[int(i)] > 0)
    ok = share <= max_share
    return Check("M5", "mechanistic", "print quality", PASS if ok else FAIL,
                 statistic=share * 100, threshold=max_share * 100,
                 detail=(f"{share:.3%} of steps are spike-and-revert, within what real prints do"
                         if ok else
                         f"{share:.2%} of steps jump past {k} robust deviations and are undone on "
                         f"the next step{when} -- moves that did not happen. Anything trained to "
                         "avoid these has learned arithmetic, not the market"),
                 evidence={"share": share, "n_obs": n_obs, "n_bad": int(bad.sum())})


def m6_input_freshness(inputs: Dict[str, np.ndarray], dates: Optional[np.ndarray] = None,
                       max_lag: int = 2, horizon: int = 1) -> Check:
    """M6 -- check every input reaches the same date as the rest.

    Distinct from M5 and from S7. S7 asks whether a series stopped moving in the
    middle; this asks whether it stops *early*. That is what a stale cache looks
    like from downstream: the numbers are fine, the shape is fine, the last row
    is simply three weeks older than the panel it is being joined to, and the
    join silently drops or forward-fills the difference.

    Rerunning a script is not recomputing its inputs, which is why this is worth
    an assertion rather than a habit.
    """
    if not inputs:
        return Check("M6", "mechanistic", "input freshness", NA, blocking=False,
                     detail="no inputs supplied")
    last, cadence = {}, {}
    for name, arr in inputs.items():
        a = np.asarray(arr)
        if a.size == 0 or a.dtype == bool:
            continue
        f = np.isfinite(a.astype(float)) if a.dtype.kind in "fc" else np.ones(a.shape, bool)
        rows = f.any(axis=1) if f.ndim > 1 else f
        idx = np.flatnonzero(rows)
        if idx.size:
            last[name] = int(idx[-1])
            # An input published every k steps legitimately ends short of the
            # panel, and a forecast ends h further short still, because its
            # label period would run past the data. The allowance is therefore
            # k + h - 1, not a fixed lag -- reading only the cadence still
            # flagged three correct pipelines as dead feeds, and reading
            # neither flagged all of them.
            cadence[name] = int(np.median(np.diff(idx))) if idx.size > 2 else 1
    if len(last) < 2:
        return Check("M6", "mechanistic", "input freshness", NA, blocking=False,
                     detail="fewer than two datable inputs")
    ref = max(last.values())
    allow = {n: max(max_lag, cadence.get(n, 1) + horizon - 1) for n in last}
    stale = {n: ref - v for n, v in last.items() if ref - v > allow[n]}
    fmt = (lambda i: str(dates[i])) if dates is not None and len(dates) > ref else (lambda i: f"row {i}")
    if not stale:
        return Check("M6", "mechanistic", "input freshness", PASS,
                     statistic=0.0, threshold=float(max_lag),
                     detail=f"every input reaches {fmt(ref)}, allowing each its own "
                            f"publication cadence plus the {horizon}-step horizon",
                     evidence={"last_row": last, "cadence": cadence})
    worst = max(stale.values())
    listed = ", ".join(f"{n} ends {fmt(last[n])}, {d} steps short of the {allow[n]} its "
                       f"cadence ({cadence.get(n, 1)}) and horizon allow" for n, d in
                       sorted(stale.items(), key=lambda kv: -kv[1])[:3])
    return Check("M6", "mechanistic", "input freshness", FAIL,
                 statistic=float(worst), threshold=float(max_lag),
                 detail=(f"inputs do not reach the same date as the rest of the panel ({fmt(ref)}): "
                         f"{listed}. A cache that was never rebuilt looks exactly like this"),
                 evidence={"last_row": last, "stale": stale, "cadence": cadence,
                           "allowed": allow})


def s9_label_persistence(fwd: np.ndarray, mask: np.ndarray, horizon: int = 1,
                         min_n: int = 30, margin: float = 0.5) -> Check:
    """S9 -- check the label actually changes between observations.

    A label that barely moves makes the IC series strongly autocorrelated, and
    every ratio built on it -- ICIR, the daily t-statistic -- inflates. The
    symptom is a factor library where most things clear ICIR 0.5, which reads as
    a rich hunting ground and is really a broken ruler.

    Overlapping forward windows explain some persistence by construction, so the
    comparison is against that floor rather than against zero. What is being
    flagged is a label that persists far past what its own horizon accounts for.
    """
    from scipy.stats import rankdata

    f = np.asarray(fwd, float)
    rows = []
    for t in range(f.shape[0] - 1):
        m = mask[t] & mask[t + 1] & np.isfinite(f[t]) & np.isfinite(f[t + 1])
        if m.sum() < min_n:
            continue
        a = rankdata(f[t, m]).astype(float)
        b = rankdata(f[t + 1, m]).astype(float)
        a -= a.mean()
        b -= b.mean()
        d = np.sqrt((a @ a) * (b @ b))
        if d > 0:
            rows.append((a @ b) / d)
    if len(rows) < 20:
        return Check("S9", "statistical", "label persistence", INCONCLUSIVE,
                     detail="too few adjacent cross-sections to measure persistence")
    persistence = float(np.median(rows))
    floor = (horizon - 1) / max(horizon, 1)
    # The margin has to be a share of the headroom left above the floor, not a
    # constant added to it. Persistence is bounded at 1, so a fixed margin makes
    # the check unable to fire at all once the horizon is long enough -- at h=5
    # the threshold came out at 1.2, which nothing can exceed.
    thr = floor + margin * (1.0 - floor)
    ok = persistence <= thr
    return Check("S9", "statistical", "label persistence", PASS if ok else FAIL,
                 statistic=persistence, threshold=thr,
                 detail=(f"the label reorders between observations (rank persistence "
                         f"{persistence:.2f}, overlap alone explains {floor:.2f})"
                         if ok else
                         f"the label barely changes between observations (rank persistence "
                         f"{persistence:.2f} against the {floor:.2f} its horizon explains). "
                         "ICIR and any daily t-statistic on it are inflated -- rank on IC "
                         "magnitude and out-of-sample agreement instead"),
                 evidence={"persistence": persistence, "overlap_floor": floor,
                           "n_pairs": len(rows)})


def m7_event_integrity(triggers: np.ndarray, fwd: np.ndarray, mask: np.ndarray,
                       n_draws: int = 200, seed: int = 0, required_pct: float = 95.0,
                       min_events: int = 10) -> Check:
    """M7 -- an event detector against random events of the same count.

    Two things go wrong with event studies, and the first hides the second. A
    detector whose trigger list was pruned using the outcome reports a hit rate
    near one, which reads as a remarkable result and is a description of the
    pruning. And a detector that fires often enough will catch the same moves a
    coin would, so the comparison has to hold the firing rate fixed.

    Both are settled by drawing random events at the strategy's own per-period
    count and asking whether the real ones landed better.
    """
    trig = np.asarray(triggers, bool) & mask & np.isfinite(fwd)
    n_per = trig.sum(axis=1)
    total = int(trig.sum())
    if total < min_events:
        return Check("M7", "mechanistic", "event integrity", INCONCLUSIVE,
                     detail=f"only {total} events")
    vals = fwd[trig]
    real = float(np.mean(vals))
    hit = float(np.mean(np.sign(vals) == np.sign(real)))
    if total >= 30 and hit >= 0.98:
        return Check("M7", "mechanistic", "event integrity", FAIL,
                     statistic=hit * 100, threshold=98.0,
                     detail=(f"{hit:.1%} of {total} events resolve the same way. A detector that "
                             "essentially never misses was selected using the outcome it claims "
                             "to predict -- rebuild the trigger list from information available "
                             "when each trigger fired"),
                     evidence={"n_events": total, "hit_rate": hit, "mean": real})

    g = np.random.default_rng(seed)
    draws = np.empty(n_draws)
    for d in range(n_draws):
        acc, n = 0.0, 0
        for t in range(trig.shape[0]):
            k = int(n_per[t])
            if k <= 0:
                continue
            pool = np.flatnonzero(mask[t] & np.isfinite(fwd[t]))
            if pool.size < k:
                continue
            pick = g.choice(pool, size=k, replace=False)
            acc += float(fwd[t, pick].sum())
            n += k
        draws[d] = acc / n if n else np.nan
    pct = percentile_of(real, draws)
    ok = pct >= required_pct
    return Check("M7", "mechanistic", "event integrity", PASS if ok else FAIL,
                 statistic=pct, threshold=required_pct,
                 detail=(f"{total} events at the {pct:.0f}th pct of random events of the same "
                         f"per-period count (hit rate {hit:.0%})"
                         if ok else
                         f"{total} events land at only the {pct:.0f}th pct of random events fired "
                         f"at the same rate: the detector is choosing when to fire, not what "
                         "will happen"),
                 evidence={"n_events": total, "percentile": pct, "hit_rate": hit,
                           "real": real, "null_mean": float(np.nanmean(draws))})


def m8_threshold_or_slope(signal: np.ndarray, fwd: np.ndarray, mask: np.ndarray,
                          n_buckets: int = 10, min_n: int = 30) -> Check:
    """M8 -- advisory: is this a threshold, being modelled as a slope?

    An effect that lives entirely in one extreme bucket is not a gradient, and
    fitting a continuous modulation to it spreads a few dozen informative
    observations across every observation there is. The result is a real effect
    that measures as nothing -- and the diagnosis reads as "the signal does not
    work" when it is really "the signal was used as the wrong shape".

    Advisory, because it changes how a signal should be used rather than whether
    it is real.
    """
    sig, f = np.asarray(signal, float), np.asarray(fwd, float)
    sums = np.zeros(n_buckets)
    counts = np.zeros(n_buckets)
    for t in range(sig.shape[0]):
        m = mask[t] & np.isfinite(sig[t]) & np.isfinite(f[t])
        n = int(m.sum())
        if n < min_n:
            continue
        idx = np.flatnonzero(m)
        order = np.argsort(np.argsort(sig[t, idx]))
        b = np.minimum((order * n_buckets) // n, n_buckets - 1)
        for k in range(n_buckets):
            sel = idx[b == k]
            if sel.size:
                sums[k] += float(f[t, sel].sum())
                counts[k] += sel.size
    if (counts == 0).any():
        return Check("M8", "mechanistic", "threshold or slope", INCONCLUSIVE, blocking=False,
                     detail="not every bucket was populated")
    means = sums / counts
    spread = float(means[-1] - means[0])
    if abs(spread) < 1e-12:
        return Check("M8", "mechanistic", "threshold or slope", INCONCLUSIVE, blocking=False,
                     detail="no spread across buckets")
    mid = means[1:-1]
    mid_spread = float(mid.max() - mid.min())
    edge_share = 1.0 - abs(mid_spread / spread)
    concentrated = edge_share >= 0.6
    return Check("M8", "mechanistic", "threshold or slope",
                 FAIL if concentrated else PASS, blocking=False,
                 statistic=edge_share, threshold=0.6,
                 detail=(f"the effect is graded across buckets (the middle {n_buckets - 2} span "
                         f"{abs(mid_spread / spread):.0%} of the extreme spread)"
                         if not concentrated else
                         f"{edge_share:.0%} of the effect sits in the two extreme buckets and the "
                         f"middle {n_buckets - 2} are flat. This is a threshold, not a gradient -- "
                         "a continuous modulation will spread a few dozen informative "
                         "observations across every observation there is and measure nothing"),
                 evidence={"bucket_means": [float(v) for v in means],
                           "edge_share": edge_share, "spread": spread})


def m9_cross_sectional_independence(signal: np.ndarray, fwd: np.ndarray, mask: np.ndarray,
                                    min_n: int = 20, max_common: float = 0.5) -> Check:
    """M9 -- advisory: how much of the cross-section is just the market moving.

    A panel of a thousand names looks like a thousand observations a day and can
    be worth one. When the dates a signal fires on are dates the whole market
    moves together, the per-trade numbers are large, the effective sample is the
    number of *days*, and every standard error computed across names is
    overstated by the square root of the redundancy.

    Reported rather than enforced, and the reason matters: a genuinely
    market-level effect produces no cross-sectional IC at all, so M0 rejects it
    first. What this adds is the explanation -- whether the money in a claim
    that did survive is the names or the day.

    A first attempt compared the IC at a small random subsample against the IC
    at full breadth, on the theory that redundant observations would not
    degrade. That was wrong: rank IC is a correlation, scale-free in the number
    of names, so it does not degrade for *any* clean signal and the check would
    have fired on all of them.
    """
    r = np.asarray(fwd, float)
    shares = []
    for t in range(r.shape[0]):
        m = mask[t] & np.isfinite(r[t])
        if m.sum() < min_n:
            continue
        v = r[t, m]
        denom = float(np.mean(v ** 2))
        if denom > 0:
            shares.append(float(np.mean(v) ** 2 / denom))
    if len(shares) < 20:
        return Check("M9", "mechanistic", "cross-sectional independence", INCONCLUSIVE,
                     blocking=False, detail="too few usable cross-sections")
    common = float(np.median(shares))
    n_eff_note = ""
    ok = common <= max_common
    if not ok:
        n_eff_note = (" -- treat the day as the observation, not the name, and divide every "
                      "cross-sectional standard error accordingly")
    return Check("M9", "mechanistic", "cross-sectional independence",
                 PASS if ok else FAIL, blocking=False,
                 statistic=common, threshold=max_common,
                 detail=(f"the market accounts for {common:.0%} of the median cross-section's "
                         f"return variation, so the names carry information of their own"
                         if ok else
                         f"the market accounts for {common:.0%} of the median cross-section's "
                         f"return variation{n_eff_note}"),
                 evidence={"common_variance_share": common, "n_dates": len(shares)})


def m10_stationarity(signal: np.ndarray, fwd: np.ndarray, mask: np.ndarray,
                     retrained: Optional[bool] = None, horizon: int = 1,
                     min_n: int = 30, n_blocks: int = 4,
                     decay_t: float = 3.0, reversal_t: float = 2.0) -> Check:
    """M10 -- does the relationship stay where it was fitted?

    The seductive form is a representation that validated well and was therefore
    frozen for reuse. What frozen means in practice is that the study measured a
    relationship over a window and then assumed it was a property of the market
    rather than of the window. When it is the latter, the frozen artefact decays
    and can reverse sign, while the same architecture retrained on a rolling
    window keeps working -- at which point the retraining is the mechanism and
    the architecture is not, and the paper is about the wrong thing.

    Two statistics, split first half against second:

      decay     the first half's IC exceeds the second's by more than noise
      reversal  the second half's IC is significantly of the opposite sign

    What the verdict is depends on a property the panel cannot show, so it has
    to be declared. A moving relationship is a defect in a frozen pipeline and
    the reason a rolling one exists, so:

      retrained=False   drift is blocking -- the artefact being shipped is stale
      retrained=True    drift is advisory, and reported as load-bearing: the
                        number belongs to the refit, and a frozen deployment of
                        the same architecture is a different product that has
                        not been measured here
      retrained=None    undeclared. Drift downgrades to inconclusive rather than
                        passing, because which of the two above applies is
                        exactly what was not said.
    """
    ic = rank_ic(np.asarray(signal, float), np.asarray(fwd, float), mask, min_n=min_n)
    idx = np.flatnonzero(np.isfinite(ic))
    if idx.size < max(4 * n_blocks, 40):
        return Check("M10", "mechanistic", "stationarity of the relationship", INCONCLUSIVE,
                     blocking=False,
                     detail=f"only {idx.size} usable cross-sections -- too few to split in time")

    def _block(rows: np.ndarray) -> tuple:
        v = ic[rows]
        v = v[np.isfinite(v)]
        if v.size < 2:
            return np.nan, np.nan, 0
        se = float(v.std(ddof=1) / np.sqrt(v.size))
        return float(v.mean()), se, int(v.size)

    halves = np.array_split(idx, 2)
    m1, se1, n1 = _block(halves[0])
    m2, se2, n2 = _block(halves[1])
    blocks = [_block(b) for b in np.array_split(idx, n_blocks)]
    profile = ", ".join(f"{m:+.4f}" for m, _, _ in blocks)

    se_d = float(np.sqrt(se1 ** 2 + se2 ** 2))
    t_d = (m1 - m2) / se_d if se_d > 0 else np.nan
    t_2 = m2 / se2 if se2 > 0 else np.nan
    reversed_ = (np.isfinite(t_2) and m1 > 0 and m2 < 0 and t_2 <= -reversal_t)
    decayed = (np.isfinite(t_d) and t_d >= decay_t and m1 > 0
               and m2 < 0.5 * m1)
    ev = {"ic_first_half": m1, "ic_second_half": m2, "t_difference": t_d,
          "t_second_half": t_2, "block_profile": [m for m, _, _ in blocks],
          "n_first": n1, "n_second": n2, "retrained": retrained}

    if not (reversed_ or decayed):
        return Check("M10", "mechanistic", "stationarity of the relationship", PASS,
                     statistic=t_d, threshold=decay_t, evidence=ev,
                     detail=(f"IC {m1:+.4f} -> {m2:+.4f} across halves (t={t_d:.2f}); "
                             f"by quarter {profile} -- no drift beyond noise"))

    what = ("reverses sign" if reversed_ else "decays")
    body = (f"IC {m1:+.4f} -> {m2:+.4f} across halves (difference t={t_d:.2f}, "
            f"second half t={t_2:.2f}); by quarter {profile} -- the relationship {what}")
    if retrained is False:
        return Check("M10", "mechanistic", "stationarity of the relationship", FAIL,
                     statistic=t_d, threshold=decay_t, evidence=ev,
                     detail=body + ". The pipeline is declared frozen, so this is the "
                                   "number the frozen artefact will keep producing")
    if retrained is True:
        return Check("M10", "mechanistic", "stationarity of the relationship", FAIL,
                     blocking=False, statistic=t_d, threshold=decay_t, evidence=ev,
                     detail=body + ". The pipeline refits, so this is not a defect -- but the "
                                   "refitting is then load-bearing and belongs in the claim, and "
                                   "a frozen version of the same construction is a different "
                                   "product that has not been measured")
    return Check("M10", "mechanistic", "stationarity of the relationship", INCONCLUSIVE,
                 statistic=t_d, threshold=decay_t, evidence=ev,
                 detail=body + ". Whether that is a defect depends on whether the production "
                               "pipeline refits, which was not declared -- pass "
                               "`retrained=True/False`")


def m11_frequency_transfer(signal: np.ndarray, ret: np.ndarray, mask: np.ndarray,
                           horizon: int = 1, strides: Sequence[int] = (2, 4),
                           claimed_strides: Sequence[int] = (),
                           min_n: int = 30, reversal_t: float = 2.0,
                           decay_ratio: float = 0.33) -> Check:
    """M11 -- the same construction, a different frequency, a different answer.

    Frequency gets treated as a hyper-parameter: a pattern is established at
    whatever cadence the data happened to arrive at, and the conclusion is then
    carried to another one because the construction is unchanged. It is not a
    hyper-parameter. A shape signal that works daily can backfire at one minute,
    a rule that pays weekly can be eaten by costs daily, and a signal read on a
    five-minute grid can be a different signal on a daily one. Each cadence is a
    separate question with a separate answer.

    So measure it rather than assume it. The panel already contains the coarser
    frequencies: hold the same signal `k` times longer and rebalance `k` times
    less often, which is what trading it at the coarser cadence means.

    The profile is always reported. It becomes a rejection only where the claim
    says it should hold: pass the strides the claim covers as `claimed_strides`
    and a sign reversal or a collapse at one of them is blocking. Without that
    the check stays advisory -- a signal that reverses at 4x has said something
    true about itself, and it is only a defect if someone claimed otherwise.
    """
    from .stats import forward_returns as _fwd

    sig = np.asarray(signal, float)
    r = np.asarray(ret, float)
    rows = []
    base = None
    for k in (1,) + tuple(int(s) for s in strides if int(s) > 1):
        f = _fwd(r, horizon * k)
        v = rank_ic(sig[::k], f[::k], mask[::k], min_n=min_n)
        v = v[np.isfinite(v)]
        if v.size < 8:
            rows.append((k, np.nan, np.nan, int(v.size)))
            continue
        m = float(v.mean())
        se = float(v.std(ddof=1) / np.sqrt(v.size))
        t = m / se if se > 0 else np.nan
        rows.append((k, m, t, int(v.size)))
        if k == 1:
            base = m
    if base is None or not np.isfinite(base):
        return Check("M11", "mechanistic", "transfer across frequency", INCONCLUSIVE,
                     blocking=False, detail="the native frequency itself is not measurable here")

    profile = "; ".join(
        (f"{k}x: IC {m:+.4f} (t={t:.2f}, n={n})" if np.isfinite(m) else f"{k}x: not measurable")
        for k, m, t, n in rows)
    reversals = [k for k, m, t, _ in rows
                 if k > 1 and np.isfinite(t) and np.sign(m) != np.sign(base) and abs(t) >= reversal_t]
    collapses = [k for k, m, t, _ in rows
                 if k > 1 and np.isfinite(m) and np.sign(m) == np.sign(base)
                 and abs(m) < decay_ratio * abs(base)]
    ev = {"native_ic": base, "profile": {k: m for k, m, _, _ in rows},
          "t": {k: t for k, _, t, _ in rows}, "reversals": reversals,
          "collapses": collapses, "claimed_strides": list(claimed_strides)}
    claimed = {int(s) for s in claimed_strides}
    broken = sorted(claimed & (set(reversals) | set(collapses)))

    if broken:
        kinds = ", ".join(f"{k}x ({'reverses' if k in reversals else 'collapses'})" for k in broken)
        return Check("M11", "mechanistic", "transfer across frequency", FAIL,
                     statistic=base, evidence=ev,
                     detail=(f"the claim is asserted at {sorted(claimed)} but does not hold there: "
                             f"{kinds}. {profile}"))
    if not (reversals or collapses):
        return Check("M11", "mechanistic", "transfer across frequency", PASS, blocking=False,
                     statistic=base, evidence=ev,
                     detail=f"same sign and comparable size at every frequency tested -- {profile}")
    note = []
    if reversals:
        note.append(f"sign reverses at {reversals}x")
    if collapses:
        note.append(f"loses more than {1 - decay_ratio:.0%} of its size at {collapses}x")
    return Check("M11", "mechanistic", "transfer across frequency", FAIL, blocking=False,
                 statistic=base, evidence=ev,
                 detail=(f"{'; '.join(note)} -- {profile}. Not blocking, because the claim did not "
                         f"say it held there; state the frequency the claim is about, and do not "
                         f"carry the conclusion to another one without re-measuring"))


def s10_identification(fit_from_start: Callable[[Dict[str, float]], Dict[str, float]],
                       starts: Sequence[Dict[str, float]],
                       follow_tol: float = 0.5) -> Check:
    """S10 -- is the parameter estimated, or is it where the optimiser stopped?

    A calibration reports a parameter surface with quantiles and a story about
    what moves it. If the objective is solved for two or more parameters against
    one observation per contract, its solution set is a curve rather than a
    point, and what gets reported is a fact about the starting value and the
    search rather than about the data. The published tell is quantiles that
    collapse onto the same constant -- worse, the same constant in two unrelated
    subsamples -- but by then the table exists and looks like a result.

    The probe is mechanical and settles it: run the identical fit from several
    different starting points. A parameter whose solution follows its own start
    is not estimated. Usually some parameters are identified and others are
    along for the ride, so this reports per parameter rather than as one verdict.

    `follow_tol` is the share of the starting spread that may survive into the
    solutions. Zero would be a perfectly identified parameter; 1.0 would be one
    that never moved from its start at all.
    """
    if len(starts) < 3:
        return Check("S10", "statistical", "identification", INCONCLUSIVE, blocking=False,
                     detail="fewer than three starting points -- nothing to compare")
    sols = [fit_from_start(dict(s)) for s in starts]
    names = sorted(set().union(*[set(s) for s in sols]) & set().union(*[set(s) for s in starts]))
    if not names:
        return Check("S10", "statistical", "identification", INCONCLUSIVE, blocking=False,
                     detail="the fit returned no parameter that was also a starting value")
    rows, unidentified = {}, []
    for n in names:
        s0 = np.array([float(s[n]) for s in starts], float)
        s1 = np.array([float(x[n]) for x in sols], float)
        spread0 = float(s0.max() - s0.min())
        spread1 = float(s1.max() - s1.min())
        # How much of the spread in the starting points survives into the
        # solutions. A scale-free version of "the answer followed the guess".
        follow = (spread1 / spread0) if spread0 > 0 else (np.inf if spread1 > 0 else 0.0)
        rows[n] = {"start_spread": spread0, "solution_spread": spread1, "follow": follow}
        if follow > follow_tol:
            unidentified.append(n)
    body = "; ".join(f"{n}: {rows[n]['follow']:.2f} of the starting spread survives"
                     for n in names)
    if not unidentified:
        return Check("S10", "statistical", "identification", PASS,
                     threshold=follow_tol, evidence=rows,
                     detail=f"every parameter converges away from its start -- {body}")
    return Check("S10", "statistical", "identification", FAIL,
                 threshold=follow_tol, evidence=rows,
                 detail=(f"{', '.join(unidentified)} follow(s) the starting value and is not "
                         f"estimated -- {body}. Report only the parameters that moved, and "
                         f"treat any quantile or subsample story about the others as a "
                         f"description of the optimiser"))


def m12_control_integrity(controls: Sequence[np.ndarray], names: Sequence[str],
                          signal: np.ndarray, fwd: np.ndarray, mask: np.ndarray,
                          horizon: int = 1, min_n: int = 30, span: int = 4) -> Check:
    """M12 -- is the control a control, or a piece of the answer?

    Orthogonalising against a regressor that contains the target does not remove
    a confound, it manufactures one: the residual inherits the part of the label
    that was sitting inside the control, and the residual IC comes out higher
    than the raw IC rather than lower. That is the diagnostic -- residualising
    against something legitimate can only take information away.

    So two questions per control. Does it read past its own timestamp -- the same
    boundary location A2 runs on the signal, applied to the control? And does it
    predict the label better than the signal being defended does, which is either
    a leak or an admission that the control is the better signal.

    This is blocking, and it runs before the orthogonalisation rather than after,
    because a contaminated control invalidates M3 and every matched null built
    on it -- the numbers still appear, and they are all of the wrong thing.
    """
    from .pit import a2_feature_shift

    if not len(controls):
        return Check("M12", "mechanistic", "control integrity", NA, blocking=False,
                     detail="no controls supplied -- nothing to orthogonalise against, "
                            "and nothing to check")
    labels = list(names) if len(names) == len(controls) else [f"c{i}" for i in range(len(controls))]
    sig_ic = rank_ic(np.asarray(signal, float), fwd, mask, min_n=min_n)
    sig_ic = sig_ic[np.isfinite(sig_ic)]
    sig_strength = float(np.abs(sig_ic.mean())) if sig_ic.size else np.nan

    leaks, stronger, rows = [], [], {}
    for c, nm in zip(controls, labels):
        arr = np.asarray(c, float)
        a2 = a2_feature_shift(arr, fwd, mask, window_based=True, min_n=min_n, span=span)
        v = rank_ic(arr, fwd, mask, min_n=min_n)
        v = v[np.isfinite(v)]
        strength = float(np.abs(v.mean())) if v.size else np.nan
        rows[nm] = {"a2": a2.outcome, "a2_detail": a2.detail[:120], "abs_ic": strength}
        if a2.outcome == FAIL:
            leaks.append(nm)
        elif np.isfinite(strength) and np.isfinite(sig_strength) and sig_strength > 0 \
                and strength > 2.0 * sig_strength:
            stronger.append(nm)

    if leaks:
        return Check("M12", "mechanistic", "control integrity", FAIL, evidence=rows,
                     detail=(f"control(s) {', '.join(leaks)} read past their own timestamp. "
                             f"Every residual and every matched null computed against them is "
                             f"inflated rather than cleaned -- residualising on something that "
                             f"contains the label raises the residual IC instead of lowering it. "
                             f"Fix the control's timestamp before reading M1 or M3"))
    if stronger:
        return Check("M12", "mechanistic", "control integrity", INCONCLUSIVE, blocking=False,
                     evidence=rows,
                     detail=(f"control(s) {', '.join(stronger)} predict the label more than twice "
                             f"as strongly as the signal does (|IC| {sig_strength:.4f}). That is "
                             f"either a control that contains the target, or an admission that "
                             f"the control is the better signal -- say which"))
    return Check("M12", "mechanistic", "control integrity", PASS, evidence=rows,
                 detail=f"{len(labels)} control(s) are timestamped ahead of the label and none "
                        f"outpredicts the signal")
