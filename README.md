# falsifier

An adversarial referee for quantitative research claims.

Ideas are cheap now. A language model will hand you a hundred factor
hypotheses before lunch, and a research note will hand you a hundred more.
What does not scale is the judgement that kills the ninety-nine — and the
judgement is where the mistakes live, because every one of those ideas arrives
with a plausible story and a backtest attached.

`falsifier` is the other half. You give it a signal, the returns it claims to
forecast, and what you already knew. It runs a fixed battery of attacks along
four axes and tells you which one killed the claim.

It never certifies anything. The best available outcome is **SURVIVES**, which
means *not yet falsified* and is not a synonym for true.

```
================================================================================
VERDICT: REJECTED   |   Trailing momentum forecasts returns
================================================================================

[STATISTICAL]
  FAIL  A2     feature time-shift                         -1 (thr 0)
          label boundary at shift -1->+0, not 0->+1: the value published for t
          already contains information from after t
          [-4:0.0256, -3:0.0273, -2:0.0290, -1:0.0348, +0:0.2151]

KILLED BY: A2 (feature time-shift)
note: stopped after the point-in-time audits: a leak makes every downstream
      number meaningless, so the nulls and the cost gate were not run
```

## The four axes

| | check | what it attacks | needs |
|---|---|---|---|
| **process** | `P0` pre-registration | a criterion chosen after seeing the number | a frozen `Prereg` |
| | `P1` stated mechanism | a result with no economic story to test | — |
| | `P2` test seal | a hold-out that has been read more than once | a `SealedSplit` |
| | `P3` frozen config enforced | a frozen file nothing reads back | the recomputed values |
| | `P6` mechanism implications | a story that was never tested past the number | tested implications |
| | `P4` fill convention declared | an engine that matches on the signal's own bar | a declaration |
| | `P5` external fact check | an error two implementations share | a fact from outside |
| **statistical** | `A0` truncation rebuild | the signal used data from after its timestamp | `recompute_at` |
| | `A1` label-shuffle refit | something fitted on the full sample | `refit` |
| | `A2` feature time-shift | an off-by-one between signal and label | — |
| | `A3` label delay decay | an alignment error, or a one-step effect sold as many | — |
| | `S4` significance after search | a t-stat that ignores how many candidates you tried | candidate count |
| | `S5` seed stability | a configuration that worked once | `seed_metric` |
| | `S6` positive control | a null result from an apparatus with no power | — |
| | `S8` knob monotonicity | train rising while validation falls | a parameter sweep |
| | `S9` label persistence | a slow label inflating every ratio built on it | — |
| **mechanistic** | `S7` input staleness | a feed that stopped moving mid-series | — |
| | `M5` print quality | moves that never happened | — |
| | `M6` input freshness | a cache that ends short of the panel | — |
| | `M0` harness sanity | a broken pipeline, or nothing at all | — |
| | `M1` matched null | a repackaging of size, liquidity, turnover, breadth | covariates |
| | `M2` identity null | shape and timing masquerading as asset selection | — |
| | `M3` residual after known factors | an old factor wearing a new name | controls |
| | `M4` ~ | beats naive baseline | machinery that does not pay for itself | a naive version |
| | `M7` event integrity | a trigger list pruned with the outcome | `triggers` |
| | `M8` ~ | threshold or slope | a threshold effect fitted as a gradient | — |
| | `M9` ~ | cross-sectional independence | days counted as if they were names | — |
| **economic** | `E1` net of cost | an edge smaller than the turnover it needs | cost in bp |
| | `E2` per-trade block | a year of P&L made on four days | — |
| | `E3` execution delay | a stale print or a spread bouncing back | `tradable_ret` |
| | `E4` cost convention | the spread charged twice, or never | a declaration |
| | `E5` capacity | an edge in a window too small to use | volume |

Checks that cannot run say so — `A0` reports `NA` with *"the decisive audit did
not run"* rather than quietly passing. A battery that goes green because you
gave it nothing to work with is worse than no battery at all.

## Wiring the decisive audits

`A0` (rebuild the signal from history truncated at each date) and `A1` (refit
the pipeline on destroyed labels) are the two checks that actually settle
whether something leaked. Neither works on a frozen array of numbers: one needs
the pipeline rebuilt, the other needs it refitted. A verdict with both reading
`NA` rests on the weaker half of the battery, and says so.

Most research pipelines share one shape — standardise a feature block, fit a
rolling regression on labels whose horizon has closed, predict the next
cross-section — so wrapping that shape once makes both audits available:

```python
fit = F.RollingFit(features, ret, mask, horizon=20, fitwin=250, stride=20)
study = F.Study(..., signal=fit.predict(),
                recompute_at=fit.recompute_at, refit=fit.refit,
                probe_dates=fit.probe_dates(6))
```

The two audits do not subsume each other, which is easiest to see in what each
one misses:

| defect | `A0` | `A1` |
|---|---|---|
| scaling constant fitted on the whole sample | **FAIL** | pass |
| coefficients fitted once over all history | **FAIL** | pass |
| training window reaching the predicted date | pass | **FAIL** |

`A0` asks whether a value could have been produced on its own date, so it
catches anything computed from data that did not exist yet. `A1` asks whether
the fit can score on labels carrying no information, so it catches a training
window that overlaps the period being predicted — which `A0` cannot see,
because that window *was* available at the time. A clean pipeline passes both.

## "Should this go in" is a different question

Whether a factor is real and whether it belongs in a book are answered with
different evidence. A candidate can be perfectly real and worth nothing to a
book that already holds something correlated with it; it can raise the headline
while turning flat years into losing ones; and it can add three points that all
come from one year nobody will see again.

So the unit is the pair — the book without it and the book with it — and the
output is the delta year by year:

```python
inc = F.Increment(dates=dates, baseline=base_net, combined=with_candidate_net,
                  benchmark=bench, label_combined="+candidate")
print(F.render_yearly(inc))
report = F.run_increment(inc, combined_of_seed=add_a_random_candidate,
                         chart="charts/increment.png")
```

| | check | what it attacks |
|---|---|---|
| `I1` | incremental contribution | a gain that is one year, or that takes the bad years with it |
| `I2` | increment against a null candidate | adding anything would have moved the curve too |

`I2` is the one that decides it, and the self-check target built for it shows
why. That target's yearly table passes every reading a person would give it —
`+0.60%` annualised, six of nine years improved, losing years down from four to
three — and adding a candidate that knows nothing does about as well:

```
I1 PASS   annualised excess +2.02% -> +2.62% (+0.60%); 6/9 years improved,
          losing years 4 -> 3, largest single year is 41% of the gain
I2 FAIL   adding an uninformative candidate does about as well (57th pct)
```

Accounting is compounded throughout — excess is `(1+strategy)/(1+benchmark)-1`
chained, never a difference of arithmetic means, and never an intraday-only
segment presented as a curve.

Charts go to PNG on disk and nowhere else. Fonts are registered explicitly and a
missing CJK face raises rather than falling back, because the fallback renders
every Chinese label as an empty box and the figure still looks finished.

## Portfolio-level claims

A signal is judged by its IC; a strategy is judged by what a book that traded it
would have earned, and the two need different nulls. Shuffling a signal says
nothing about a rotation rule, because most of what a rotation rule does is
decided by how often it trades and how wide it holds.

```python
report = F.run_strategy(F.StrategyStudy(
    claim="12-1 momentum rotation beats holding the broad-market ETF",
    selection=held,            # (T, N) bool: held from t to t+1
    ret=ret, mask=investable,
    benchmark=bench, cost_bp=10.0,
))
```

| | check | what it attacks |
|---|---|---|
| `SM0` ~ | free-selection null | nothing — reported only as a contrast |
| `SM1` | turnover-matched null | a result produced by trading often, not by picking well |
| `SE1` | beats benchmark after cost | an edge that loses to buy-and-hold |
| `SE2` ~ | per-trade block | a record carried by a handful of trades |

`SM1` is the one that decides it. Random books rebalance on the same dates, hold
the same count, and retain the same number of positions from period to period,
so turnover and cost are identical and the only remaining claim is that *these*
names were the ones to hold. `SM0` is the null most studies run, and the reason
so many rotation rules look good: it churns far harder than the strategy and
loses to almost anything. Clearing it is not evidence, which is why it carries
no veto.

## Quickstart

```python
import falsifier as F

study = F.Study(
    claim="Trailing 5-day momentum forecasts the next 5 days",
    signal=signal,                      # (T, N), the value known at t
    ret=ret,                            # (T, N), one-period returns
    mask=tradable,                      # (T, N)
    horizon=5,
    covariates={"size": size, "turnover": turnover},   # what M1 matches on
    controls=[size], control_names=["size"],           # what M3 removes
    cost_bp=5.0,
    n_candidates_searched=40,           # every variant you looked at
)

report = F.run(study)
print(report.render())
report.to_json("verdict.json")
```

The two arguments that decide whether the verdict means anything are
`covariates` and `n_candidates_searched`. Match the null on every dimension
that determined which names entered the portfolio, not just the convenient
one — a null matched on size when the real selector was liquidity will
cheerfully confirm your result. And count the candidates you discarded in the
first minute; those are the ones that set the threshold.

## Why you should trust the referee

`examples/selfcheck.py` builds thirty synthetic claims whose correct verdict is
known before the machine sees them, and asserts both the verdict **and the
cause of death** — being rejected for the wrong reason teaches the wrong
lesson and is scored as a failure. Every check that carries a veto has at least
one target built to trip it; a check with no target is a check nobody has shown
to work.

```
target                                      expected      got           want   got killer    
--------------------------------------------------------------------------------------------
survivor                                  SURVIVES      SURVIVES      -      -             ok
noise                                     REJECTED      REJECTED      M0     M0            ok
leaky                                     REJECTED      REJECTED      A2     A2            ok
size_proxy                                REJECTED      REJECTED      M1     M1,M3         ok
costly                                    REJECTED      REJECTED      E1     E1            ok
increment_real                            SURVIVES      SURVIVES      -      -             ok
increment_is_noise                        REJECTED      REJECTED      I2     I2            ok
increment_one_year                        REJECTED      REJECTED      I1     I1            ok
filtered_events                           REJECTED      REJECTED      M7     M7            ok
same_bar_fill                             REJECTED      REJECTED      P4     P4            ok
external_fact_wrong                       REJECTED      REJECTED      P5     P5            ok
spread_twice                              REJECTED      REJECTED      E4     E4            ok
mechanism_falsified                       REJECTED      REJECTED      P6     P6            ok
no_capacity                               REJECTED      REJECTED      E5     E5            ok
bad_prints                                REJECTED      REJECTED      M5     M5            ok
stale_cache                               REJECTED      REJECTED      M6     M6            ok
sticky_label                              REJECTED      REJECTED      S9     S9            ok
stale_index                               REJECTED      REJECTED      A2     A2            ok
bounce                                    REJECTED      REJECTED      E3     E3            ok
overfit_knob                              REJECTED      REJECTED      S8     S8            ok
config_drifted                            REJECTED      REJECTED      P3     P3            ok
frozen_covariate                          REJECTED      REJECTED      S7     S7            ok
dead_panel                                INCONCLUSIVE  INCONCLUSIVE  -      -             ok
seed_lucky                                REJECTED      REJECTED      S5     S5            ok
clean                                     SURVIVES      SURVIVES      -      -             ok
full-sample scaling                       REJECTED      REJECTED      A0     A0            ok
training window reaches the predicted dateREJECTED      REJECTED      A1     A1            ok
one fit over the whole history            REJECTED      REJECTED      A0     A0            ok
skilled_book                              SURVIVES      SURVIVES      -      -             ok
blind_book                                REJECTED      REJECTED      SM1    SM1           ok

veto-carrying checks never seen to reject anything: P0, S4
```

`dead_panel` is the only target whose correct answer is INCONCLUSIVE. The signal
really is nothing — but so is a known effect on the same panel, so nothing was
established either way, and a referee that returns REJECTED there is claiming
evidence from an apparatus never shown able to produce any.

One target deserves a caveat stated out loud: `blind_book` is a single draw from
the very distribution `SM1` compares against, so by construction it clears the
95th percentile about one time in twenty, and a seed exists for which it
"passes". That is the check behaving correctly. The property that actually has
to hold is the *rate*, which `tests/` measures directly rather than inferring
from one book being rejected: over thirty skill-free books the matched null
places the median at the **48th percentile** — it should be the 50th — and lets
**3 of 30** through above the 95th, against a nominal 1.5. Three out of thirty
is well inside sampling error for a threshold set at one-in-twenty; what would
matter is a median far from 50, and it is not.

```bash
python examples/selfcheck.py
```

## Checking an idea against what has already been killed

The expensive part of judging a new idea is not running the checks — it is
knowing which check is going to matter. That knowledge accumulates one
post-mortem at a time and evaporates unless it is written down where something
can search it.

`falsifier.taxonomy` is 51 named ways a quantitative result turns out to be
wrong, each extracted from a study that was built, believed, and then killed.
Almost every mode names a check that catches it:

```python
>>> from falsifier import taxonomy as T
>>> T.coverage()
{'null': (8, 8), 'pit': (8, 8), 'cost': (6, 6),
 'stat': (10, 10), 'mech': (14, 16), 'proc': (3, 3)}
>>> [m.id for m in T.uncovered()]
['non-stationary-needs-retraining',
 'conclusion-does-not-transfer-across-frequency']
```

The table was briefly full, and then backfilling a few years of post-mortems
into the prior records turned up five modes it did not have — two of which
still have no check. That is the intended direction of traffic, and it is why
`uncovered()` stays published rather than being quietly closed: this is a
record of what has gone wrong so far, and the next entry arrives the way all
the others did, from a study that was built, believed, and then killed. A
full-looking table is a reason to add modes, not a reason to relax.

`validate()` is what keeps the two honest with each other. A record whose cause
of death cannot be keyed to a mode means either the record is vague or the
taxonomy is short one, and the check refuses to let either pass silently.

What keeps "covered" from being a mapping exercise is the test suite: every
check a mode points at must be some target's named cause of death, or be listed
as corroborating with a written reason. And the self-check reports, at the end
of every run, which veto-carrying checks were never actually seen to reject
anything — measured, not declared.

`falsifier.priors` searches a record of past verdicts — what was claimed, what
killed it, the number that decided it — and turns the nearest hits into a
prioritised list of traps:

```python
hits = F.search_priors("learn an adjacency matrix over stocks to forecast returns")
print(F.render_priors("...", hits))
```
```
  [REJECTED] a volatility graph network improves implied-vol forecasts
      killed by  : null-random-structure-wins
      evidence   : random adjacency scores +0.436 against the full model's +0.332

TRAPS TO CHECK FIRST:
  - a random structure does as well   [M4]
      Rerun with a random adjacency / basis / split / weight vector.
```

Retrieval is BM25 over character bigrams for CJK and word stems for ASCII — no
segmenter, no embedding service, no key. It is a prefilter: it surfaces
candidates and you judge relevance. The records are research output and do not
ship here; point `FALSIFIER_PRIORS` at your own.

## Calibrated against past verdicts

Synthetic targets prove the referee is internally consistent; they cannot prove
it is right, because whoever wrote the targets also wrote the checks. So it was
replayed against real studies whose verdicts had already been reached by hand,
months earlier, without it.

| study | conclusion reached by hand | referee |
|---|---|---|
| signature features on intraday paths | real, but no better than a random basis of the same size | **SURVIVES**, with `M4` flagging the random basis at paired t=1.38 |
| a published model with R²=0.89 | a row-order bug made the target the same-day return | **REJECTED by `A2`**, boundary located at `-1->+0` — leak depth exactly one step |
| a momentum ETF rotation | 99.8th pct against free random books, 21st against turnover-matched ones | **REJECTED by `SM1`** at the 31st pct, and by `SE1` for losing to buy-and-hold |
| the same rolling ridge, wired for `A0`/`A1` | the decisive audits had never run on anything real | **SURVIVES**, `A0` reproducing the signal exactly from truncated history |
| a breadth cache that froze in 2016, and the rebuild that replaced it | found by hand, months later | **FAIL** on the incident (2419 of 2930 steps flat), **PASS** on the rebuild |

That replay broke the referee six times, in six ways no synthetic target
produced — including one where the original project's *own* audit rule turned
out to be unsound past a one-step horizon, and one where the decisive audit had
been silently unavailable on exactly the pipelines it exists for.
[`docs/calibration.md`](docs/calibration.md) has the full account.

## Design notes

Three things that took more than one attempt, kept here because each one is a
way the obvious implementation quietly fails.

**`A2` locates the label boundary; it does not compare two shifts.** Score the
feature at shifts −4…+1 and find where the score *first* steps up. That position
is the boundary between legal and illegal information, and for an honestly
timestamped signal it must sit between shift 0 and +1. Three things this
deliberately avoids, each of which passed the synthetic targets and failed on
real data. Comparing shift 0 against shift +1 alone clears a leaked feature as
readily as an honest one, because past a one-step horizon both keep gaining as
they slide forward. Taking the *largest* step as the boundary fails too: with a
multi-step horizon the second and third label days are worth about as much as
the first, and which of them wins is noise. And scoring signed IC fails when the
honest IC is small — orienting on its sign turns the peek-ahead score strongly
negative and hides the jump — so `A2` scores `|IC|`, and floors falls at zero
before clustering, because a boundary is a rise and a same-day feature against a
same-day label produces a spike whose far side would otherwise open a wider gap
than the real one. `A2` reports leak depth, so `-2->-1` says the signal is two
steps early rather than one.

**`M1` fails on effect size as well as on percentile.** Bucketing a continuous
covariate leaves some of its ordering inside each cell, so a signal that simply
*is* the covariate still edges past its own matched null and lands at the 100th
percentile. The percentile answers "is the excess reliable"; it does not answer
"is the excess worth having". A null that knows nothing but size and volatility
and still reproduces 95% of the effect has explained the result, however tight
its error bars.

**`A3` informs, it does not veto.** A signal that genuinely lives on one step
will fail the overlap floor at a five-step horizon without anything being wrong
with it — the finding is that its multi-step IC is really a one-step effect,
which changes what may be claimed but does not make the claim false. Checks
whose failure has an innocent explanation are marked advisory (`~`) and carry
no veto.

## The check that falsifies the story without falsifying the number

`P6` is the one that had no representation at all until a few years of paper
replications were written into the prior records, where the same shape appeared
four times: a model reproduces its predictive content closely — to the decimal,
in places — while the economic channel it claims to work through has the wrong
sign, or the proxy for it carries nothing.

The number survives and the story does not, and it is the story that was
supposed to generalise to the next market and the next decade.

So the pre-registration asks for the implications — what else must be true if
the mechanism is real — and `P6` asks what came back. Declaring them and never
testing them reads INCONCLUSIVE rather than passing: the claim as stated
includes its mechanism, and an untested mechanism leaves it exactly where it
was. A weaker claim, without the story, may well survive; it is just not the
one that was made.

## The one check that can stop a rejection

Every other check here can kill a claim. `S6` is the only one that can stop a
killing from meaning anything.

A measurement is interpretable only if the apparatus could have produced a
different one. So `S6` runs a known effect — short-horizon reversal by default,
built from the return panel the study already has — through the identical panel,
mask, horizon and scoring code. If that comes back flat, the gauntlet stops and
the verdict is INCONCLUSIVE.

It stops in *both* directions, which took a wrong turn to get right. The first
version downgraded only the rejections: absence of evidence is not evidence of
absence, so a failing `M0` on a powerless panel should not close a line of work.
But the same argument runs the other way. A random signal on a structureless
panel cleared `M0` at the 98th percentile and `S4` at t=2.29 — which is exactly
what noise does a couple of times in a hundred — and the run continued to price
it. If a known effect cannot be found here, a finding is not evidence either.

The verdict is inconclusive rather than a rejection because the way out is
actionable: short-horizon reversal is only a guess at what a panel should
contain, and a study that knows better should pass its own `positive_control`.
On the A-share panel these calibrations use, the default clears at t=+8.7 (h=5)
and t=+7.2 (h=20), so it is not a formality.

Getting this wrong is expensive in the direction nobody notices: a false
rejection leaves no trace, produces no bad trade, and quietly ends the
investigation.

## Two checks for one family

A price artefact and a leak are the same observation from a shift analysis, and
separating them took two checks rather than one.

`A2` sees that the signal published for t overlaps the period it claims to
forecast. That happens when the signal peeked — and equally when the *label* is
a stale print that still contains period t, which is what an index does for
every constituent that has not traded yet. Both make the score something other
than a forecast, so `A2` rejects either way and names both readings; which one
it is decides what to fix, not whether to.

`E3` catches the half `A2` cannot see. Bid-ask bounce makes consecutive observed
returns negatively autocorrelated for reasons that have nothing to do with
forecasting, and a reversal signal picks it up mechanically — without ever
overlapping its own label. In the self-check that target clears `A2`, clears
every null, carries its costs, and then loses 99% of its edge the moment it is
priced on the series a book would actually realise:

```
bounce   A2 PASS   label boundary at shift 0->+1, as an honest window requires
         E3 FAIL   only 1% of the edge survives on the instrument a book would
                   hold (+188.49% -> +2.33%)
```

`E3` is off unless the study declares where its prices came from. That default
is deliberate: the check cannot tell an index from a tradable series, and asking
is cheaper than guessing wrong in either direction.

## Starting a study in this shape

```bash
python -m falsifier.scaffold /path/to/project "what it is"
```

Lays out a directory that runs: a pre-registration the study refuses to start
without, a health check over the inputs, a reproduce script that rebuilds from
source and asserts the numbers did not move, and a one-command runner that
sequences build, health check and gauntlet, stopping at the first failure.

Build comes first because there is nothing to check until the panel exists; the
health check comes before the study because a verdict on inputs that failed it
is a number rather than evidence.

Three details that look cosmetic and are not. `_pick_py.sh` tests the import
rather than trusting the interpreter's name, because `python3` can point at an
environment missing every dependency and a scheduled job that calls it fails
silently and forever. `verify/health_check.py` keys its exemptions by check id
and requires a written reason for each, because an exemption without one is how
a red light becomes a light nobody looks at. And on the stub panel the first run
stops at `S6` with INCONCLUSIVE — the scaffold telling you, before you have
written anything, that there is no real data in it yet.

## What this is not

It does not search for signals, fit models, or fetch data. It has no opinion
about your data vendor and no adapter for one; `Study` takes arrays. It cannot
tell you a claim is true, and a `SURVIVES` verdict on a badly specified study
means only that a badly specified study was not falsified.

## Install

```bash
pip install -e .          # numpy, pandas, scipy
```

MIT.
