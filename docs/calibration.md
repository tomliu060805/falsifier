# Calibration against real studies

Synthetic targets prove the referee is internally consistent. They cannot prove
it is right, because whoever wrote the targets also wrote the checks. The test
that carries weight is whether it reaches the same verdict a careful human
process already reached on real data, months earlier, without it.

Five studies were replayed. Each had a documented conclusion arrived at by
hand, and each was chosen for a different reason: one that had been accepted,
one rejected for look-ahead, one rejected on a null baseline, one to exercise
the two audits that had never run on anything real, and one data incident found
months after the fact.

The studies are private research and are described here by shape rather than by
name; the numbers reported are the referee's, not the strategies'. What matters
for this repository is not what those studies concluded — it is what replaying
them did to the referee.

| # | study | conclusion reached by hand | referee |
|---|---|---|---|
| 1 | intraday path features over a large equity panel, 20-day horizon | real, but no better than a random basis of the same dimension | **SURVIVES**, with `M4` independently flagging the random basis at paired t=1.38 |
| 2 | a published model reporting R²≈0.89 on daily returns | a row-order bug made the target the same-day return | **REJECTED by `A2`**, boundary located at `-1->+0` — leak depth exactly one step |
| 3 | a 12-1 momentum rotation over an ETF universe | 99.8th pct against free random books, ~21st against turnover-matched ones | **REJECTED by `SM1`** at the 31st pct, and by `SE1` for losing to buy-and-hold |
| 4 | the same rolling ridge, wired for `A0`/`A1` | the decisive audits had never run on anything real | **`A0` PASS at `0.00e+00`**, `A1` collapsing to −0.0010 on shuffled labels |
| 5 | a breadth cache that froze mid-sample, and its rebuild | found by hand, months later | **FAIL** on the incident (2419 of 2930 steps flat), **PASS** on the rebuild |

Study 2 is a replication of a published paper; the others are private. Study 5
is the only one where the check was written *from* the incident and then tested
against the original files — the frozen cache and the rebuild that replaced it
both survive, which makes it a controlled test rather than a demonstration.

`M5` (print quality) is calibrated the same way, on data rather than on a
guess: clean one-minute index data runs about **0.049%** spike-and-revert over
28,800 bars, and the incident the check was written from ran about **1%**. The
threshold sits at 0.2% — four times the headroom above real prints, five times
below the incident.

---

## What the replay changed

Replaying real studies broke the referee eight times. Every one was a case the
synthetic targets could not produce, and two of them overturned rules that had
been used by hand for years.

**1. `A2` compared two shifts instead of locating a boundary.** The original
hand rule was "peeking one step ahead must help". That holds at a one-step
horizon and fails past it: a leaked feature keeps gaining as it slides forward
for the same reason an honest one does, so the comparison clears both. `A2` now
finds where the score *first* steps up across shifts −4…+1, and reports leak
depth.

**2. `A2` scored signed IC and oriented on the published shift.** Where the
honest IC is small and noisy, orienting on its sign turns the peek-ahead score
strongly negative and hides the jump. It scores `|IC|`, and floors falls at
zero before clustering — a boundary is a rise, and a same-day feature against a
same-day label produces a spike whose far side would otherwise open a wider gap
than the real one.

**3. `A0`'s probe dates came from the calendar, not the prediction grid.** With
any stride, most dates carry no prediction, so the probes compared NaN against
NaN and `A0` reported that nothing was comparable. The decisive audit was
silently unavailable on exactly the pipelines it exists for.

**4. A weaker check could overrule a stronger one.** `A2` is a stand-in for
`A0` — it infers from behaviour what `A0` establishes by rebuilding — and a
fitted model's output has no trailing-window boundary for it to find, so it
returned INCONCLUSIVE and dragged down a claim `A0` had just cleared at
`0.00e+00`. It now yields whenever `A0` has reached a verdict, and a
disagreement between them is flagged rather than resolved in favour of the
weaker one.

**5. The overlap correction double-counted.** `ic_summary` divided the
observation count by the horizon unconditionally, reporting 83 deliberately
non-overlapping observations as "effective n=4" while setting the HAC lag to 19
on a series sampled every 20 steps. It understated the sample twentyfold and
overstated the t-statistic at the same time — not even conservative. The
cadence is recoverable from the gaps between the series' own non-NaN entries.

**6. `quantile_portfolio` rebalanced on a calendar modulus.** A signal
published every 20th step lands on dates no modulus knows about, so the book
never traded and the cost gate reported a flat zero, rejecting an otherwise
clean claim.

**7. `M6` rejected four correct pipelines as dead feeds.** A forecast published
every k steps ends k+h−1 short of the panel by construction, because its label
period would run past the data. The allowance is the cadence plus the horizon,
not a fixed lag. Same class of error as (3): a check assuming signals are dense
when real pipelines are not.

**8. `S6` protected only one direction.** The first version downgraded
rejections — absence of evidence is not evidence of absence, so a failing `M0`
on a powerless panel should not close a line of work. Right, and incomplete. A
random signal on a structureless panel then cleared `M0` at the 98th percentile
and `S4` at t=2.29, which is what noise does a couple of times in a hundred,
and the run carried on to price it. If a known effect cannot be found on a
panel, a finding cannot be trusted on it either. `S6` now stops the run in both
directions.

---

## What the test suite catches that a reviewer would not

Two design errors of mine were found by the assertion that every check claimed
to catch a failure mode must be some target's named cause of death.

**`M2` (identity null) cannot fail once `M0` has run.** Permuting asset
identity destroys a real cross-sectional edge by construction, so it passes
whenever `M0` passes and fails only where the edge was already near zero —
which `M0` reaches first. It is corroboration, and is now advisory.

**`M9` was simply wrong.** It was meant to catch a market-level effect by
shrinking the cross-section and watching the discriminating power fall. But
rank IC is a correlation, scale-free in the number of names: it does not
degrade for *any* clean signal, and the check would have fired on all of them.
It now reports how much of the cross-section is the market moving together,
advisory, with `M0` carrying the veto.

One more, from a target that failed to fail. `seed_lucky` landed exactly on
both of `S5`'s thresholds — positive fraction at 0.80 against a threshold of
0.80, reported value at the 85th percentile against 90. That was fixed by
making the target worse, not by moving the thresholds. A target that only just
fails tests the threshold; a check tuned until its target fails tests nothing.
