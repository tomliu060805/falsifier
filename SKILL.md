---
name: falsify
description: "Adversarially referee a quantitative research claim -- a factor, a signal, a timing rule, a paper replication, or an idea an LLM just proposed -- along four axes: process (pre-registration, sealed test period, whether the study's own data guards can be made to fail, whether the source that proposed the idea already knew the outcome), point-in-time (truncation rebuild, label-shuffle refit, feature time-shift boundary location, label delay decay), mechanism (matched nulls, identity nulls, orthogonalisation against known factors, naive-baseline comparison, drift of the relationship under a frozen fit, transfer of the conclusion across frequency) and economics (net of turnover cost, per-trade block, whether the book could have entered the positions it holds). Returns one verdict -- SURVIVES, REJECTED or INCONCLUSIVE -- naming the check that killed the claim. Use whenever a new signal, factor, strategy or research idea needs to be tested rather than believed, including Chinese requests such as 判负、证伪、零基准、前视审计、这个因子能不能用、这个想法靠不靠谱."
---

# Falsify a research claim

Attack the claim. Do not look for reasons it might work — that is what the
person proposing it already did, and it is why the claim needs a referee.

## Rules of engagement

0a. **The idea's source has an information boundary, and truncating the price
   database does not establish it.** Training corpora, search results and tool
   output all carry later events back across it. Ask where the hypothesis came
   from; if the answer is a model or a search, `P8` is not optional, because no
   point-in-time audit downstream can see this one.
0. **Nothing measured on a panel without power is a finding, in either
   direction.** When `S6` (positive control) is flat, the run stops and the
   verdict is INCONCLUSIVE — report it as "nothing was established", never as
   "this does not work" and never as a result. If the default control
   (short-horizon reversal) is not the right known effect for this universe and
   horizon, supply one that is as `positive_control` and run again; that is the
   intended response, not an override.
1. **Never report SURVIVES for a study whose blocking checks did not run.** A
   green battery obtained by supplying nothing is the failure mode this tool
   exists to prevent. If `A0` and `A1` both read `NA`, say plainly in the
   summary that the decisive point-in-time audits did not run.
2. **Write the pre-registration before running anything.** Freeze it with
   `Prereg.freeze()`. A criterion produced after the number is a description of
   the number.
3. **Ask what decided which names entered**, and match `covariates` on every
   one of them. This single argument determines whether `M1` means anything.
   Ask the same question of every `control`: a regressor measured over a window
   that overlaps the label is not a control, and orthogonalising against it
   *raises* the residual IC instead of lowering it. `M12` runs the boundary
   locator on the controls for exactly this reason, and it is blocking, because
   `M1` and `M3` built on a contaminated control look healthy and measure
   something else.
4. **Ask how many variants were tried**, including the ones abandoned early,
   and pass the honest count as `n_candidates_searched`.
5. **Do not touch the test period.** Development happens on train and valid.
   Unsealing is a one-way door and requires a frozen specification.
8. **Report the killer, not just the verdict.** "REJECTED by M1 — a null that
   knows only size and turnover reproduces 94% of the effect" is the finding.
   "REJECTED" alone is not.

## Workflow

1. Restate the claim in one sentence, and the mechanism in one more. If the
   mechanism cannot be stated, that is already the answer — report it and stop.
2. Write down the implications: what else must be true if the mechanism is
   real? These turn a story into something falsifiable. Put them in
   `Prereg.implications` and test the ones you can.
3. Assemble `Study`: signal, returns, tradable mask, horizon, covariates,
   controls, a naive baseline, the cost in basis points, the candidate count.
   Add `seed_metric` if anything about the result was random, and
   `positive_control` if the default (short-horizon reversal on the study's own
   returns) is not a fair test of this panel's power. Declare `price_source` --
   an index or a constructed series switches on `E3`, and `tradable_ret` makes
   it decisive. **Declare the entry constraints**: `price_limit` (the daily
   limit as a simple return, `None` only for a market that has none) and
   `min_dollar_volume`. `E6` prices them one at a time, and on event-driven
   claims that line is routinely larger than the cost line -- a locked board is
   not expensive to buy, it is impossible, and nothing on the cost axis can see
   the difference. If a parameter was chosen, pass the whole sweep as
   `knob_metric`/`knob_params`; if anything was frozen, pass
   `frozen_config`/`frozen_config_path` so the freeze actually binds.
   If the study leans on a data check of its own -- a no-NaN assertion, a
   release gate, a reconciliation -- pass it as `validator` with a
   `corruptions` map that injects the defects it claims to catch. `P7` requires
   it to actually fail on them: a guard that cannot fail and a guard with
   nothing to report produce the same output on every real run.
   **If the idea came from a model, a search, or a knowledge base, say so and
   pass `ask`** with `hindsight_probes` (questions whose answers became knowable
   only after `information_boundary`) and `hindsight_controls` (questions from
   before it). `P8` is the only check that reaches a leak in *what was proposed*
   rather than in how it was computed -- and that leak makes `A0` and `A1` come
   back clean on a pipeline that was aimed by the answer.
   Say whether the production pipeline refits or ships a frozen fit
   (`retrained=True/False`): a relationship that moves is the reason a rolling
   pipeline exists and a defect in a frozen one, and `M10` cannot tell which is
   being shipped. If the conclusion is asserted at cadences other than the
   panel's own, pass them as `claimed_strides` -- otherwise `M11` reports the
   frequency profile without holding anyone to it.
   If the claim rests on a calibrated parameter, pass `fit_from_start` and three
   or more `starts`: `S10` reruns the identical fit from each and reports which
   parameters converge and which follow their own starting value.
4. **Wire the pipeline in if you can.** `A0` and `A1` are the only checks that
   settle a leak, and both are unavailable for a frozen array -- one rebuilds
   the signal from truncated history, the other refits it on destroyed labels.
   If the study is a rolling fit over a feature block, `RollingFit` supplies
   both callables; otherwise write `recompute_at(t)` by hand. A verdict with
   `A0` and `A1` reading `NA` rests on the weaker half of the battery, and the
   summary has to say so.
5. `report = falsifier.run(study, prereg=prereg)`.
5. Read the verdict, then read the checks that reported `NA` — those are the
   attacks that did not happen, and they bound what may be claimed.
6. Archive `report.to_json()` alongside the pre-registration.

## Commands

```bash
python examples/selfcheck.py           # five known cases; must print SELF-CHECK PASSED
pytest tests/ -q -m "not slow"         # unit tests
pytest tests/ -q                       # including the full gauntlet
```

Run the self-check after changing any threshold in the battery. A referee that
no longer gets the known cases right has stopped being evidence.

## Reading a verdict

- `ok` — the attack ran and the claim did not die.
- `FAIL` — the attack landed. A blocking `FAIL` rejects the claim.
- `??` (INCONCLUSIVE) — the check could not reach a judgement. Blocking and
  inconclusive means the claim was **not judged**, which is not survival.
- `--` (NA) — the check does not apply, or its prerequisite was not supplied.
  `A0`/`A1` NA means the decisive audits did not run; `S5` NA means a single
  run is being taken on faith.
- `~` — advisory: informative, but its failure has innocent explanations, so it
  carries no veto.

## Judging an addition rather than a claim

When the question is "should this factor go into the book" rather than "is this
factor real", use `run_increment` with an `Increment` (the book without the
candidate and the book with it). It reports the delta year by year and checks
two things a headline number hides: whether the gain is one year or takes the
bad years with it (`I1`), and whether adding a candidate that knows nothing
would have moved the curve about as much (`I2`).

Always pass `combined_of_seed`. Without it `I2` reads NA, and the yearly table
on its own routinely passes additions that a coin would match.

## Scope

This skill judges claims. It does not search for signals, fit models, tune
hyper-parameters or fetch data. Interpretation of any result still depends on
data quality, universe construction, timing assumptions and execution
assumptions; a verdict is research evidence, not production approval and not
investment advice.
