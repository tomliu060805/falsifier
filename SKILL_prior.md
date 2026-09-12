---
name: prior-check
description: "Before testing a new quantitative research idea -- a factor, signal, timing rule, paper replication, or something a language model just proposed -- search the record of claims already tested and return the traps most likely to kill this one. Retrieves comparable past verdicts by claim text and study shape, then produces a prioritised checklist of failure modes with the specific check that catches each. Use at the start of any new study, and whenever an idea arrives with a backtest attached. Chinese requests: 先查历史陷阱、这个想法以前测过吗、历史判负、有没有踩过同样的坑。"
---

# Check an idea against what has already been killed

Run this before designing the study, not after getting a number. The point is to
choose which checks matter *for this idea* while it is still cheap to change the
design.

## Use

```bash
export FALSIFIER_PRIORS=/path/to/private/priors     # a directory of .jsonl records
python -c "
import falsifier as F
hits = F.search_priors('<the idea, in one or two sentences>', k=6)
print(F.render_priors('<the idea>', hits))
"
```

## How to read the result

The retrieval is lexical, so it will surface some near-misses. Judge relevance
yourself before repeating anything back:

- **A past verdict on the same study shape** is the useful hit, even when the
  asset class differs. A turnover-matched null killing an ETF rotation applies
  to a stock rotation and to a sector rotation.
- **A `SURVIVES` hit matters as much as a `REJECTED` one.** It tells you what
  form of the idea worked, which is often a different form than the one being
  proposed -- an avoidance overlay rather than a selection signal, for example.
- **`NO CHECK YET` in the checklist is the important line.** It means the
  gauntlet does not defend against that trap and you have to do it by hand.
- **No hits is not reassurance.** It means the idea has no precedent in the
  record, so the full gauntlet applies with nothing prioritised.

## What to do with it

1. State the idea in one sentence and run the search.
2. Read the traps. For each one, decide before touching data whether this study
   is exposed to it, and if so what the study design has to include -- the
   covariates the null must match, the callable A0 needs, the candidate count.
3. Put those decisions in the `Prereg` for the study. That is the handoff: the
   prior check chooses what the pre-registration has to promise.
4. Run the gauntlet.

## Writing a record back

After a study concludes -- either way -- append one line to the priors file:

```json
{"id": "...", "claim": "one sentence, what was asserted",
 "verdict": "REJECTED|SURVIVES|PARTIAL", "killed_by": ["<taxonomy mode id>"],
 "evidence": "the number that decided it, not a summary",
 "lesson": "what transfers to the next study",
 "applies_to": ["tags describing the shape of study"],
 "project": "...", "date": "YYYY-MM-DD", "source": "..."}
```

`lesson` is the field that gets retrieved and read, so write it for a reader who
has not seen the project. `killed_by` must use ids from `falsifier.taxonomy`; if
nothing fits, the taxonomy is missing a mode and that is worth adding.

Records are research output. They stay outside any repository that might become
public.
