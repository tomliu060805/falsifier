---
name: record
description: "Turn a finished piece of research -- or anything new the user brings in -- into a verdict record, put it in the prior library, and let the library re-summarise itself. Checks that the cause of death keys to the failure-mode taxonomy, and when it does not, proposes the new mode rather than forcing the record into the nearest fit; reports how the shape of the record changed and whether what a new study should check first has moved. Use after any study reaches a verdict, including one that survived, and whenever a new finding, paper, incident or correction arrives that the record does not yet know. Chinese requests: 记到判负库里、这个结果入库、更新判负库、把这条结论沉淀下来、重新归纳一下。"
---

# Record it, and let the library re-summarise itself

This is the step that makes the rest of it compound. A referee that judges and
forgets is a calculator. The library is what turns one post-mortem into
something the next study can retrieve, and it only exists if this happens every
time.

**Including the ones that survived.** A record of nothing but rejections
teaches only pessimism, and eleven of the verdicts here are survivals — they
are what tells you the machine is not simply saying no.

## When to run this

- a study reached a verdict — any verdict, `SURVIVES` included
- a conclusion you already held turned out to be wrong
- a paper, report or incident taught something the record does not know
- a project produced a trap: a data convention, an alignment rule, a silent
  failure

## Step 1 — write the claim as it was tested, not as it was hoped

One sentence, in the past tense, specific enough that retrieval can match it
later: the construction, the universe, the horizon, the cost. "动量有效" is not
a claim. "12-1 动量 ETF 轮动有选股 alpha" is.

## Step 2 — the cause of death, keyed to the taxonomy

```python
from falsifier import taxonomy as T
[m.id for m in T.MODES if "null" in m.family]
```

**If nothing fits, do not key it to the nearest thing that does.** An
approximate cause of death teaches the wrong lesson to whoever retrieves it
later. What it means is one of two things, and both are useful:

- the record is vague — go back and say what actually decided it, or
- **the taxonomy is short a mode** — write the mode first, then the record

Backfilling this library has twice opened the taxonomy back up, five modes the
first time and three the second. That is the intended direction of traffic, not
a failure.

## Step 3 — evidence is the number, not a summary

The field is called `evidence` because it holds the figure that decided it.
"零基准表现更好" is not evidence. "换手匹配后从 100 分位塌到 21.4 分位" is.

## Step 4 — the lesson is the part that transfers

The claim is about one study. The lesson has to be usable by a study that has
nothing else in common with it. Write what the next person should do
differently, not what happened.

Weak: "这个因子没有增量。"
Strong: "轮动/TopN 类策略一律先做换手匹配零基准,否则比的是手续费不是选股。"

## Step 5 — put it in and read what changed

```python
import falsifier as F
r = dict(id="...", claim="...", verdict="REJECTED", killed_by=["..."],
         evidence="...", lesson="...", applies_to=["..."],
         project="...", date="2026-09-15", source="...")
print(F.render_append(F.append_prior(r)))        # dry_run=True to rehearse
```

It refuses the record if the id collides, the schema does not fit, the cause of
death keys to nothing, or the lesson is too thin to teach. **Each refusal is
the check working.**

Then read the summary. It prints what a new study should check first, and says
so when that has changed — which it has twice: `null-random-structure-wins`,
then `signal-duplicates-existing`, now `cost-eats-it`. That number is the
library re-summarising itself, and it is the closest thing here to the machine
learning something.

## Step 6 — if it overturned something, retract that something

A library that only grows teaches its own mistakes forever. When this finding
overturns an earlier record, mark the earlier one:

```python
# in priors/verdicts.jsonl, on the record being retired:
"withdrawn": "为什么它不再成立, 以及新的证据是什么",
"superseded_by": "<新记录的 id>",
"withdrawn_date": "2026-09-15"
```

A withdrawn verdict **stays searchable** — it is exactly what you want surfaced
when you are about to re-derive it — but leaves the checklist of traps, because
a trap that turned out not to be one is not a trap.

**Both retractions here had a `lesson` that had become bad advice**, not just a
stale number. That is the case to watch for: the number being wrong costs one
study, and the lesson being wrong costs every study that retrieves it.

## Step 7 — check it did not land on top of something

```python
print(F.validate_priors() or "clean")
[x.id for x in F.load_priors() if x.project == "<项目名>"]
```

And before the next study on the same project, read that list. Two records
written on one day here carried conclusions a robustness review had overturned
a month earlier; they were caught only because someone happened to open the
right memory.

## What this is not

It is not a lab notebook. A record is one claim, one verdict, one number, one
transferable lesson. Everything else belongs in the project's own documents —
and if a record needs a paragraph to explain, the claim was not stated sharply
enough in step 1.
