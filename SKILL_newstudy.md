---
name: new-study
description: "Lay out a new quantitative research project in a shape that survives contact with production, with the referee wired in from the first hour. Creates the directory, the pre-registration the study refuses to run without, a health check over the inputs, a reproduce script, and a one-command runner that sequences build, health check and gauntlet. Use when starting any new factor, signal, strategy, replication or data study -- before the first number, not after. Chinese requests: 新开一个研究项目、建项目骨架、标准管线、这个想法要开始做了。"
---

# Start a study

Run this before the first number exists. Most of what decides whether a project
can be re-run, audited, or handed over six months later is settled in the first
hour, by two things: whether the pre-registration was written before any result,
and whether anything reads the frozen config back.

## Create

```bash
python -m falsifier.scaffold /path/to/project "what it is"
```

Lays out:

```
code/_cfg.py        the only place a path appears; externals via env vars
code/build.py       assemble the panel and nothing else
code/study.py       Study construction and the gauntlet
verify/health_check.py   inputs: staleness, freshness, print quality, shapes
verify/reproduce.py      rebuild from source and assert the numbers did not move
config/prereg.json  written first; study.py refuses to run without it
run_study.sh        build -> health check -> study, stops on first failure
```

## The order, and why

`run_study.sh` runs build, then the health check, then the study. Build first
because there is nothing to check until the panel exists; the health check
before the study because a verdict on inputs that failed it is a number rather
than evidence.

## What to fill in, in order

1. **`config/prereg.json`.** The claim in one sentence, the mechanism in
   economic terms, and what else must be true if the mechanism is real. If the
   mechanism cannot be stated, that is already an answer -- record it and stop.
   Count every variant you intend to try, including the ones you will abandon
   in the first minute, as `n_candidates_searched`.
2. **`code/build.py`.** Assembly only. Keep it out of `study.py`: when the two
   are mixed, changing how a covariate is built quietly changes what the null
   controls for.
3. **`code/study.py`.** The arguments that decide whether the verdict means
   anything are the ones the template leaves blank on purpose -- `covariates`,
   `price_convention`, `fill_convention`, `n_candidates_searched`,
   `external_facts`. Leaving them at their defaults does not make the study
   safer; it makes the checks unable to fire, and the report will say so.

## Patterns worth keeping

`_pick_py.sh` tests the import rather than trusting the interpreter's name,
because `python3` can point at an environment with none of the dependencies and
a cron job that calls it fails silently and forever.

`verify/health_check.py` carries an `EXEMPT` map keyed by check id, and every
entry needs a written reason. An exemption without one is how a red light
becomes a light nobody looks at.

If the study will run on a schedule, poll for the data rather than firing at a
fixed time, skip steps that are already done, and require the source file's
mtime to have settled before reading it. A fixed-time job that runs once and
never retries is the shape that silently stops producing for a month.

## First run

On the stub panel the gauntlet will stop at `S6` and report INCONCLUSIVE,
because the stub is noise and a known effect is flat on it too. That is the
scaffold working: it says, on the first run, that there is no real data in it
yet.
