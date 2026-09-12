"""Start a study in the shape that survives contact with production.

A research project that works is not a notebook that produced a number. It is a
directory somebody else can run, re-run six months later, and audit -- and the
difference is mostly decided in the first hour, by whether the pre-registration
was written before the first result and whether anything reads the frozen
config back.

This lays out that shape, with the referee already wired in:

    <project>/
      README.md            what this is, and what has actually been established
      _pick_py.sh          an interpreter that really imports what it needs
      run_study.sh         health check -> build -> study -> verdict, stops on failure
      config/prereg.json   written first; the study refuses to run without it
      config/frozen.json   cut points, enforced by P3 on every run
      code/_cfg.py         the single place a path appears
      code/build.py        data assembly
      code/study.py        Study construction and the gauntlet
      verify/health_check.py   data chain, conventions, config consistency
      verify/reproduce.py      re-run and assert the numbers did not move
      data/ output/ logs/ docs/

Every template here is working code rather than a placeholder, because a
skeleton that does not run is one more thing to debug before the work starts.
"""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Dict, Optional

PICK_PY = '''#!/usr/bin/env bash
# Sourced by run_*.sh. Finds an interpreter that *actually imports* what this
# project needs, and exports it as $PY.
#
# Not cosmetic: `python3` can point at a conda base with none of these
# installed, and a cron job that calls it fails silently and forever. Test the
# import, do not trust the name.
if [[ -z "${PY:-}" ]]; then
  for c in "${FALSIFIER_PY:-}" /usr/bin/python3 python3 python; do
    [[ -z "$c" ]] && continue
    command -v "$c" >/dev/null 2>&1 || continue
    "$c" -c "import numpy, pandas, scipy" >/dev/null 2>&1 && { PY="$c"; break; }
  done
fi
: "${PY:?no interpreter here can import numpy/pandas/scipy -- set FALSIFIER_PY or pip install -r requirements.txt}"
export PY
'''

RUN_SH = '''#!/usr/bin/env bash
# One command, in the order that makes the later steps mean anything.
#
#   1 build    assemble the panel
#   2 health   inputs reach today, prints are real, config still binds
#   3 study    construct the Study and run the gauntlet
#
# Build comes first because there is nothing to health-check until the panel
# exists, and the health check comes before the study because a verdict on
# inputs that failed it is a number rather than evidence.
#
# Stops at the first failure. A verdict computed on inputs that failed their own
# health check is a number, not evidence.
#
# Exit: 0 all steps passed; 1 a step failed.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/_pick_py.sh"
cd "${HERE}"

step () {
  echo "--- $1"
  shift
  "$@" || { echo "FAILED at that step; stopping."; exit 1; }
}

step "build"         "$PY" code/build.py
step "health check"  "$PY" verify/health_check.py
step "study"         "$PY" code/study.py
echo "--- done. verdict written to output/verdict.json"
'''

CFG_PY = '''"""The only place a path appears.

Every external location is read from an environment variable with a default, so
moving the project does not silently unhook it from its data -- which is what
happens when paths are scattered through twenty scripts.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
CONFIG = ROOT / "config"
LOGS = ROOT / "logs"

# --- external, read-only -----------------------------------------------------
# Override with environment variables rather than editing this file.
SHARED = Path(os.environ.get("SHARED_ROOT", "data/shared"))

PREREG = CONFIG / "prereg.json"
FROZEN = CONFIG / "frozen.json"
VERDICT = OUTPUT / "verdict.json"
PANEL = DATA / "panel.npz"

for d in (DATA, OUTPUT, LOGS):
    d.mkdir(parents=True, exist_ok=True)
'''

BUILD_PY = '''#!/usr/bin/env python3
"""Assemble the panel this study runs on.

Write the arrays the gauntlet needs and nothing else: signal, returns, a
tradable mask, dates, and whatever the nulls have to match on. Keep the
assembly here and the judging in study.py -- when they are mixed, changing how
a covariate is built quietly changes what the null controls for.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from _cfg import PANEL

# TODO: replace with the real assembly.
T, N = 500, 200
rng = np.random.default_rng(0)
ret = rng.standard_normal((T, N)) * 0.02
signal = np.full((T, N), np.nan)
signal[1:] = ret[:-1]
mask = np.ones((T, N), bool)
dates = np.arange(20200101, 20200101 + T)

np.savez_compressed(PANEL, signal=signal, ret=ret, mask=mask, dates=dates)
print(f"panel written: {signal.shape} -> {PANEL}")
'''

STUDY_PY = '''#!/usr/bin/env python3
"""Construct the Study and run the gauntlet.

The arguments that decide whether the verdict means anything are the ones that
say what the nulls must match, where the prices came from, and how many
candidates were searched. Leaving them at their defaults does not make the
study safer -- it makes the checks unable to fire.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import falsifier as F
from _cfg import FROZEN, PANEL, PREREG, VERDICT

if not PREREG.exists():
    raise SystemExit(
        f"{PREREG} does not exist. Write the pre-registration before the first result -- "
        "a criterion chosen afterwards is a description of the number."
    )
prereg = F.Prereg.load(str(PREREG))
p = np.load(PANEL, allow_pickle=True)

study = F.Study(
    claim=prereg.claim,
    signal=p["signal"], ret=p["ret"], mask=p["mask"].astype(bool), dates=p["dates"],
    horizon=prereg.horizon,
    cost_bp=prereg.cost_bp,
    n_candidates_searched=prereg.n_candidates_searched,

    # --- what the nulls must match on; this is where the verdict is decided ---
    covariates={},          # every dimension that governed which names entered
    controls=[], control_names=[],
    naive_baseline=None,    # the dumbest construction using the same information

    # --- what the checks need in order to fire at all -------------------------
    price_convention=None,      # 'mid' | 'touch' | 'trade' | 'vwap'   -> E4
    cost_components=None,       # what cost_bp contains                -> E4
    fill_convention=None,       # 'next-open' | ...                    -> P4
    external_facts=None,        # facts the pipeline never saw         -> P5
    frozen_config=None,         # recomputed cut points                -> P3
    frozen_config_path=str(FROZEN) if FROZEN.exists() else None,
    seed_metric=None,           # metric_of(seed) if anything is random -> S5
)

report = F.run(study, prereg=prereg)
print(report.render())
report.to_json(str(VERDICT))

na = [c.id for c in report.checks if c.outcome == "NA"]
if na:
    print(f"\\nchecks that did not run: {', '.join(na)}")
    print("Each is an attack that did not happen, and bounds what may be claimed.")
raise SystemExit(0 if report.outcome != "REJECTED" else 1)
'''

HEALTH_PY = '''#!/usr/bin/env python3
"""One command: are the inputs fit to compute anything on?

Runs before the study, because a verdict on inputs that failed their own health
check is a number rather than evidence. Each finding is either a pass, a
warning, or a problem; problems set the exit code.

Exemptions live in EXEMPT with a written reason. An exemption without a reason
is how a red light becomes a light nobody looks at.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import numpy as np
import falsifier as F
from _cfg import FROZEN, PANEL, PREREG

OK, WARN, BAD = [], [], []
def ok(m): OK.append(m); print(f"  ok   {m}")
def warn(m): WARN.append(m); print(f"  warn {m}")
def bad(m): BAD.append(m); print(f"  BAD  {m}")

EXEMPT = {
    # "some check": "why this specific case is known-good, confirmed on <date>",
}

print("== files ==")
for p in (PREREG, PANEL):
    (ok if p.exists() else bad)(f"{p.name} {'present' if p.exists() else 'MISSING'}")
if FROZEN.exists():
    ok("frozen config present")
else:
    warn("no frozen config -- nothing is pinned, and P3 cannot bind")

if not PANEL.exists():
    print("\\ncannot continue without the panel."); raise SystemExit(1)

z = np.load(PANEL, allow_pickle=True)
sig, ret = z["signal"], z["ret"]
mask = z["mask"].astype(bool)
dates = z["dates"] if "dates" in z.files else None

print("\\n== inputs ==")
inputs = {"signal": sig, "ret": ret}
for check in (F.s7_input_staleness(inputs, dates=dates),
              F.m6_input_freshness(inputs, dates=dates),
              F.m5_print_quality(ret, dates=dates)):
    msg = f"{check.id} {check.name}: {check.detail}"
    if check.id in EXEMPT:
        warn(f"{msg}  [exempt: {EXEMPT[check.id]}]")
    elif check.outcome == "FAIL":
        bad(msg)
    elif check.outcome in ("PASS", "NA"):
        ok(msg)
    else:
        warn(msg)

print("\\n== shapes ==")
(ok if sig.shape == ret.shape == mask.shape else bad)(
    f"signal/ret/mask shapes {sig.shape} {ret.shape} {mask.shape}")
cover = float(np.isfinite(sig).sum() / max(sig.size, 1))
(ok if cover > 0.05 else bad)(f"signal coverage {cover:.1%}")

print(f"\\n  passed {len(OK)}  warnings {len(WARN)}  problems {len(BAD)}")
for m in BAD:
    print(f"   BAD  {m}")
raise SystemExit(1 if BAD else 0)
'''

REPRODUCE_PY = '''#!/usr/bin/env python3
"""Re-run and assert the numbers did not move.

Rerunning a script is not recomputing its inputs. This rebuilds from source and
compares against the last verdict, so a cache that was never refreshed shows up
as a difference rather than as a report that quietly shows last month's answer.
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from _cfg import ROOT, VERDICT

if not VERDICT.exists():
    raise SystemExit(f"{VERDICT} does not exist -- run run_study.sh first")
before = json.loads(VERDICT.read_text(encoding="utf-8"))

subprocess.run([sys.executable, str(ROOT / "code" / "build.py")], check=True)
subprocess.run([sys.executable, str(ROOT / "code" / "study.py")], check=True)
after = json.loads(VERDICT.read_text(encoding="utf-8"))

drift = [c["id"] for b, c in zip(before["checks"], after["checks"])
         if b["outcome"] != c["outcome"] or b.get("statistic") != c.get("statistic")]
if before["outcome"] != after["outcome"] or drift:
    print(f"NOT REPRODUCIBLE: verdict {before['outcome']} -> {after['outcome']}; "
          f"checks that moved: {', '.join(drift) or 'none'}")
    raise SystemExit(1)
print(f"reproduces exactly: {len(after['checks'])} checks, verdict {after['outcome']}")
'''

README_MD = '''# {name}

<!-- One sentence: what is claimed, and what has actually been established. -->

**Status:** not yet run.

## Read this first

- `config/prereg.json` -- the criterion, written before the first result
- `output/verdict.json` -- what the referee said
- `docs/STATUS.md` -- current state on one page

## Run it

```bash
bash run_study.sh
```

That sequences: health check, build, study. It stops at the first failure,
because a verdict computed on inputs that failed their own health check is a
number rather than evidence.

## What has been established

<!-- Fill in from output/verdict.json. State the killer, not just the verdict.
     Keep the sealed period in its own section and say plainly whether it has
     been read. -->

| | |
|---|---|
| verdict | — |
| killed by | — |
| checks that did not run | — |

**Sealed period:** not opened.

## Handover

Anyone picking this up should be able to run `bash run_study.sh` and
`python verify/reproduce.py` and get the same numbers. If they cannot, that is
the first bug, ahead of anything about the signal.
'''

STATUS_MD = '''# {name} -- status

_One page. What is true right now, and what is not yet known._

## Where it stands

- Claim:
- Verdict:
- Killed by:

## What has not been attacked

<!-- Every check reporting NA. These bound what may be claimed. -->

## Sealed period

Not opened. Opening is one-way: after it is read, nothing measured on it may be
tuned, and the only honest continuation is forward.

## Next

-
'''

PREREG_JSON = {
    "claim": "state the claim in one sentence",
    "mechanism": "why this should work, in economic terms. If it cannot be stated, that is already an answer",
    "implications": [
        "what else must be true if the mechanism is real"
    ],
    "primary_metric": "rank_ic_mean",
    "threshold": 0.0,
    "null_percentile_required": 95.0,
    "cost_bp": 0.0,
    "horizon": 1,
    "train": None,
    "valid": None,
    "test": None,
    "n_candidates_searched": 1,
}

FILES: Dict[str, str] = {
    "_pick_py.sh": PICK_PY,
    "run_study.sh": RUN_SH,
    "code/_cfg.py": CFG_PY,
    "code/build.py": BUILD_PY,
    "code/study.py": STUDY_PY,
    "verify/health_check.py": HEALTH_PY,
    "verify/reproduce.py": REPRODUCE_PY,
}
EXECUTABLE = {"_pick_py.sh", "run_study.sh", "code/build.py", "code/study.py",
              "verify/health_check.py", "verify/reproduce.py"}


def create(path: str, name: Optional[str] = None, force: bool = False) -> Path:
    """Lay out a study directory. Refuses to overwrite unless ``force``."""
    root = Path(path).resolve()
    name = name or root.name
    if root.exists() and any(root.iterdir()) and not force:
        raise FileExistsError(f"{root} is not empty; pass force=True to write into it anyway")

    for d in ("code", "verify", "config", "data", "output", "logs", "docs", "charts"):
        (root / d).mkdir(parents=True, exist_ok=True)

    for rel, body in FILES.items():
        p = root / rel
        p.write_text(body, encoding="utf-8")
        if rel in EXECUTABLE:
            p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)

    (root / "README.md").write_text(README_MD.format(name=name), encoding="utf-8")
    (root / "docs" / "STATUS.md").write_text(STATUS_MD.format(name=name), encoding="utf-8")
    (root / "requirements.txt").write_text(
        "numpy>=1.22\\npandas>=1.5\\nscipy>=1.9\\nfalsifier\\n"
        "# matplotlib only if this study draws anything:\\n# falsifier[charts]\\n",
        encoding="utf-8")
    (root / ".gitignore").write_text("data/\\noutput/\\nlogs/\\n__pycache__/\\n*.py[cod]\\n", encoding="utf-8")

    pre = root / "config" / "prereg.json"
    if not pre.exists():
        pre.write_text(json.dumps(PREREG_JSON, indent=2, ensure_ascii=False), encoding="utf-8")
    return root


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m falsifier.scaffold <directory> [name]")
    out = create(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None,
                 force="--force" in sys.argv)
    print(f"study laid out at {out}")
    print("Next: fill in config/prereg.json, then code/build.py, then run "
          "`bash run_study.sh`. The pre-registration comes first -- study.py "
          "refuses to run without it.")
