#!/usr/bin/env bash
# Run before every push. Nothing in this repository may name a private path, a
# private project, or a research conclusion that belongs in the private record.
#
# The list is deliberately blunt: a false alarm costs one look, and a miss is
# not recoverable once the commit is public.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# The denylist itself names private paths and private projects, so it lives in
# `.public-guard.conf`, which is git-ignored. Publishing the list would leak
# exactly what the list exists to protect. Copy `.public-guard.conf.example` and
# fill it in on each machine.
#
# A missing config is a hard error, never a pass: a guard that reports "ok"
# because it was never configured is the one failure this script cannot afford.
CONF="$HERE/.public-guard.conf"
if [[ ! -f "$CONF" ]]; then
  echo "BAD  $CONF is missing -- the guard cannot run unconfigured."
  echo "     cp .public-guard.conf.example .public-guard.conf  and fill it in."
  echo "NOT SAFE TO PUSH"
  exit 1
fi
# shellcheck source=/dev/null
source "$CONF"
if [[ -z "${FORBIDDEN:-}" || -z "${PROJECTS:-}" ]]; then
  echo "BAD  $CONF defines no FORBIDDEN/PROJECTS patterns -- refusing to run."
  echo "NOT SAFE TO PUSH"
  exit 1
fi
bad=0

scan () {
  local what="$1" pat="$2"
  local hits
  hits=$(grep -rInE "$pat" . --exclude-dir=.git --exclude-dir=__pycache__ \
         --exclude-dir='*.egg-info' --exclude=verify_public.sh \
         --exclude='.public-guard.conf' --exclude='.public-guard.conf.example' 2>/dev/null)
  if [[ -n "$hits" ]]; then
    echo "BAD  $what:"; echo "$hits" | sed 's/^/     /'; bad=1
  else
    echo "ok   no $what"
  fi
}

scan "private paths"        "$FORBIDDEN"
scan "private project names" "$PROJECTS"

# Commit history is published too, so it gets the same scan.
if [[ -d .git ]]; then
  h=$(git log --format='%H %s%n%b' 2>/dev/null | grep -InE "$FORBIDDEN|$PROJECTS" || true)
  if [[ -n "$h" ]]; then
    echo "BAD  commit messages mention private names:"; echo "$h" | sed 's/^/     /'; bad=1
  else
    echo "ok   commit messages clean"
  fi
fi

# The priors themselves must never be here, under any name. Detect them by what
# they contain rather than by filename: a check keyed to "*verdicts*.jsonl"
# reported clean on a file called priors_test.jsonl, and only the path and
# project-name scans caught it. A record set about public strategies only would
# have passed all three.
recs=$(grep -rlE '"killed_by"' . --include='*.json' --include='*.jsonl'        --exclude-dir=.git 2>/dev/null || true)
if [[ -n "$recs" ]] || [[ -d priors ]]; then
  echo "BAD  prior records present (research output, stays private):"
  echo "$recs" | sed 's/^/     /'; bad=1
else
  echo "ok   no prior records"
fi

[[ $bad -eq 0 ]] && echo "PUBLIC-SAFE" || echo "NOT SAFE TO PUSH"
exit $bad
