"""Pre-registration.

The criterion has to be written down before the number is seen, otherwise the
criterion is a description of the number. Freezing it produces a content hash
that the verdict carries, so a report can be checked against the question it
claimed to be answering.
"""
from __future__ import annotations

import math
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence


@dataclass
class Prereg:
    claim: str
    mechanism: str
    """Why this should work, in economic terms. If it cannot be stated, that is
    already an answer."""

    implications: List[str] = field(default_factory=list)
    """Other things that must also be true if the mechanism is real. These are
    what turn a story into something falsifiable; a mechanism that predicts
    nothing beyond the original result is not a mechanism."""

    primary_metric: str = "rank_ic_mean"
    threshold: float = 0.0
    null_percentile_required: float = 95.0
    cost_bp: float = 0.0
    horizon: int = 1
    train: Optional[List[str]] = None
    valid: Optional[List[str]] = None
    test: Optional[List[str]] = None
    n_candidates_searched: int = 1
    """Every variant you looked at, including the ones discarded in the first
    minute. This is the number that sets the multiple-testing threshold."""

    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def digest(self) -> str:
        body = {k: v for k, v in asdict(self).items() if k != "created_utc"}
        blob = json.dumps(body, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    @property
    def id(self) -> str:
        return f"prereg-{self.digest()}"

    def freeze(self, path: str) -> str:
        if os.path.exists(path):
            raise FileExistsError(
                f"{path} already exists. A pre-registration is written once; "
                "editing it after seeing results is the thing this file prevents.")
        payload = asdict(self) | {"id": self.id}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        return self.id

    @classmethod
    def load(cls, path: str) -> "Prereg":
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        stated = d.pop("id", None)
        obj = cls(**d)
        if stated and stated != obj.id:
            raise ValueError(f"{path} was edited after freezing: id {stated} != {obj.id}")
        return obj


def p3_frozen_config(recomputed: Dict[str, Any], path: str,
                     rtol: float = 0.0) -> "Check":
    """P3 -- make the frozen file actually bind.

    A config committed as frozen binds nothing unless something reads it back
    and refuses to continue when the numbers have moved. Left unread it is a
    document, and the common failure is silent: every downstream script
    recomputes its own cut points, upstream data drifts, the results change and
    the frozen file still says what it always said.
    """
    from .verdict import FAIL, INCONCLUSIVE, NA, PASS, Check

    if not path:
        return Check("P3", "process", "frozen config enforced", NA, blocking=False,
                     detail="no frozen config declared -- nothing is pinned")
    if not os.path.exists(path):
        return Check("P3", "process", "frozen config enforced", INCONCLUSIVE,
                     detail=f"{path} does not exist")
    with open(path, encoding="utf-8") as fh:
        frozen = json.load(fh)

    drift, missing = {}, []
    for k, want in frozen.items():
        if k not in recomputed:
            missing.append(k)
            continue
        got = recomputed[k]
        if isinstance(want, (int, float)) and isinstance(got, (int, float)) and not isinstance(want, bool):
            if want != got and (rtol <= 0 or abs(got - want) > rtol * max(abs(want), 1e-12)):
                drift[k] = (want, got)
        elif want != got:
            drift[k] = (want, got)

    if not drift and not missing:
        return Check("P3", "process", "frozen config enforced", PASS,
                     detail=f"all {len(frozen)} frozen values reproduce from the current pipeline",
                     evidence={"n_keys": len(frozen)})
    parts = []
    if drift:
        parts.append("; ".join(f"{k}: frozen {w!r} but recomputed {g!r}" for k, (w, g) in list(drift.items())[:4]))
    if missing:
        parts.append(f"missing from the recomputed set: {', '.join(missing[:4])}")
    return Check("P3", "process", "frozen config enforced", FAIL,
                 statistic=float(len(drift) + len(missing)), threshold=0.0,
                 detail=("the pipeline no longer reproduces what was frozen -- " + " | ".join(parts)
                         + ". Upstream drift has changed the results without changing the file"),
                 evidence={"drift": {k: list(v) for k, v in drift.items()}, "missing": missing})


FILL_CONVENTIONS = ("next-open", "next-close", "next-vwap", "same-close", "same-vwap")


def p4_fill_convention(convention: Optional[str] = None,
                       bar_includes_signal_period: Optional[bool] = None) -> "Check":
    """P4 -- make the engine's fill convention an explicit claim.

    The leak here is usually not in the study's code. A vendor backtester that
    matches at the close of the bar the signal was formed on, or whose history
    call already contains the current bar, will produce a clean-looking result
    from an ordinary script, and nothing in the script is wrong. It is caught by
    reading the engine's documentation, which is why the only mechanical form is
    to require the answer to be written down.

    NA here is not a pass. It means nobody has said which convention was used.
    """
    from .verdict import FAIL, INCONCLUSIVE, NA, PASS, Check

    if convention is None and bar_includes_signal_period is None:
        return Check("P4", "process", "fill convention declared", NA, blocking=False,
                     detail="no fill convention declared. Which bar the engine matches on, and "
                            "whether its history call already contains the current one, is the "
                            "most common leak that lives outside the study's own code")
    if convention is not None and convention not in FILL_CONVENTIONS:
        return Check("P4", "process", "fill convention declared", INCONCLUSIVE,
                     detail=f"unrecognised convention {convention!r}; expected one of "
                            f"{', '.join(FILL_CONVENTIONS)}")
    bad = []
    if convention in ("same-close", "same-vwap"):
        bad.append(f"fills at {convention}: the signal is formed on the bar it trades into, "
                   "so the fill price is already known when the decision is made")
    if bar_includes_signal_period:
        bad.append("the history call returns the current bar, so anything read from it is "
                   "information the decision could not have had")
    if bad:
        return Check("P4", "process", "fill convention declared", FAIL,
                     detail="; ".join(bad),
                     evidence={"convention": convention,
                               "bar_includes_signal_period": bar_includes_signal_period})
    return Check("P4", "process", "fill convention declared", PASS,
                 detail=f"fills at {convention}, current bar excluded from the decision",
                 evidence={"convention": convention})


def p5_external_facts(facts: Optional[List[Dict[str, Any]]] = None) -> "Check":
    """P5 -- check the pipeline against something it has never seen.

    Reproducing a pipeline value by value proves the two implementations agree.
    It cannot find an error both of them make, and a shared misunderstanding of
    what a field means survives every internal check there is. The only thing
    that catches it is a fact from outside: a day when roughly three thousand
    names should have hit the limit, a count that is public, a number someone
    can look up.

    Each fact is ``{"what", "expected", "observed"}`` with an optional ``tol``.
    """
    from .verdict import FAIL, INCONCLUSIVE, NA, PASS, Check

    if not facts:
        return Check("P5", "process", "external fact check", NA, blocking=False,
                     detail="no external fact declared. Value-by-value agreement with another "
                            "implementation cannot find an error that both of them make")
    bad = []
    for fct in facts:
        what = fct.get("what", "?")
        exp, obs = fct.get("expected"), fct.get("observed")
        tol = float(fct.get("tol", 0.0))
        if exp is None or obs is None:
            return Check("P5", "process", "external fact check", INCONCLUSIVE,
                         detail=f"{what}: expected/observed not both supplied")
        try:
            off = abs(float(obs) - float(exp))
            ok = off <= (tol if tol > 0 else 0.0) or (tol > 0 and off <= tol * abs(float(exp)))
        except (TypeError, ValueError):
            ok = obs == exp
            off = float("nan")
        if not ok:
            bad.append(f"{what}: expected {exp!r}, pipeline gives {obs!r}")
    if bad:
        return Check("P5", "process", "external fact check", FAIL,
                     statistic=float(len(bad)), threshold=0.0,
                     detail="the pipeline disagrees with a fact it never saw -- " + "; ".join(bad[:3]),
                     evidence={"failed": bad})
    return Check("P5", "process", "external fact check", PASS,
                 detail=f"{len(facts)} external fact(s) reproduce",
                 evidence={"n_facts": len(facts)})


def p6_mechanism_implications(prereg: Optional["Prereg"] = None,
                              results: Optional[Dict[str, str]] = None) -> "Check":
    """P6 -- test what else must be true, and record what came back.

    A model can reproduce its predictive content closely -- to the decimal, in
    places -- while the economic channel it claims to work through does not hold
    at all: the proposed interaction has the wrong sign, or the proxy for it
    carries nothing. The number survives and the story does not, and it is the
    story that was supposed to generalise to the next market and the next decade.

    The implications written into the pre-registration are what make the story
    falsifiable. Declaring them and never testing them leaves the claim exactly
    where it was, which is why that reads INCONCLUSIVE here rather than passing.

    ``results`` maps each implication to "held", "failed" or "untested".
    """
    from .verdict import FAIL, INCONCLUSIVE, NA, PASS, Check

    imps = list(prereg.implications) if prereg and prereg.implications else []
    if not imps:
        return Check("P6", "process", "mechanism implications", NA, blocking=False,
                     detail="no implications declared. A mechanism that predicts nothing "
                            "beyond the original result is not a mechanism, and nothing "
                            "here can be tested")
    if not results:
        return Check("P6", "process", "mechanism implications", INCONCLUSIVE,
                     statistic=0.0, threshold=float(len(imps)),
                     detail=f"{len(imps)} implication(s) declared and none tested. They are "
                            "what turn the story into something falsifiable; untested, the "
                            "claim rests on the number alone")
    held = [i for i in imps if results.get(i) == "held"]
    failed = [i for i in imps if results.get(i) == "failed"]
    untested = [i for i in imps if i not in held and i not in failed]
    ev = {"held": held, "failed": failed, "untested": untested}
    if failed:
        return Check("P6", "process", "mechanism implications", FAIL,
                     statistic=float(len(failed)), threshold=0.0,
                     detail=(f"{len(failed)} of {len(imps)} implications do not hold: "
                             + "; ".join(failed[:2])
                             + ". The result may still be real, but the mechanism claimed "
                               "for it is not the one producing it -- and a weaker claim, "
                               "without the story, is what survives"),
                     evidence=ev)
    if untested:
        return Check("P6", "process", "mechanism implications", INCONCLUSIVE,
                     statistic=float(len(held)), threshold=float(len(imps)),
                     detail=f"{len(held)}/{len(imps)} implications held, {len(untested)} untested",
                     evidence=ev)
    return Check("P6", "process", "mechanism implications", PASS,
                 statistic=float(len(held)), threshold=float(len(imps)),
                 detail=f"all {len(imps)} implications of the stated mechanism hold",
                 evidence=ev)


def p7_validator_control(validator: Optional[Callable[[Any], Any]] = None,
                         corruptions: Optional[Dict[str, Callable[[Any], Any]]] = None,
                         data: Any = None) -> "Check":
    """P7 -- can the guard be made to fail?

    The same question S6 asks of a panel, asked of a check. S6 exists because a
    null result from an apparatus with no demonstrated power is not evidence of
    absence; P7 exists because a green light from a guard with no demonstrated
    power is not evidence of correctness, and that one is worse -- a panel with
    no power at least announces itself by finding nothing, while a guard that
    cannot fire announces itself by agreeing with you.

    Both of the cases this was written from had passed on every run since the
    day they were written:

      a no-NaN assertion built on ``~isfinite`` running against a nullable
      dtype, where NA passes straight through the comparison and is then
      skipped by the sum, so it can only ever see ``inf``;

      a release gate comparing ``old.notna() & new.notna()``, which excludes
      exactly the rows where a value appeared or vanished between releases --
      the rows it was written to find.

    Neither is a subtle bug. Both are invisible from the outside, because a
    guard that cannot fail and a guard with nothing to report produce the same
    output, and there is no run of the pipeline that distinguishes them. Only
    injecting the defect does.

    ``validator(data)`` returns something truthy when the data is acceptable.
    Each entry of ``corruptions`` maps a name to a function that introduces one
    defect the validator claims to catch. Raising counts as detection -- failing
    loudly is failing. Passing does not.
    """
    from .verdict import FAIL, INCONCLUSIVE, NA, PASS, Check

    if validator is None or not corruptions:
        return Check("P7", "process", "validator control", NA, blocking=False,
                     detail="no validator and injected defect supplied -- any data check this "
                            "study relies on has not been shown capable of failing")
    try:
        clean = validator(data)
    except Exception as exc:
        return Check("P7", "process", "validator control", INCONCLUSIVE,
                     detail=f"the validator raised on data believed clean ({exc!r}); whether it "
                            f"can detect an injected defect cannot be read until that is fixed")
    if not clean:
        return Check("P7", "process", "validator control", INCONCLUSIVE,
                     detail="the validator already rejects the data believed clean, so a "
                            "rejection of the corrupted data would say nothing -- fix the "
                            "baseline first")

    missed, caught = [], []
    for name, corrupt in corruptions.items():
        try:
            verdict = validator(corrupt(data))
        except Exception:
            caught.append(name)          # failing loudly is failing
            continue
        (missed if verdict else caught).append(name)

    ev = {"caught": caught, "missed": missed, "n": len(corruptions)}
    if missed:
        return Check("P7", "process", "validator control", FAIL, evidence=ev,
                     detail=(f"the validator passes data corrupted by: {', '.join(missed)}. "
                             f"It cannot fail on the defect it exists to catch, so its green "
                             f"is not evidence about the data -- it is evidence about itself. "
                             f"({len(caught)}/{len(corruptions)} defects detected)"))
    return Check("P7", "process", "validator control", PASS, evidence=ev,
                 detail=f"the validator fails on every injected defect "
                        f"({len(caught)}/{len(corruptions)}): {', '.join(caught)}")


def _p8_matches(response: str, probe: Dict[str, Any]) -> bool:
    """Did the answer actually land, as opposed to merely being produced?"""
    import re as _re

    accept = probe.get("accept")
    if callable(accept):
        try:
            return bool(accept(response))
        except Exception:
            return False
    want = probe.get("answer")
    text = (response or "").strip()
    if not text:
        return False
    if isinstance(want, (int, float)) and not isinstance(want, bool):
        tol = float(probe.get("tol", 0.0))
        nums = [float(x) for x in _re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))]
        if not nums:
            return False
        w = float(want)
        # Relative tolerance when one is given as a fraction, absolute otherwise.
        lim = tol if tol >= 1 else abs(w) * tol
        return any(abs(v - w) <= lim for v in nums)
    return str(want).strip().lower() in text.lower()


def p8_hindsight_control(ask: Optional[Callable[[str], str]] = None,
                         after_boundary: Optional[Sequence[Dict[str, Any]]] = None,
                         before_boundary: Optional[Sequence[Dict[str, Any]]] = None,
                         boundary: str = "", alpha: float = 0.05) -> "Check":
    """P8 -- does the source of the idea know things it should not?

    A negative control, and the third of a family. S6 asks whether the panel can
    detect an effect it is known to contain; P7 asks whether a guard can be made
    to fail on a defect it claims to catch; P8 asks whether the thing that
    proposed the hypothesis can answer a question from after the date it claims
    to be reasoning from. The first two should succeed. This one should fail.

    It exists because a study can be point-in-time clean in every line of its
    code and still be contaminated before any code runs. Truncating the price
    database at a date does not establish that the system has never seen what
    happened next: training corpora, search results, retrieved documents and
    tool output all carry later events back across the boundary. When the
    hypothesis itself was proposed by something that already knew the answer,
    A0 and A1 audit a pipeline that was pointed in the right direction for the
    wrong reason, and they will both come back clean.

    Two sets of probes, and the second is what makes the first readable:

      `after_boundary`   questions whose answers became knowable only after
                         `boundary`. Getting them right is the finding.
      `before_boundary`  questions the source ought to be able to answer. If it
                         cannot answer these either, then failing to answer the
                         future says nothing about its boundary and everything
                         about its willingness to answer -- the same trap S6
                         exists to close, in the other direction.

    Each probe is ``{"question", "answer"}`` plus optionally ``tol`` for a
    numeric answer, ``accept`` for a custom matcher, and ``chance`` -- the
    probability of getting it right by guessing, which must be declared
    honestly. A yes/no probe at ``chance=0.5`` proves almost nothing on its own
    and the arithmetic will say so; an exact closing price does not need many.
    """
    from .verdict import FAIL, INCONCLUSIVE, NA, PASS, Check

    if ask is None or not after_boundary:
        return Check("P8", "process", "hindsight control", NA, blocking=False,
                     detail="no idea source and post-boundary probes supplied -- if this "
                            "hypothesis came from a model, a search or a knowledge base, "
                            "nothing here establishes that it did not already know the answer")

    def _run(probes):
        hit, asked = [], []
        for pr in probes:
            q = pr.get("question", "")
            try:
                resp = ask(q)
            except Exception:
                resp = ""
            asked.append(q)
            if _p8_matches(resp, pr):
                hit.append(q)
        return hit, asked

    # Power first: a source that answers nothing is not a source with a boundary.
    if before_boundary:
        ctrl_hit, ctrl_asked = _run(before_boundary)
        if len(ctrl_hit) * 2 < len(ctrl_asked):
            return Check("P8", "process", "hindsight control", INCONCLUSIVE,
                         evidence={"control_hit": len(ctrl_hit), "control_n": len(ctrl_asked)},
                         detail=(f"the source answered only {len(ctrl_hit)}/{len(ctrl_asked)} "
                                 f"questions from *before* {boundary or 'the boundary'}, so its "
                                 f"silence on the ones after it says nothing about what it knows. "
                                 f"Ask questions it should be able to answer, or ask differently"))
    else:
        ctrl_hit, ctrl_asked = [], []

    hit, asked = _run(after_boundary)
    k, n = len(hit), len(asked)
    chances = [float(p.get("chance", 0.0)) for p in after_boundary]
    c = sum(chances) / n if n else 0.0
    # P(X >= k) under the declared guessing rate. With chance 0 declared and any
    # hit at all this is zero, which is the honest reading of an exact answer to
    # something unguessable -- and the reason `chance` has to be declared.
    p_value = sum(math.comb(n, i) * (c ** i) * ((1 - c) ** (n - i))
                  for i in range(k, n + 1)) if n else 1.0
    ev = {"hit": hit, "n": n, "k": k, "chance": c, "p_value": p_value,
          "control_hit": len(ctrl_hit), "control_n": len(ctrl_asked), "boundary": boundary}

    if k and p_value <= alpha:
        return Check("P8", "process", "hindsight control", FAIL,
                     statistic=p_value, threshold=alpha, evidence=ev,
                     detail=(f"the source answered {k}/{n} questions from after "
                             f"{boundary or 'its declared boundary'} (p={p_value:.4f} under a "
                             f"declared guessing rate of {c:.2f}): {'; '.join(hit[:3])}. "
                             f"The boundary is not where it is claimed to be, so the hypothesis "
                             f"cannot be treated as having been formed without the outcome -- "
                             f"and no point-in-time audit downstream can repair that, because "
                             f"the leak is in what was proposed, not in how it was computed"))
    if k:
        return Check("P8", "process", "hindsight control", INCONCLUSIVE, blocking=False,
                     statistic=p_value, threshold=alpha, evidence=ev,
                     detail=(f"{k}/{n} post-boundary probes were answered, but at a declared "
                             f"guessing rate of {c:.2f} that is what chance produces "
                             f"(p={p_value:.2f}). Use probes that cannot be guessed"))
    return Check("P8", "process", "hindsight control", PASS, evidence=ev,
                 detail=(f"none of {n} questions from after {boundary or 'the boundary'} were "
                         f"answered"
                         + (f", while {len(ctrl_hit)}/{len(ctrl_asked)} from before it were"
                            if ctrl_asked else
                            " -- but no control probes were supplied, so this has not been shown "
                            "to be a boundary rather than a refusal to answer")))
