"""Verdict schema.

A falsifier never certifies a claim as true. The best outcome is SURVIVES:
the claim was attacked along three axes and did not die. That asymmetry is
the whole point, so it is encoded in the vocabulary rather than left to the
reader's discipline.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

AXES = ("process", "statistical", "mechanistic", "economic")

PASS = "PASS"
FAIL = "FAIL"
INCONCLUSIVE = "INCONCLUSIVE"
NA = "NA"  # check does not apply to this study design

_MARK = {PASS: "ok  ", FAIL: "FAIL", INCONCLUSIVE: "??  ", NA: "--  "}


@dataclass
class Check:
    """One falsification attempt.

    ``blocking`` is what gives a check veto power. A blocking FAIL rejects the
    claim outright; a blocking INCONCLUSIVE means the claim could not be judged
    (which is not the same as surviving, and must never be read as one).
    """

    id: str
    axis: str
    name: str
    outcome: str
    blocking: bool = True
    statistic: Optional[float] = None
    threshold: Optional[float] = None
    detail: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.axis not in AXES:
            raise ValueError(f"unknown axis {self.axis!r}, expected one of {AXES}")
        if self.outcome not in (PASS, FAIL, INCONCLUSIVE, NA):
            raise ValueError(f"unknown outcome {self.outcome!r}")


@dataclass
class Report:
    claim: str
    checks: List[Check] = field(default_factory=list)
    prereg_id: Optional[str] = None
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    notes: List[str] = field(default_factory=list)

    def add(self, check: Check) -> "Report":
        self.checks.append(check)
        return self

    @property
    def outcome(self) -> str:
        blocking = [c for c in self.checks if c.blocking]
        if any(c.outcome == FAIL for c in blocking):
            return "REJECTED"
        if any(c.outcome == INCONCLUSIVE for c in blocking):
            return "INCONCLUSIVE"
        if not any(c.outcome == PASS for c in blocking):
            return "INCONCLUSIVE"  # nothing actually attacked it
        return "SURVIVES"

    @property
    def killers(self) -> List[Check]:
        return [c for c in self.checks if c.blocking and c.outcome == FAIL]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["outcome"] = self.outcome
        return d

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        s = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=float)
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(s)
        return s

    def render(self, width: int = 96) -> str:
        rule = "=" * width
        out = [rule, f"VERDICT: {self.outcome}   |   {self.claim}", rule]
        if self.prereg_id:
            out.append(f"pre-registration: {self.prereg_id}")
        for axis in AXES:
            rows = [c for c in self.checks if c.axis == axis]
            if not rows:
                continue
            out.append("")
            out.append(f"[{axis.upper()}]")
            for c in rows:
                stat = "" if c.statistic is None else f"{c.statistic:>10.4g}"
                thr = "" if c.threshold is None else f" (thr {c.threshold:g})"
                veto = " " if c.blocking else "~"  # ~ = advisory, no veto power
                out.append(f"  {_MARK[c.outcome]}{veto} {c.id:<6} {c.name:<34}{stat}{thr}")
                if c.detail:
                    out.append(f"          {c.detail}")
        if self.killers:
            out.append("")
            out.append("KILLED BY: " + ", ".join(f"{c.id} ({c.name})" for c in self.killers))
        if self.notes:
            out.append("")
            for n in self.notes:
                out.append(f"note: {n}")
        out.append(rule)
        out.append("SURVIVES means 'not yet falsified', never 'true'. ~ marks advisory checks (no veto).")
        return "\n".join(out)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.render()
