"""A searchable record of claims that were already tested, and how they died.

The expensive part of judging a new idea is not running the checks -- it is
knowing which check is going to matter. That knowledge accumulates one
post-mortem at a time, and it evaporates unless it is written down in a form
something can search. This module is that form.

A prior is one past verdict: what was claimed, what killed it, the number that
decided it, and the tags that say what kind of idea it generalises to. Given a
new idea, `search` returns the nearest past verdicts and `checklist` turns them
into the specific traps to check for first.

The records themselves are research output and do not ship with this package.
Point `FALSIFIER_PRIORS` at a directory of `.jsonl` files, or pass a path.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .taxonomy import BY_ID, Mode

_CJK = re.compile(r"[一-鿿]+")
_WORD = re.compile(r"[a-z0-9][a-z0-9_+\-.]*")


def tokenize(text: str) -> List[str]:
    """Character bigrams for CJK, word stems for ASCII.

    Bigrams rather than a segmenter: no dependency, no dictionary to fall out
    of date, and for retrieval over a few hundred short records the loss
    against proper segmentation is not worth the install.
    """
    text = (text or "").lower()
    toks: List[str] = _WORD.findall(text)
    for run in _CJK.findall(text):
        toks.extend(run[i:i + 2] for i in range(len(run) - 1))
        if len(run) == 1:
            toks.append(run)
    return toks


@dataclass
class Prior:
    id: str
    claim: str
    """What was asserted, in one sentence."""
    verdict: str
    """REJECTED / SURVIVES / PARTIAL."""
    killed_by: List[str] = field(default_factory=list)
    """Failure-mode ids from `taxonomy`. Empty for a claim that survived."""
    evidence: str = ""
    """The number that decided it. Not a summary -- the actual figure."""
    lesson: str = ""
    """What transfers to the next study. This is the part worth retrieving."""
    applies_to: List[str] = field(default_factory=list)
    """Tags describing the shape of study this generalises to."""
    project: str = ""
    date: str = ""
    source: str = ""

    def text(self) -> str:
        return " ".join([self.claim, self.lesson, self.evidence,
                         " ".join(self.applies_to), " ".join(self.killed_by),
                         self.project])

    def modes(self) -> List[Mode]:
        return [BY_ID[m] for m in self.killed_by if m in BY_ID]


def load(path: Optional[str] = None) -> List[Prior]:
    """Load priors from a .jsonl file or a directory of them."""
    p = path or os.environ.get("FALSIFIER_PRIORS", "priors")
    root = Path(p)
    files = sorted(root.glob("*.jsonl")) if root.is_dir() else ([root] if root.exists() else [])
    out: List[Prior] = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("//"):
                out.append(Prior(**json.loads(line)))
    return out


class Index:
    """BM25 over the prior records."""

    def __init__(self, priors: Sequence[Prior], k1: float = 1.4, b: float = 0.7):
        self.priors = list(priors)
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(p.text())) for p in self.priors]
        self.lens = [sum(d.values()) or 1 for d in self.docs]
        self.avg = sum(self.lens) / max(1, len(self.lens))
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = max(1, len(self.docs))
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, k: int = 8, min_score: float = 1.0) -> List[Tuple[float, Prior]]:
        q = tokenize(query)
        if not q:
            return []
        scored: List[Tuple[float, Prior]] = []
        for doc, ln, prior in zip(self.docs, self.lens, self.priors):
            s = 0.0
            for t in set(q):
                f = doc.get(t, 0)
                if not f:
                    continue
                s += self.idf.get(t, 0.0) * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * ln / self.avg))
            if s >= min_score:
                scored.append((s, prior))
        scored.sort(key=lambda x: -x[0])
        return scored[:k]


def search(query: str, priors: Optional[Sequence[Prior]] = None, k: int = 8,
           path: Optional[str] = None) -> List[Tuple[float, Prior]]:
    return Index(priors if priors is not None else load(path)).search(query, k=k)


def checklist(hits: Iterable[Tuple[float, Prior]]) -> List[Mode]:
    """The union of failure modes that killed the retrieved claims.

    Ordered by how often they appear: what has killed the most things like this
    is what to look at first.
    """
    counts: Counter = Counter()
    for _, prior in hits:
        counts.update(prior.killed_by)
    return [BY_ID[mid] for mid, _ in counts.most_common() if mid in BY_ID]


def render(query: str, hits: Sequence[Tuple[float, Prior]], width: int = 96) -> str:
    rule = "=" * width
    out = [rule, f"PRIOR CHECK: {query}", rule]
    if not hits:
        out += ["", "No comparable past verdict found. That is not reassurance -- it means",
                "this idea has no precedent in the record, so run the full gauntlet.", rule]
        return "\n".join(out)
    out.append("")
    out.append(f"{len(hits)} comparable claim(s) already tested:")
    for score, p in hits:
        out.append("")
        mark = {"REJECTED": "REJECTED", "SURVIVES": "SURVIVED", "PARTIAL": "PARTIAL "}.get(p.verdict, p.verdict)
        out.append(f"  [{mark}] {p.claim}")
        out.append(f"      {p.project}{(' / ' + p.date) if p.date else ''}   (relevance {score:.1f})")
        if p.killed_by:
            out.append(f"      killed by  : {', '.join(p.killed_by)}")
        if p.evidence:
            out.append(f"      evidence   : {p.evidence}")
        if p.lesson:
            out.append(f"      lesson     : {p.lesson}")
    modes = checklist(hits)
    if modes:
        out += ["", rule, "TRAPS TO CHECK FIRST (most frequent among the above):", ""]
        for m in modes[:8]:
            caught = ", ".join(m.caught_by) if m.caught_by else "NO CHECK YET -- do this by hand"
            out.append(f"  - {m.name}   [{caught}]")
            out.append(f"      {m.probe or m.why}")
    out += ["", rule,
            "A near-miss in this list is a warning, not a verdict. Run the gauntlet anyway;",
            "these are the checks most likely to be the ones that matter."]
    return "\n".join(out)
