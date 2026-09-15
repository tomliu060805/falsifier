"""The sealed test period.

Holding out a test period only works if reading it is expensive. Here it costs
a written record: the first unseal stamps the configuration hash into a ledger,
and any later attempt with a different configuration raises. That converts
"I promised not to look" into something the machine enforces.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np


class SealError(RuntimeError):
    pass


@dataclass
class SealedSplit:
    train_end: Any
    valid_end: Any
    ledger: str = "unseal_ledger.json"

    def mask_for(self, dates: np.ndarray, segment: str, unsealed: bool = False) -> np.ndarray:
        d = np.asarray(dates)
        if segment == "train":
            return d <= self.train_end
        if segment == "valid":
            return (d > self.train_end) & (d <= self.valid_end)
        if segment == "test":
            if not unsealed:
                raise SealError(
                    "test segment is sealed. Call unseal(config, reason) once, and only "
                    "after the specification is frozen -- reading it to choose a "
                    "specification is what makes the number meaningless.")
            return d > self.valid_end
        raise ValueError(f"unknown segment {segment!r}")

    def _read(self) -> Dict[str, Any]:
        if not os.path.exists(self.ledger):
            return {}
        with open(self.ledger, encoding="utf-8") as fh:
            return json.load(fh)

    def unseal(self, config: Dict[str, Any], reason: str, prereg_id: Optional[str] = None) -> Dict[str, Any]:
        h = hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12]
        rec = self._read()
        if rec:
            if rec.get("config_hash") != h:
                raise SealError(
                    f"test was already unsealed on {rec.get('when')} with config {rec.get('config_hash')}, "
                    f"and the configuration has since changed to {h}. Re-running the test after "
                    "tuning on its result is not an out-of-sample test; open a new project or "
                    "report both numbers as one in-sample search.")
            rec["reads"] = rec.get("reads", 1) + 1
            with open(self.ledger, "w", encoding="utf-8") as fh:
                json.dump(rec, fh, indent=2, ensure_ascii=False)
            return rec
        rec = {"when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "config_hash": h, "config": config, "reason": reason,
               "prereg_id": prereg_id, "reads": 1}
        with open(self.ledger, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2, ensure_ascii=False)
        return rec

    def status(self) -> Optional[Dict[str, Any]]:
        return self._read() or None

    # ---- doing the right thing has to be one line ---------------------------

    def redact(self, arr: np.ndarray, dates: np.ndarray) -> np.ndarray:
        """A copy with the sealed rows removed rather than merely unused.

        Masking a sealed row is not the same as not having it: a full-sample
        z-score, a quantile cut point or a rolling statistic will still consume
        it. Take it out of the array and the question cannot arise.
        """
        d = np.asarray(dates)
        out = np.array(arr, copy=True)
        sealed = d > self.valid_end
        if out.ndim == 1:
            out[sealed] = False if out.dtype == bool else np.nan
        else:
            out[sealed, ...] = False if out.dtype == bool else np.nan
        return out

    def assert_clean(self, dates: np.ndarray, **arrays: np.ndarray) -> None:
        """Raise if any array carries a value inside the sealed period.

        For use anywhere, not only in a study -- the peeks in this record did
        not come from unsealing. They came from a yearly diagnostic that was
        printed without excluding the test rows, and from a panel handed to the
        referee that quietly included them. Both would have been one call away
        from being caught.
        """
        bad = _sealed_rows(self, dates, arrays)
        if bad:
            names = ", ".join(f"{k} ({v} rows)" for k, v in bad.items())
            raise SealError(
                f"these carry values inside the sealed period (after {self.valid_end}): {names}. "
                f"Nothing computed from them is out of sample. Use redact(), or slice the panel "
                f"before it reaches the computation.")


def _sealed_rows(seal: "SealedSplit", dates: np.ndarray,
                 arrays: Dict[str, np.ndarray]) -> Dict[str, int]:
    """Per array, how many sealed rows carry a value. Empty means clean."""
    d = np.asarray(dates)
    sealed = d > seal.valid_end
    out: Dict[str, int] = {}
    if not sealed.any():
        return out
    for name, a in arrays.items():
        if a is None:
            continue
        arr = np.asarray(a)
        if arr.shape[0] != d.shape[0]:
            continue                      # not aligned to the date axis
        rows = arr[sealed]
        if rows.dtype == bool:
            live = np.asarray(rows).reshape(rows.shape[0], -1).any(axis=1)
        else:
            live = np.isfinite(np.asarray(rows, float)).reshape(rows.shape[0], -1).any(axis=1)
        n = int(live.sum())
        if n:
            out[name] = n
    return out


def p9_panel_respects_seal(seal: Optional["SealedSplit"], dates: Optional[np.ndarray],
                           arrays: Dict[str, np.ndarray], horizon: int = 1,
                           unsealed: bool = False) -> "Check":
    """P9 -- does the panel that was handed over actually respect the seal?

    `P2` reports whether the seal is still intact. It does not look at the data,
    and every peek in this record got in through that gap. None of them was an
    unseal:

      a yearly alignment diagnostic was printed without excluding the test rows,
      so two years of trigger-day means were read;

      the panel handed to this referee in the first round quietly included the
      test segment. Every check ran, every number was computed correctly, and
      all of them were about a panel that was not allowed to exist.

    That is the shape to defend against -- not someone deciding to look, but a
    computation that was never told where the boundary was. So this runs before
    anything is measured and stops the run: a contaminated panel does not make
    the numbers below it wrong-looking, it makes them look ordinary and be about
    something else.

    A decision at the last rows before the boundary consumes forward returns
    that reach past it. That is reported rather than rejected -- it is a handful
    of rows with a mechanical fix, and a check that cries wolf on it stops being
    read.
    """
    from .verdict import FAIL, NA, PASS, Check

    if seal is None or dates is None:
        return Check("P9", "process", "panel respects the seal", NA, blocking=False,
                     detail=("no seal and dates supplied, so which rows this panel was built "
                             "from cannot be verified. A held-out period that nothing checks "
                             "is a promise, not a split"))
    d = np.asarray(dates)
    bad = _sealed_rows(seal, d, arrays)
    reaching = int(((d > seal.valid_end - horizon) & (d <= seal.valid_end)).sum()) if horizon else 0
    note = (f" The last {reaching} row(s) before the boundary consume forward returns that reach "
            f"past it; trim them or say so." if reaching else "")

    if bad and not unsealed:
        names = ", ".join(f"{k} ({v} rows)" for k, v in sorted(bad.items(), key=lambda kv: -kv[1]))
        return Check("P9", "process", "panel respects the seal", FAIL,
                     evidence={"sealed_rows": bad, "valid_end": str(seal.valid_end)},
                     detail=(f"the panel carries values inside the sealed period (after "
                             f"{seal.valid_end}): {names}. Nothing computed from it is out of "
                             f"sample -- and the checks below would all have run and all have "
                             f"been about the wrong panel. Slice the panel, or use "
                             f"SealedSplit.redact(), before anything reads it"))
    if bad and unsealed:
        return Check("P9", "process", "panel respects the seal", PASS, blocking=False,
                     evidence={"sealed_rows": bad, "unsealed": True},
                     detail=(f"the panel includes the sealed period and the seal has been opened "
                             f"-- this is a test-segment run and must be reported as one. It can "
                             f"happen once: tuning after seeing this number and running again is "
                             f"not an out-of-sample test.{note}"))
    return Check("P9", "process", "panel respects the seal", PASS,
                 detail=f"no values inside the sealed period (after {seal.valid_end}).{note}")
