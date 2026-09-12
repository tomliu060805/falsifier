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
from typing import Any, Dict, Optional

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
