"""Minimal shared experiment logger.

Writes one CSV row per epoch (fold, epoch, split, macro_auc, per-label AUCs,
loss, runtime) to `<log_dir>/<run_name>.csv`. This is deliberately low-tech —
per the team playbook ("track every experiment"), the point is that all four
of us log to the same simple format so results are comparable, not that it's
fancy. If the team later adopts Weights & Biases, swap this module's body but
keep the same call sites (log_epoch(...)) so training scripts don't change.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(self, log_dir: str | Path, run_name: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / f"{run_name}.csv"
        self._fieldnames: list[str] | None = None
        self._start_time = time.time()

    def log_epoch(self, row: dict[str, Any]) -> None:
        row = {"elapsed_sec": round(time.time() - self._start_time, 1), **row}
        is_new_file = not self.path.exists()
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if is_new_file:
                writer.writeheader()
            # Any key not in the original header is dropped rather than
            # crashing the run — logging should never be why training fails.
            writer.writerow({k: row.get(k, "") for k in self._fieldnames})

    def print_epoch(self, prefix: str, **kwargs: Any) -> None:
        parts = [f"{k}={v}" for k, v in kwargs.items()]
        print(f"[{prefix}] " + " | ".join(parts), flush=True)
