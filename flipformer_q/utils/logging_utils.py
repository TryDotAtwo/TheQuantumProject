from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, Iterable


class ProgressPrinter:
    def __init__(self, name: str, total: int, every_sec: float = 2.0):
        self.name = name
        self.total = max(1, int(total))
        self.every_sec = every_sec
        self.start = time.time()
        self.last = self.start

    def update(self, current: int, extra: str = "") -> None:
        now = time.time()
        if current < self.total and now - self.last < self.every_sec:
            return
        elapsed = max(now - self.start, 1e-9)
        rate = current / elapsed
        eta = max((self.total - current) / max(rate, 1e-9), 0.0)
        pct = 100.0 * current / self.total
        msg = (
            f"[{self.name}] {current}/{self.total} | {pct:6.2f}% | "
            f"{rate:8.2f} it/s | elapsed {elapsed:8.1f}s | eta {eta:8.1f}s"
        )
        if extra:
            msg += f" | {extra}"
        print(msg, flush=True)
        self.last = now


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, row: Dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class CsvLogger:
    def __init__(self, path: Path, fieldnames: Iterable[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row: Dict) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)
