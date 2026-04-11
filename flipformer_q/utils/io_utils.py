from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def list_checkpoints(path: Path):
    path = Path(path)
    if not path.exists():
        return []
    return sorted(path.glob("*.pt"))
