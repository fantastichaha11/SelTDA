"""Filesystem I/O for pseudo-label JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List


def load_records(path) -> List[dict]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected JSON list at top level")
    return data


def dump_records(records: List[dict], path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(records, f)
