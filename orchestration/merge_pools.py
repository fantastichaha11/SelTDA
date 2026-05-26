"""J-1: merge easy/hard synthetic pools with duplicate-sampling weights."""

from __future__ import annotations

import copy


def merge_with_weights(pools: dict[str, tuple[list[dict], int]]) -> list[dict]:
    """Expand each pool by integer weight via deep-copy."""
    merged: list[dict] = []
    for _name, (records, weight) in pools.items():
        for _ in range(max(0, weight)):
            merged.extend(copy.deepcopy(records))
    return merged
