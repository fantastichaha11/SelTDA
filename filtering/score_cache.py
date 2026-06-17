"""Persist and reload per-gate filter scores for score-once / filter-many workflows."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CACHE_VERSION = 1

# Gates whose scores are absolute (safe to cache per record independently).
CACHEABLE_GATES = frozenset({"itm", "vqascore", "xcons", "lp", "kcons"})

# conf is recomputed each run (cheap; pool min-max depends on full distribution).
ALWAYS_RECOMPUTE_GATES = frozenset({"conf"})


def record_key(record: dict) -> str:
    """Stable key for aligning scores across filter runs."""
    qid = record.get("question_id")
    if qid is not None:
        return str(qid)
    return f"{record.get('image', '')}\0{record.get('question', '')}"


def cache_file_path(cache_dir: Path, input_path: Path, gate: str) -> Path:
    stem = Path(input_path).stem
    return cache_dir / f"{stem}__{gate}.json"


def load_gate_cache(
    cache_dir: Path | None,
    input_path: Path,
    gate: str,
    *,
    expected_n: int | None = None,
) -> dict[str, Any] | None:
    if cache_dir is None:
        return None
    path = cache_file_path(cache_dir, input_path, gate)
    if not path.is_file():
        return None
    try:
        with open(path, "r") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read score cache %s: %s", path, exc)
        return None

    if payload.get("version") != CACHE_VERSION:
        logger.warning("Ignoring score cache with unsupported version: %s", path)
        return None
    if payload.get("gate") != gate:
        logger.warning("Ignoring score cache with gate mismatch: %s", path)
        return None
    if str(payload.get("input")) != str(input_path):
        logger.warning(
            "Score cache input path differs (%s vs %s); using cache anyway",
            payload.get("input"),
            input_path,
        )
    if expected_n is not None and payload.get("n_records") != expected_n:
        logger.warning(
            "Score cache record count mismatch (%s vs %s); ignoring %s",
            payload.get("n_records"),
            expected_n,
            path,
        )
        return None

    scores = payload.get("scores")
    if not isinstance(scores, dict):
        logger.warning("Invalid score cache payload (missing scores): %s", path)
        return None

    logger.info("[cache hit] %s (%d scores)", path, len(scores))
    return payload


def save_gate_cache(
    cache_dir: Path | None,
    input_path: Path,
    gate: str,
    records: list[dict],
    *,
    extras: dict[str, dict[str, Any]] | None = None,
) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_file_path(cache_dir, input_path, gate)

    scores: dict[str, float] = {}
    for record in records:
        gate_scores = record.get("scores") or {}
        if gate not in gate_scores:
            continue
        scores[record_key(record)] = float(gate_scores[gate])

    payload: dict[str, Any] = {
        "version": CACHE_VERSION,
        "input": str(input_path),
        "n_records": len(records),
        "gate": gate,
        "scores": scores,
    }
    if extras:
        payload["extras"] = extras

    with open(path, "w") as f:
        json.dump(payload, f)
    logger.info("[cache save] %s (%d scores)", path, len(scores))


def apply_gate_cache(
    records: list[dict],
    gate: str,
    payload: dict[str, Any],
) -> int:
    """Attach cached scores (and optional extras) to records. Returns apply count."""
    scores = payload.get("scores") or {}
    extras = payload.get("extras") or {}
    applied = 0
    for record in records:
        key = record_key(record)
        if key not in scores:
            continue
        if record.get("scores") is None:
            record["scores"] = {}
        record["scores"][gate] = float(scores[key])
        extra = extras.get(key)
        if isinstance(extra, dict):
            for field, value in extra.items():
                record[field] = value
        applied += 1
    return applied


def records_missing_gate(records: list[dict], gate: str) -> list[dict]:
    return [r for r in records if gate not in (r.get("scores") or {})]


def merge_input_scores(records: list[dict]) -> int:
    """Count records that already carry a scores dict from the input JSON."""
    return sum(1 for r in records if r.get("scores"))
