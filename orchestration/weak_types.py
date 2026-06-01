"""WT-1: derive frozen weak question types from the base student's eval."""

from __future__ import annotations

import json
from pathlib import Path

from filtering.matchers import exact_match
from filtering.strata import classify_question_type


def error_rate_per_type(results: list[dict]) -> dict[str, float]:
    counts: dict[str, list[int]] = {}
    for r in results:
        ans = r["answer"][0] if isinstance(r["answer"], list) else r["answer"]
        t = classify_question_type(r["question"], ans)
        correct = exact_match(str(r["pred"]), str(ans)) >= 1.0
        counts.setdefault(t, []).append(0 if correct else 1)
    return {t: sum(v) / len(v) for t, v in counts.items()}


def top_k_weak_types(results: list[dict], k: int = 3) -> list[str]:
    er = error_rate_per_type(results)
    return [t for t, _ in sorted(er.items(), key=lambda kv: kv[1], reverse=True)[:k]]


def compute_and_save(results_path: Path, out_path: Path, k: int = 3) -> list[str]:
    results = json.loads(Path(results_path).read_text())
    weak = top_k_weak_types(results, k=k)
    Path(out_path).write_text(json.dumps({"weak_types": weak}, indent=2))
    return weak
