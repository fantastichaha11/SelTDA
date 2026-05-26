"""Skill-gap report from per-sample eval results (IT-1)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from filtering.strata import classify_question_type


@dataclass
class SkillGapReport:
    per_type_accuracy: dict[str, float]
    per_type_error_rate: dict[str, float]
    weak_types: list[str]


def compute_skill_gap_from_results(results_path: Path) -> SkillGapReport:
    rows = json.loads(results_path.read_text())
    stats: dict[str, list[bool]] = {}
    for r in rows:
        t = classify_question_type(r["question"], r.get("answer", ""))
        stats.setdefault(t, []).append(bool(r.get("correct", False)))
    acc = {t: sum(v) / len(v) for t, v in stats.items() if v}
    err = {t: 1.0 - acc[t] for t in acc}
    weak = sorted(err, key=err.get, reverse=True)[:3]
    return SkillGapReport(acc, err, weak)
