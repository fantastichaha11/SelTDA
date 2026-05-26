"""TS-1: question-type strata and per-stratum quantile thresholds."""

from __future__ import annotations

import re

from filtering.gates import thresholds_from_quantile

EK_PATTERNS = [
    r"\bwhat kind of\b",
    r"\bwhat type of\b",
    r"\bwhat brand\b",
    r"\bwhat sport\b",
    r"\bwhat country\b",
]


def classify_question_type(question: str, answer: str) -> str:
    q = question.lower().strip()
    a = answer.lower().strip()
    if a in {"yes", "no"}:
        return "yes_no"
    if re.search(r"\bhow many\b", q):
        return "how_many"
    if re.search(r"\bwhat color\b", q):
        return "color"
    for pat in EK_PATTERNS:
        if re.search(pat, q):
            return "external_knowledge"
    return "visual_reasoning"


def assign_types(records: list[dict]) -> None:
    for r in records:
        ans = r["answer"][0] if isinstance(r["answer"], list) else r["answer"]
        r["question_type"] = classify_question_type(r["question"], ans)


def thresholds_per_stratum(
    records: list[dict],
    gate: str,
    keep_top: float,
    min_stratum_size: int = 50,
    global_fallback: float | None = None,
) -> dict[str, float]:
    assign_types(records)
    global_vals = [r["scores"][gate] for r in records if gate in r.get("scores", {})]
    global_tau = thresholds_from_quantile(global_vals, keep_top)
    by_type: dict[str, list[float]] = {}
    for r in records:
        if gate not in r.get("scores", {}):
            continue
        by_type.setdefault(r["question_type"], []).append(r["scores"][gate])
    out: dict[str, float] = {}
    for t, vals in by_type.items():
        if len(vals) < min_stratum_size and global_fallback is not None:
            out[t] = global_fallback
        else:
            out[t] = thresholds_from_quantile(vals, keep_top)
    if not out and global_vals:
        out["visual_reasoning"] = global_tau
    return out
