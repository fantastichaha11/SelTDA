"""Build a JSON-serializable filter report from per-record decisions."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Tuple

import numpy as np


def build_filter_report(
    decisions: Iterable[Tuple[dict, str]],
    max_examples: int = 50,
    n_bins: int = 20,
) -> dict:
    decisions = list(decisions)
    total_in = len(decisions)
    reasons = Counter(r for _, r in decisions)
    kept = reasons.get("kept", 0)

    score_lists = {"conf": [], "itm": [], "xcons": []}
    for rec, _ in decisions:
        for k in score_lists:
            v = rec.get("scores", {}).get(k)
            if v is not None:
                score_lists[k].append(float(v))

    histograms = {}
    for k, vals in score_lists.items():
        if not vals:
            histograms[k] = []
            continue
        hist, _ = np.histogram(np.asarray(vals, dtype=np.float64), bins=n_bins, range=(0.0, 1.0))
        histograms[k] = [int(x) for x in hist.tolist()]

    rejected_examples: List[dict] = []
    if max_examples > 0:
        for rec, reason in decisions:
            if reason == "kept":
                continue
            rejected_examples.append(
                {
                    "question_id": rec.get("question_id"),
                    "question": rec.get("question"),
                    "answer": rec.get("answer"),
                    "image": rec.get("image"),
                    "scores": rec.get("scores"),
                    "reason": reason,
                }
            )
            if len(rejected_examples) >= max_examples:
                break

    return {
        "counts": {
            "total_in": total_in,
            "kept": kept,
            "dropped": {
                "conf": reasons.get("conf", 0),
                "itm": reasons.get("itm", 0),
                "xcons": reasons.get("xcons", 0),
            },
        },
        "histograms": histograms,
        "rejected_examples": rejected_examples,
    }
