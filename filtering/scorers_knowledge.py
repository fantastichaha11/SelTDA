"""KC-1: knowledge-consistency via NLI over retrieved passages."""

from __future__ import annotations

import numpy as np


def score_knowledge_consistency(
    question: str,
    answer: str,
    passages: list[str],
    nli_model,
) -> float:
    if not passages:
        return 0.0
    hypothesis = f"Question: {question} Answer: {answer}"
    scores = []
    for p in passages:
        pred = nli_model.predict([(p, hypothesis)])
        if isinstance(pred, (list, tuple)) and pred:
            val = pred[0]
            if isinstance(val, (list, tuple, np.ndarray)):
                arr = np.asarray(val, dtype=np.float64).ravel()
                scores.append(float(arr.max()) if arr.size else 0.0)
            else:
                scores.append(float(val))
        else:
            scores.append(float(pred))
    return max(scores) if scores else 0.0
