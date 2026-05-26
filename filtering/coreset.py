"""CS-X: greedy submodular coreset selection on embeddings."""

from __future__ import annotations

import numpy as np


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def select_coreset(
    embeddings: np.ndarray,
    budget: int,
    diversity_weight: float = 0.5,
) -> list[int]:
    n = len(embeddings)
    if budget >= n:
        return list(range(n))
    selected: list[int] = [0]
    while len(selected) < budget:
        best_j, best_gain = -1, -1.0
        for j in range(n):
            if j in selected:
                continue
            relevance = float(np.linalg.norm(embeddings[j]))
            diversity = min(_cosine_sim(embeddings[j], embeddings[s]) for s in selected)
            gain = (1 - diversity_weight) * relevance + diversity_weight * (1 - diversity)
            if gain > best_gain:
                best_gain, best_j = gain, j
        selected.append(best_j)
    return selected
