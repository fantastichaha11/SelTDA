"""Pure-function matchers for cross-consistency scoring."""

from __future__ import annotations

import string
from typing import Protocol

import numpy as np

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize(text: str) -> str:
    return text.lower().translate(_PUNCT_TABLE).strip()


def exact_match(pred: str, ref: str) -> float:
    """Lowercase + strip punct + exact string compare. Returns 1.0 / 0.0."""
    return 1.0 if _normalize(pred) == _normalize(ref) else 0.0


class SbertLike(Protocol):
    def encode(self, texts, convert_to_numpy: bool = True): ...


def sbert_match(pred: str, ref: str, model: SbertLike) -> float:
    """Cosine similarity of SBERT embeddings; rescaled to [0, 1]."""
    embs = model.encode([pred, ref], convert_to_numpy=True)
    a, b = embs[0], embs[1]
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    cos = float(np.dot(a, b) / (na * nb))
    return max(0.0, min(1.0, (cos + 1.0) / 2.0))


def max_match(pred: str, ref: str, sbert_model: SbertLike) -> float:
    """max(exact, sbert) — see spec §3.3."""
    e = exact_match(pred, ref)
    if e >= 1.0:
        return 1.0
    return max(e, sbert_match(pred, ref, sbert_model))
