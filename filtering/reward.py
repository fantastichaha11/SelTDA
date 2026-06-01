"""P2 Grounded-Learnability reward terms (pure functions, DI of scorers)."""

from __future__ import annotations

import string
from typing import Iterable

from filtering.strata import classify_question_type

_PUNCT = str.maketrans("", "", string.punctuation)


def _norm(text: str) -> str:
    return text.lower().translate(_PUNCT).strip()


def type_match(question: str, answer: str, weak_types: set[str]) -> float:
    """1.0 if the (Q,A) question-type is in the targeted weak set, else 0.0."""
    return 1.0 if classify_question_type(question, answer) in weak_types else 0.0


def repetition_penalty(question: str, seen: Iterable[str], threshold: float = 0.9) -> float:
    """1.0 if `question` token-Jaccard-overlaps any seen question above threshold."""
    q = set(_norm(question).split())
    if not q:
        return 0.0
    for s in seen:
        ss = set(_norm(s).split())
        if not ss:
            continue
        jac = len(q & ss) / len(q | ss)
        if jac >= threshold:
            return 1.0
    return 0.0
