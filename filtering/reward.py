"""P2 Grounded-Learnability reward terms (pure functions, DI of scorers)."""

from __future__ import annotations

import string
from typing import Callable, Iterable, Protocol

from filtering.matchers import max_match
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


class StudentProbLike(Protocol):
    def answer_prob(self, image, question: str) -> float: ...


def learnability(image, question: str, student: StudentProbLike) -> float:
    """1 - max_prob of the frozen base student on (image, question). Higher = harder."""
    p = float(student.answer_prob(image, question))
    p = min(1.0, max(0.0, p))
    return 1.0 - p


def lp_flip(image, question: str, answerer, corrupt: Callable, sbert) -> float:
    """1.0 if the frozen answerer's answer CHANGES when the image is corrupted."""
    a_full = answerer.answer_question(image, question)
    a_corr = answerer.answer_question(corrupt(image), question)
    same = max_match(a_full, a_corr, sbert_model=sbert)
    return 1.0 - float(same)


def grounding(image, question: str, answer: str, answerer, corrupt: Callable, sbert) -> float:
    """XCONS_frozen (answerer agrees with pseudo-answer given image) x LP_flip."""
    pred_full = answerer.answer_question(image, question)
    xcons = float(max_match(pred_full, answer, sbert_model=sbert))
    flip = lp_flip(image, question, answerer, corrupt, sbert)
    return xcons * flip
