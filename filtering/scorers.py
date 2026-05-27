"""Per-sample score functions for the 3 filter gates."""

from __future__ import annotations

from typing import List, Optional, Protocol, Sequence

import numpy as np

from filtering.matchers import max_match


def score_confidence(record: dict) -> Optional[float]:
    """Return the teacher decoder mean log-prob attached to a record."""
    v = record.get("gen_logprob")
    if v is None:
        return None
    return float(v)


def normalize_min_max(values: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Min-max normalize a sequence, preserving None positions."""
    non_none = [v for v in values if v is not None]
    if not non_none:
        return [None for _ in values]
    lo, hi = min(non_none), max(non_none)
    if lo == hi:
        return [0.5 if v is not None else None for v in values]
    span = hi - lo
    return [None if v is None else (v - lo) / span for v in values]


class ClipLike(Protocol):
    def embed_image(self, image): ...
    def embed_text(self, text: str): ...


def _format_qa_for_clip(record: dict) -> str:
    answer = record["answer"]
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    return f"{record['question']} {answer}".strip()


def _clip_cosine_score(img_emb: np.ndarray, txt_emb: np.ndarray) -> float:
    img_emb = np.asarray(img_emb, dtype=np.float64).reshape(-1)
    txt_emb = np.asarray(txt_emb, dtype=np.float64).reshape(-1)
    ni = np.linalg.norm(img_emb)
    nt = np.linalg.norm(txt_emb)
    if ni == 0 or nt == 0:
        return 0.0
    cos = float(np.dot(img_emb, txt_emb) / (ni * nt))
    return max(0.0, min(1.0, (cos + 1.0) / 2.0))


def score_clip_itm(record: dict, image, clip: ClipLike) -> float:
    """Cosine similarity between CLIP image embedding and "Q? A." text embedding."""
    text = _format_qa_for_clip(record)
    img_emb = clip.embed_image(image)
    txt_emb = clip.embed_text(text)
    return _clip_cosine_score(img_emb, txt_emb)


def score_clip_itm_from_image_emb(record: dict, img_emb, clip: ClipLike) -> float:
    """ITM score when the image embedding is already computed (e.g. batched)."""
    text = _format_qa_for_clip(record)
    txt_emb = clip.embed_text(text)
    return _clip_cosine_score(img_emb, txt_emb)


class StudentLike(Protocol):
    def answer_question(self, image, question: str) -> str: ...


def _pseudo_answer(record: dict) -> str:
    answer = record["answer"]
    if isinstance(answer, list):
        return answer[0] if answer else ""
    return str(answer)


def score_xcons(
    record: dict,
    image,
    student: StudentLike,
    sbert,
    *,
    predicted: str | None = None,
) -> tuple[float, str]:
    """Cross-consistency: frozen Student zero-shot answer vs pseudo-answer.

    Returns (score, student_prediction). Pass ``predicted`` to skip a forward pass.
    """
    answer = _pseudo_answer(record)
    if predicted is None:
        predicted = student.answer_question(image, record["question"])
    score = max_match(predicted, answer, sbert_model=sbert)
    return float(score), predicted
