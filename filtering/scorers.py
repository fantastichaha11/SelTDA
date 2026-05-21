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


def score_clip_itm(record: dict, image, clip: ClipLike) -> float:
    """Cosine similarity between CLIP image embedding and "Q? A." text embedding."""
    text = _format_qa_for_clip(record)
    img_emb = np.asarray(clip.embed_image(image), dtype=np.float64).reshape(-1)
    txt_emb = np.asarray(clip.embed_text(text), dtype=np.float64).reshape(-1)
    ni = np.linalg.norm(img_emb)
    nt = np.linalg.norm(txt_emb)
    if ni == 0 or nt == 0:
        return 0.0
    cos = float(np.dot(img_emb, txt_emb) / (ni * nt))
    return max(0.0, min(1.0, (cos + 1.0) / 2.0))


class StudentLike(Protocol):
    def answer_question(self, image, question: str) -> str: ...


def score_xcons(record: dict, image, student: StudentLike, sbert) -> float:
    """Cross-consistency: frozen Student zero-shot answer vs pseudo-answer."""
    answer = record["answer"]
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    predicted = student.answer_question(image, record["question"])
    return max_match(predicted, answer, sbert_model=sbert)
