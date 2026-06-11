"""VQAScore gate: image–text alignment via CLIP-FlanT5 (t2v_metrics)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

from filtering.scorers import _format_qa_for_clip


class VQAScoreLike(Protocol):
    def score_pairs(
        self, image_paths: Sequence[str | Path], texts: Sequence[str]
    ) -> list[float]: ...


def score_vqascore(record: dict, image_path: str | Path, scorer: VQAScoreLike) -> float:
    """P('Yes' | image, 'Does this image show "{qa_text}"?') via CLIP-FlanT5."""
    text = _format_qa_for_clip(record)
    scores = scorer.score_pairs([image_path], [text])
    return float(scores[0])
