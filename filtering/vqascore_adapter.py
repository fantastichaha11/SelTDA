"""Lazy wrapper for Lin et al. VQAScore (CLIP-FlanT5 via t2v_metrics)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATE = 'Does this figure show "{qa}"? Please answer yes or no.'


def format_vqascore_text(question: str, answer: str, template: str = DEFAULT_TEMPLATE) -> str:
    qa = f"{question} {answer}".strip()
    return template.replace("{qa}", qa)


def _score_to_float(raw: Any) -> float:
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore

    if torch is not None and isinstance(raw, torch.Tensor):
        v = raw.detach().float().reshape(-1)
        return float(v[0].item())
    if isinstance(raw, (list, tuple)) and raw:
        return _score_to_float(raw[0])
    return float(raw)


class VQAScoreAdapter:
    """Frozen VQAScore scorer; loads t2v_metrics on first use."""

    def __init__(
        self,
        *,
        model: str = "clip-flant5-xl",
        backend: str = "t2v_metrics",
        device: str | None = None,
        template: str = DEFAULT_TEMPLATE,
    ) -> None:
        self.model = model
        self.backend = backend.strip().lower()
        self.device = device
        self.template = template
        self._scorer: Any = None

    def enabled(self) -> bool:
        return self.backend not in ("", "none", "disabled")

    def _load_scorer(self) -> Any:
        if self._scorer is not None:
            return self._scorer
        if not self.enabled():
            raise RuntimeError("VQAScoreAdapter is disabled (vqascore_backend=none)")
        try:
            import t2v_metrics
        except ImportError as e:
            raise ImportError(
                "VQAScore requires t2v-metrics. Install: pip install t2v-metrics"
            ) from e

        kwargs: dict[str, Any] = {}
        if self.device:
            kwargs["device"] = self.device
        try:
            self._scorer = t2v_metrics.VQAScore(model=self.model, **kwargs)
        except TypeError:
            if kwargs:
                logger.warning(
                    "t2v_metrics.VQAScore ignores device=%s; using library default",
                    self.device,
                )
            self._scorer = t2v_metrics.VQAScore(model=self.model)
        logger.info("VQAScore loaded: model=%s device=%s", self.model, self.device)
        return self._scorer

    def score(self, image_path: str, question: str, answer: str) -> float:
        if not self.enabled():
            return 0.0
        scorer = self._load_scorer()
        text = format_vqascore_text(question, answer, self.template)
        raw = scorer(images=[image_path], texts=[text])
        val = _score_to_float(raw)
        return min(1.0, max(0.0, val))
