"""Lazy wrapper for Lin et al. VQAScore (CLIP-FlanT5 via t2v_metrics)."""

from __future__ import annotations

import logging
from pathlib import Path
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
        checkpoint: str | None = None,
    ) -> None:
        self.model = model
        self.backend = backend.strip().lower()
        self.device = device
        self.template = template
        self.checkpoint = checkpoint
        self._scorer: Any = None

    def enabled(self) -> bool:
        return self.backend not in ("", "none", "disabled")

    def _load_scorer(self) -> Any:
        if self._scorer is not None:
            return self._scorer
        if not self.enabled():
            raise RuntimeError("VQAScoreAdapter is disabled (vqascore_backend=none)")
        self._scorer = _load_clip_flant5_model(
            model=self.model,
            device=self.device,
            backend=self.backend,
            checkpoint=self.checkpoint,
        )
        src = self.checkpoint or self.model
        logger.info("VQAScore loaded: model=%s src=%s device=%s", self.model, src, self.device)
        return self._scorer

    def _score_answer(
        self,
        scorer: Any,
        image_path: str,
        qa: str,
        *,
        answer_template: str,
    ) -> float:
        question_template = self.template.replace("{qa}", "{}")
        raw = scorer.forward(
            [image_path],
            [qa],
            question_template=question_template,
            answer_template=answer_template,
        )
        return min(1.0, max(0.0, _score_to_float(raw)))

    def score(self, image_path: str, question: str, answer: str) -> float:
        """P(yes) under the VQAScore yes/no template."""
        if not self.enabled():
            return 0.0
        scorer = self._load_scorer()
        qa = f"{question} {answer}".strip()
        return self._score_answer(scorer, image_path, qa, answer_template="Yes")

    def score_yes_no_margin(self, image_path: str, question: str, answer: str) -> float:
        """P(yes) - P(no), same template as VQAScore paper."""
        if not self.enabled():
            return 0.0
        scorer = self._load_scorer()
        qa = f"{question} {answer}".strip()
        p_yes = self._score_answer(scorer, image_path, qa, answer_template="Yes")
        p_no = self._score_answer(scorer, image_path, qa, answer_template="No")
        return p_yes - p_no


def _load_clip_flant5_model(
    *,
    model: str,
    device: str | None,
    backend: str,
    checkpoint: str | None = None,
) -> Any:
    """Load CLIP-FlanT5 only; avoid t2v_metrics top-level import (heavy optional deps)."""
    if backend not in ("t2v_metrics", "t2v-metrics"):
        raise RuntimeError(f"Unsupported vqascore_backend={backend!r}")
    import importlib
    import importlib.util
    import sys
    import types
    from pathlib import Path  # used by _load_clip_flant5_model

    try:
        importlib.util.find_spec("t2v_metrics")
    except ImportError as e:
        raise ImportError(
            "VQAScore requires t2v-metrics. Install: pip install t2v-metrics"
        ) from e

    spec = importlib.util.find_spec("t2v_metrics")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("t2v-metrics package not found")
    root = Path(spec.submodule_search_locations[0])

    def _stub(name: str, path: list[str] | None = None) -> None:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            if path is not None:
                mod.__path__ = path  # type: ignore[attr-defined]
            sys.modules[name] = mod

    _stub("t2v_metrics", [str(root)])
    _stub("t2v_metrics.models", [str(root / "models")])
    _stub(
        "t2v_metrics.models.vqascore_models",
        [str(root / "models" / "vqascore_models")],
    )

    clip_mod = importlib.import_module(
        "t2v_metrics.models.vqascore_models.clip_t5_model"
    )
    if model not in clip_mod.CLIP_T5_MODELS:
        raise ValueError(
            f"Unknown VQAScore model {model!r}; choose from {list(clip_mod.CLIP_T5_MODELS)}"
        )

    load_name = model
    if checkpoint:
        ckpt_path = Path(checkpoint).expanduser().resolve()
        if not ckpt_path.is_dir():
            raise FileNotFoundError(f"vqascore_checkpoint not found: {ckpt_path}")
        import copy

        local_name = f"{model}__local"
        entry = copy.deepcopy(clip_mod.CLIP_T5_MODELS[model])
        entry["model"]["path"] = str(ckpt_path)
        clip_mod.CLIP_T5_MODELS[local_name] = entry
        load_name = local_name

    kwargs: dict[str, Any] = {}
    if device:
        kwargs["device"] = device
    return clip_mod.CLIPT5Model(load_name, **kwargs)
