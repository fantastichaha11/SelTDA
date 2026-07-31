from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from judge.prometheus import DEFAULT_RUBRIC, PrometheusVisionScorer


@dataclass(frozen=True)
class ResolvedPrometheusConfig:
    source_config: str
    model_path: str
    model_base: str | None
    conv_mode: str
    device: str
    temperature: float
    max_new_tokens: int | None
    rubric_name: str
    criteria_count: int
    rubric: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _nullish(value: object) -> bool:
    return value is None or str(value).lower() in {"none", "null"}


def _load_config(config_or_path) -> tuple[DictConfig, str]:
    if isinstance(config_or_path, (str, Path)):
        path = Path(config_or_path)
        return OmegaConf.load(str(path)), str(path)
    return config_or_path, "<inline>"


def resolve_prometheus_config(
    config_or_path,
    *,
    selected_model_path: str | None = None,
    device: str | None = None,
) -> ResolvedPrometheusConfig:
    cfg, source = _load_config(config_or_path)
    raw_model_path = OmegaConf.select(cfg, "model.model_path")
    if raw_model_path is None and selected_model_path is None:
        raise ValueError("Prometheus config requires model.model_path or selected_model_path")

    raw_base = OmegaConf.select(cfg, "model.model_base", default=None)
    model_path = str(selected_model_path or raw_model_path)
    model_base = None if _nullish(raw_base) else str(raw_base)
    if model_base == model_path:
        model_base = None

    raw_tokens = OmegaConf.select(cfg, "model.max_new_tokens", default=512)
    max_new_tokens = None if _nullish(raw_tokens) else int(raw_tokens)

    rubric = OmegaConf.select(cfg, "prompt.rubric", default=None) or DEFAULT_RUBRIC
    return ResolvedPrometheusConfig(
        source_config=source,
        model_path=model_path,
        model_base=model_base,
        conv_mode=str(OmegaConf.select(cfg, "model.conv_mode", default="vicuna_v1")),
        device=str(device or OmegaConf.select(cfg, "model.device", default="cuda")),
        temperature=float(OmegaConf.select(cfg, "model.temperature", default=0.0)),
        max_new_tokens=max_new_tokens,
        rubric_name=str(OmegaConf.select(cfg, "prompt.rubric_name", default="default")),
        criteria_count=int(OmegaConf.select(cfg, "prompt.criteria_count", default=4)),
        rubric=str(rubric),
    )


def build_prometheus_scorer(config: ResolvedPrometheusConfig) -> PrometheusVisionScorer:
    return PrometheusVisionScorer(
        model_path=config.model_path,
        model_base=config.model_base,
        conv_mode=config.conv_mode,
        device=config.device,
        temperature=config.temperature,
        max_new_tokens=config.max_new_tokens,
        rubric=config.rubric,
    )


def write_judge_snapshot(path: str | Path, config: ResolvedPrometheusConfig) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
