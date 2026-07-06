from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

import utils


def _config_get(config: DictConfig | Mapping[str, Any], key: str, default: Any = None) -> Any:
    if isinstance(config, DictConfig):
        return OmegaConf.select(config, key, default=default)
    return config.get(key, default)


def _as_plain_config(config: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)
    return dict(config)


def _args_to_dict(args: Any) -> dict[str, Any]:
    values = vars(args).copy()
    return {key: str(value) if isinstance(value, Path) else value for key, value in values.items()}


def _to_log_value(value: Any) -> Any:
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _prefixed(values: Mapping[str, Any], prefix: str | None) -> dict[str, Any]:
    if not prefix:
        return {key: _to_log_value(value) for key, value in values.items()}
    return {f"{prefix}/{key}": _to_log_value(value) for key, value in values.items()}


def flatten_metrics(values: Mapping[str, Any], *, parent_key: str | None = None) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in values.items():
        metric_key = f"{parent_key}/{key}" if parent_key else key
        if isinstance(value, Mapping):
            flattened.update(flatten_metrics(value, parent_key=metric_key))
        else:
            flattened[metric_key] = _to_log_value(value)
    return flattened


class DisabledWandbLogger:
    enabled = False

    def log_metrics(self, *args: Any, **kwargs: Any) -> None:
        return None

    def update_summary(self, *args: Any, **kwargs: Any) -> None:
        return None

    def log_artifact(self, *args: Any, **kwargs: Any) -> None:
        return None

    def watch_model(self, *args: Any, **kwargs: Any) -> None:
        return None

    def finish(self) -> None:
        return None


class WandbLogger:
    enabled = True

    def __init__(self, run: Any, wandb_module: Any | None = None) -> None:
        self.run = run
        self._wandb = wandb_module

    def log_metrics(
        self,
        metrics: Mapping[str, Any],
        *,
        prefix: str | None = None,
        step: int | None = None,
    ) -> None:
        self.run.log(_prefixed(flatten_metrics(metrics), prefix), step=step)

    def update_summary(self, values: Mapping[str, Any], *, prefix: str | None = None) -> None:
        for key, value in _prefixed(flatten_metrics(values), prefix).items():
            self.run.summary[key] = value

    def log_artifact(
        self,
        path: str | Path,
        *,
        name: str,
        artifact_type: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        path = Path(path)
        if not path.exists():
            return
        wandb = self._wandb
        if wandb is None:
            import wandb as wandb  # type: ignore[no-redef]
        artifact = wandb.Artifact(name=name, type=artifact_type, metadata=dict(metadata or {}))
        artifact.add_file(str(path))
        self.run.log_artifact(artifact)

    def watch_model(self, model: Any, *, log: str = "gradients", log_freq: int = 100) -> None:
        wandb = self._wandb
        if wandb is None:
            import wandb as wandb  # type: ignore[no-redef]
        wandb.watch(model, log=log, log_freq=log_freq)

    def finish(self) -> None:
        self.run.finish()


def init_wandb(
    args: Any,
    config: DictConfig | Mapping[str, Any],
    *,
    job_type: str,
) -> WandbLogger | DisabledWandbLogger:
    if not bool(_config_get(config, "wandb", False)) or not utils.is_main_process():
        return DisabledWandbLogger()

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging is enabled but wandb is not installed. "
            "Install it with `pip install wandb`, or run with `wandb=false`."
        ) from exc

    config_dict = _as_plain_config(config)
    output_dir = str(getattr(args, "output_dir", "."))
    run = wandb.init(
        project=_config_get(config, "wandb_project", "seltda"),
        entity=_config_get(config, "wandb_entity", None),
        name=_config_get(config, "wandb_name", None),
        group=_config_get(config, "wandb_group", _config_get(config, "dataset_name", None)),
        tags=_config_get(config, "wandb_tags", None),
        notes=_config_get(config, "wandb_notes", None),
        mode=_config_get(config, "wandb_mode", None),
        id=_config_get(config, "wandb_run_id", None),
        resume=_config_get(config, "wandb_resume", "allow"),
        job_type=job_type,
        dir=output_dir,
        config={"config": config_dict, "args": _args_to_dict(args)},
    )
    return WandbLogger(run, wandb)
