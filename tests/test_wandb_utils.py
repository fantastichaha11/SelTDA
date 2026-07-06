import sys
import types
from argparse import Namespace

from omegaconf import OmegaConf

import pytest


def test_init_wandb_disabled_does_not_import_wandb(monkeypatch, tmp_path):
    sys.modules.pop("wandb", None)

    from wandb_utils import init_wandb

    logger = init_wandb(
        Namespace(output_dir=str(tmp_path), evaluate=False),
        OmegaConf.create({"wandb": False}),
        job_type="train",
    )

    assert not logger.enabled
    assert "wandb" not in sys.modules


def test_init_wandb_uses_configured_run_metadata(monkeypatch, tmp_path):
    calls = {}

    class FakeRun:
        def __init__(self):
            self.summary = {}

        def log(self, *args, **kwargs):
            calls.setdefault("log", []).append((args, kwargs))

        def finish(self):
            calls["finished"] = True

    fake_wandb = types.SimpleNamespace(
        init=lambda **kwargs: calls.setdefault("init", kwargs) or FakeRun(),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    from wandb_utils import init_wandb

    config = OmegaConf.create(
        {
            "wandb": True,
            "wandb_project": "thesis-vqa",
            "wandb_entity": "lab",
            "wandb_name": "vizwiz-run",
            "wandb_group": "vizwiz",
            "wandb_tags": ["vizwiz", "debug"],
            "wandb_mode": "offline",
            "dataset_name": "generic_vqa",
            "max_epoch": 2,
        }
    )
    args = Namespace(output_dir=str(tmp_path), evaluate=False)

    logger = init_wandb(args, config, job_type="vqa-train")

    assert logger.enabled
    assert calls["init"]["project"] == "thesis-vqa"
    assert calls["init"]["entity"] == "lab"
    assert calls["init"]["name"] == "vizwiz-run"
    assert calls["init"]["group"] == "vizwiz"
    assert calls["init"]["tags"] == ["vizwiz", "debug"]
    assert calls["init"]["mode"] == "offline"
    assert calls["init"]["job_type"] == "vqa-train"
    assert calls["init"]["dir"] == str(tmp_path)
    assert calls["init"]["config"]["config"]["dataset_name"] == "generic_vqa"
    assert calls["init"]["config"]["args"]["output_dir"] == str(tmp_path)


def test_wandb_logger_logs_numeric_metrics_and_summary():
    calls = []
    run = types.SimpleNamespace(summary={})
    run.log = lambda data, step=None: calls.append((data, step))

    from wandb_utils import WandbLogger

    logger = WandbLogger(run)
    logger.log_metrics({"loss": "1.250", "lr": "0.000020", "label": "train"}, prefix="train", step=3)
    logger.update_summary({"best_epoch": 4, "note": "ok"}, prefix="run")

    assert calls == [
        ({"train/loss": 1.25, "train/lr": 0.00002, "train/label": "train"}, 3)
    ]
    assert run.summary == {"run/best_epoch": 4, "run/note": "ok"}


def test_wandb_logger_logs_existing_artifact(monkeypatch, tmp_path):
    artifact_calls = {}

    class FakeArtifact:
        def __init__(self, name, type, metadata=None):
            artifact_calls["created"] = (name, type, metadata)
            self.files = []

        def add_file(self, path):
            self.files.append(path)
            artifact_calls["file"] = path

    run = types.SimpleNamespace()
    run.log_artifact = lambda artifact: artifact_calls.setdefault("logged", artifact)
    fake_wandb = types.SimpleNamespace(Artifact=FakeArtifact)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    from wandb_utils import WandbLogger

    result = tmp_path / "vqa_result.json"
    result.write_text("[]")
    WandbLogger(run).log_artifact(result, name="eval-results", artifact_type="result", metadata={"rows": 0})

    assert artifact_calls["created"] == ("eval-results", "result", {"rows": 0})
    assert artifact_calls["file"] == str(result)
    assert artifact_calls["logged"] is not None


def test_flatten_metrics_preserves_nested_metric_paths():
    from wandb_utils import flatten_metrics

    assert flatten_metrics(
        {"overall": 0.8, "by_answer_type": {"other": 1.0, "unanswerable": 0.5}}
    ) == {
        "overall": 0.8,
        "by_answer_type/other": 1.0,
        "by_answer_type/unanswerable": 0.5,
    }


def test_wandb_logger_raises_helpful_error_when_enabled_without_dependency(monkeypatch, tmp_path):
    sys.modules.pop("wandb", None)

    from wandb_utils import init_wandb

    real_import = __import__

    def blocked_import(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked_import)

    with pytest.raises(RuntimeError, match="pip install wandb"):
        init_wandb(
            Namespace(output_dir=str(tmp_path), evaluate=False),
            OmegaConf.create({"wandb": True}),
            job_type="train",
        )
