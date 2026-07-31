import json

from omegaconf import OmegaConf

from judge.prometheus import DEFAULT_RUBRIC, StaticPrometheusScorer
import scripts.eval_prometheus_judge as eval_prometheus_judge
import scripts.train_prometheus_judge as train_prometheus_judge
from scripts.eval_prometheus_judge import build_scorer, run_eval
from scripts.train_prometheus_judge import (
    build_external_command,
    build_training_artifacts,
    run_train,
    validate_external_llava_command,
)


def test_prometheus_judge_config_loads():
    cfg = OmegaConf.load("configs/prometheus_judge_pathvqa.yaml")
    resolved = OmegaConf.to_container(cfg, resolve=True)

    assert cfg.data.train_annotations == "datasets/pathvqa/train.json"
    assert cfg.data.val_annotations == "datasets/pathvqa/val.json"
    assert cfg.prompt.no_reference is True
    assert cfg.eval.split == "val"
    assert cfg.train.use_pseudo_qa is False
    assert resolved["model"]["model_base"] is None
    assert "prometheus-eval/prometheus-vision-7b-v1.0" in resolved["train"]["external_command"]["extra_args"]
    assert "vicuna_v1" in resolved["train"]["external_command"]["extra_args"]
    assert "outputs/prometheus_judge/pathvqa_prometheus_sft.json" in resolved["train"]["external_command"]["extra_args"]
    assert "outputs/prometheus_judge/pathvqa_llava_lora_adapted" in resolved["train"]["external_command"]["extra_args"]


def test_external_llava_command_rejects_plain_template_for_judge_sft():
    command = ["deepspeed", "llava/train/train.py", "--version", "plain"]

    try:
        validate_external_llava_command(command)
    except ValueError as exc:
        assert "drops the rubric and question prompt" in str(exc)
    else:
        raise AssertionError("plain LLaVA template should be rejected for judge SFT")


def test_grpo_teacher_config_uses_pathvqa_train_images_only():
    cfg = OmegaConf.load("configs/grpo_teacher_pathvqa_prometheus.yaml")
    resolved = OmegaConf.to_container(cfg, resolve=True)

    assert cfg.image_pool.name == "pathvqa_train"
    assert cfg.image_pool.annotations == "datasets/pathvqa/train.json"
    assert cfg.image_pool.use_ground_truth_qa is False
    assert cfg.reward.judge_config == "configs/prometheus_judge_pathvqa.yaml"
    assert resolved["reward"]["selected_judge_model_path"] == "cache/prometheus-vision-7b-v1.0"
    assert resolved["teacher"]["pretrained"] == "cache/pathvqa_teacher_weights/checkpoint_04.pth"


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class LengthAwareScorer:
    def score(self, image_path: str, question: str, candidate_answer: str):
        del image_path, question
        score = 5 if candidate_answer in {"yes", "nuclei"} else 1
        return StaticPrometheusScorer(score=score, feedback="mock").score(
            image_path="unused",
            question="unused",
            candidate_answer="unused",
        )


def test_run_eval_scores_val_pairs_without_reference_prompt(tmp_path):
    train_path = tmp_path / "fixtures" / "train.json"
    val_path = tmp_path / "fixtures" / "val.json"
    output_path = tmp_path / "reports" / "eval.json"

    _write_json(
        train_path,
        [
            {"image": "train-1.jpg", "question": "Is there necrosis?", "answer": "yes"},
            {"image": "train-2.jpg", "question": "What structure is highlighted?", "answer": "nuclei"},
            {"image": "train-3.jpg", "question": "What structure is highlighted?", "answer": "cytoplasm"},
            {"image": "train-4.jpg", "question": "What structure is highlighted?", "answer": "stroma"},
        ],
    )
    _write_json(
        val_path,
        [
            {"image": "val-1.jpg", "question": "Is there necrosis?", "answer": "yes"},
            {"image": "val-2.jpg", "question": "What structure is highlighted?", "answer": "nuclei"},
        ],
    )

    cfg = OmegaConf.create(
        {
            "data": {
                "train_annotations": str(train_path),
                "val_annotations": str(val_path),
                "image_root": str(tmp_path),
                "negatives_per_positive": 1,
            },
            "model": {
                "model_path": "unused",
                "model_base": None,
                "conv_mode": "vicuna_v1",
                "device": "cpu",
                "temperature": 0.0,
                "max_new_tokens": 32,
            },
            "eval": {
                "output": str(output_path),
                "max_examples": None,
                "mock_score": None,
            },
            "train": {
                "seed": 42,
            },
        }
    )

    metrics = run_eval(cfg, scorer=LengthAwareScorer())

    assert metrics["num_pairs"] == 2
    assert metrics["pairwise_accuracy"] == 1.0
    assert metrics["auroc"] == 1.0
    assert "score_distribution" in metrics

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["metrics"]["num_pairs"] == 2
    assert metrics["score_distribution"] == {
        "by_question_prefix": {
            "is/are": {"1": 1, "5": 1},
            "what": {"1": 1, "5": 1},
        },
        "by_answer_type": {
            "yes/no": {"1": 1, "5": 1},
            "phrase": {"1": 1, "5": 1},
        },
    }
    assert report["metrics"]["score_distribution"] == metrics["score_distribution"]

    by_question = {example["question"]: example for example in report["examples"]}
    assert by_question["Is there necrosis?"]["negative_answer"] == "no"
    assert by_question["What structure is highlighted?"]["negative_answer"] == "cytoplasm"
    assert by_question["What structure is highlighted?"]["negative_answer"] != "stroma"

    example = report["examples"][0]
    assert {
        "image",
        "question",
        "positive_answer",
        "negative_answer",
        "positive_score",
        "negative_score",
        "question_prefix",
        "positive_answer_type",
        "negative_answer_type",
        "positive_feedback",
        "negative_feedback",
    }.issubset(example)
    assert by_question["Is there necrosis?"]["negative_score"] == 1
    assert by_question["Is there necrosis?"]["negative_answer_type"] == "yes/no"
    assert by_question["What structure is highlighted?"]["negative_answer_type"] == "phrase"


def test_build_scorer_uses_static_scorer_for_mock_score():
    cfg = OmegaConf.create(
        {
            "eval": {"mock_score": 4},
            "model": {
                "model_path": "unused",
                "model_base": None,
                "conv_mode": "vicuna_v1",
                "device": "cpu",
                "temperature": 0.0,
                "max_new_tokens": 32,
            },
        }
    )

    scorer = build_scorer(cfg)

    assert isinstance(scorer, StaticPrometheusScorer)
    assert scorer.static_score == 4


def test_build_scorer_normalizes_nullish_model_base(monkeypatch):
    calls = []

    class FakePrometheusVisionScorer:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(
        eval_prometheus_judge,
        "PrometheusVisionScorer",
        FakePrometheusVisionScorer,
    )

    for raw_model_base in (None, "null", "none"):
        cfg = OmegaConf.create(
            {
                "eval": {"mock_score": None},
                "model": {
                    "model_path": "vision-model",
                    "model_base": raw_model_base,
                    "conv_mode": "vicuna_v1",
                    "device": "cpu",
                    "temperature": 0.0,
                    "max_new_tokens": 32,
                },
            }
        )

        build_scorer(cfg)

    assert [call["model_base"] for call in calls] == [None, None, None]


def test_build_scorer_preserves_non_null_model_base(monkeypatch):
    captured = {}

    class FakePrometheusVisionScorer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        eval_prometheus_judge,
        "PrometheusVisionScorer",
        FakePrometheusVisionScorer,
    )

    cfg = OmegaConf.create(
        {
            "eval": {"mock_score": None},
            "model": {
                "model_path": "vision-model",
                "model_base": "adapter-base",
                "conv_mode": "vicuna_v1",
                "device": "cuda:0",
                "temperature": 0.2,
                "max_new_tokens": 64,
            },
        }
    )

    build_scorer(cfg)

    assert captured == {
        "model_path": "vision-model",
        "model_base": "adapter-base",
        "conv_mode": "vicuna_v1",
        "device": "cuda:0",
        "temperature": 0.2,
        "max_new_tokens": 64,
        "rubric": DEFAULT_RUBRIC,
    }


def test_build_scorer_passes_yaml_rubric(monkeypatch):
    captured = {}

    class FakePrometheusVisionScorer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        eval_prometheus_judge,
        "PrometheusVisionScorer",
        FakePrometheusVisionScorer,
    )

    cfg = OmegaConf.create(
        {
            "eval": {"mock_score": None},
            "model": {
                "model_path": "vision-model",
                "model_base": None,
                "conv_mode": "vicuna_v1",
                "device": "cuda:0",
                "temperature": 0.0,
                "max_new_tokens": 64,
            },
            "prompt": {
                "rubric_name": "custom",
                "criteria_count": 8,
                "rubric": "Use the custom rubric.",
            },
        }
    )

    build_scorer(cfg)

    assert captured["rubric"] == "Use the custom rubric."


def test_main_applies_overrides_and_explicit_flags(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
data:
  train_annotations: train.json
  val_annotations: val.json
  image_root: images
  negatives_per_positive: 1
model:
  model_path: base-model
  model_base: null
  conv_mode: vicuna_v1
  device: cpu
  temperature: 0.0
  max_new_tokens: 32
eval:
  output: default.json
  max_examples: null
  mock_score: null
train:
  seed: 42
""".strip(),
        encoding="utf-8",
    )

    captured = {}

    def fake_run_eval(config, scorer=None):
        del scorer
        captured["config"] = OmegaConf.to_container(config, resolve=True)
        return {"num_pairs": 1}

    monkeypatch.setattr(eval_prometheus_judge, "run_eval", fake_run_eval)
    monkeypatch.setattr(
        "sys.argv",
        [
            "eval_prometheus_judge.py",
            "--config",
            str(config_path),
            "--overrides",
            "model.device=cuda:1",
            "eval.output=from_override.json",
            "--output",
            str(tmp_path / "explicit.json"),
            "--max-examples",
            "3",
            "--mock-score",
            "4",
        ],
    )

    eval_prometheus_judge.main()

    assert captured["config"]["model"]["device"] == "cuda:1"
    assert captured["config"]["eval"]["output"] == str(tmp_path / "explicit.json")
    assert captured["config"]["eval"]["max_examples"] == 3
    assert captured["config"]["eval"]["mock_score"] == 4
    assert json.loads(capsys.readouterr().out)["num_pairs"] == 1


def test_build_training_artifacts_use_train_split_only(tmp_path):
    train_path = tmp_path / "train.json"
    _write_json(
        train_path,
        [
            {"image": "img/a.jpg", "question": "Is this benign?", "answer": "yes"},
            {"image": "img/b.jpg", "question": "What is visible?", "answer": "nuclei"},
        ],
    )
    cfg = OmegaConf.create(
        {
            "data": {
                "train_annotations": str(train_path),
                "image_root": str(tmp_path / "images"),
                "negatives_per_positive": 1,
            },
            "train": {
                "seed": 5,
                "train_pairs_jsonl": str(tmp_path / "pairs.jsonl"),
                "prometheus_sft_json": str(tmp_path / "sft.json"),
            },
        }
    )
    summary = build_training_artifacts(cfg)
    assert summary["num_pairs"] == 2
    assert "val" not in (tmp_path / "pairs.jsonl").read_text(encoding="utf-8")
    sft = json.loads((tmp_path / "sft.json").read_text(encoding="utf-8"))
    assert len(sft) == 4
    assert sft[0]["conversations"][1]["value"].endswith("[RESULT] 5")
    assert sft[1]["conversations"][1]["value"].endswith("[RESULT] 1")


def test_build_external_command_resolves_data_and_output_paths():
    cfg = OmegaConf.create(
        {
            "train": {
                "output_dir": "out/judge",
                "prometheus_sft_json": "out/sft.json",
                "external_command": {
                    "executable": "deepspeed",
                    "script": "llava/train/train_mem.py",
                    "extra_args": ["--data_path", "${train.prometheus_sft_json}", "--output_dir", "${train.output_dir}"],
                },
            }
        }
    )
    command = build_external_command(cfg)
    assert command == [
        "deepspeed",
        "llava/train/train_mem.py",
        "--data_path",
        "out/sft.json",
        "--output_dir",
        "out/judge",
    ]


def test_run_train_dry_run_skips_subprocess(tmp_path, monkeypatch):
    train_path = tmp_path / "train.json"
    _write_json(train_path, [{"image": "img/a.jpg", "question": "Is this benign?", "answer": "yes"}])
    cfg = OmegaConf.create(
        {
            "data": {
                "train_annotations": str(train_path),
                "image_root": str(tmp_path / "images"),
                "negatives_per_positive": 1,
            },
            "train": {
                "seed": 5,
                "backend": "dry_run",
                "train_pairs_jsonl": str(tmp_path / "pairs.jsonl"),
                "prometheus_sft_json": str(tmp_path / "sft.json"),
            },
        }
    )

    def fail_run(*args, **kwargs):
        raise AssertionError(f"subprocess.run should not be called: {args} {kwargs}")

    monkeypatch.setattr(train_prometheus_judge.subprocess, "run", fail_run)

    result = run_train(cfg)

    assert result["backend"] == "dry_run"
    assert result["num_pairs"] == 1
    assert (tmp_path / "pairs.jsonl").exists()
    assert (tmp_path / "sft.json").exists()


def test_run_train_external_llava_runs_resolved_command(monkeypatch):
    cfg = OmegaConf.create(
        {
            "train": {
                "backend": "external_llava",
                "output_dir": "out/judge",
                "prometheus_sft_json": "out/sft.json",
                "external_command": {
                    "executable": "deepspeed",
                    "script": "llava/train/train_mem.py",
                    "extra_args": ["--data_path", "${train.prometheus_sft_json}", "--output_dir", "${train.output_dir}"],
                },
            }
        }
    )
    captured = {}

    def fake_build_training_artifacts(config):
        captured["build_config"] = config
        return {"num_pairs": 2, "num_sft_examples": 4}

    def fake_run(command, check):
        captured["command"] = command
        captured["check"] = check

    monkeypatch.setattr(train_prometheus_judge, "build_training_artifacts", fake_build_training_artifacts)
    monkeypatch.setattr(train_prometheus_judge.subprocess, "run", fake_run)

    result = run_train(cfg)

    assert result == {
        "backend": "external_llava",
        "command": [
            "deepspeed",
            "llava/train/train_mem.py",
            "--data_path",
            "out/sft.json",
            "--output_dir",
            "out/judge",
        ],
        "num_pairs": 2,
        "num_sft_examples": 4,
    }
    assert captured["check"] is True
    assert captured["command"] == result["command"]


def test_run_train_rejects_invalid_backend_before_building_outputs(monkeypatch):
    cfg = OmegaConf.create({"train": {"backend": "invalid"}})
    called = {"build": False}

    def fake_build_training_artifacts(config):
        del config
        called["build"] = True
        return {"num_pairs": 1, "num_sft_examples": 2}

    monkeypatch.setattr(train_prometheus_judge, "build_training_artifacts", fake_build_training_artifacts)

    try:
        run_train(cfg)
        raise AssertionError("run_train should reject invalid backends")
    except ValueError as exc:
        assert str(exc) == "train.backend must be dry_run or external_llava"

    assert called["build"] is False


def test_train_main_applies_backend_override(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
data:
  train_annotations: train.json
  image_root: images
  negatives_per_positive: 1
train:
  seed: 42
  backend: external_llava
  train_pairs_jsonl: pairs.jsonl
  prometheus_sft_json: sft.json
  output_dir: out
  external_command:
    executable: deepspeed
    script: llava/train/train_mem.py
    extra_args: []
""".strip(),
        encoding="utf-8",
    )
    captured = {}

    def fake_run_train(config):
        captured["config"] = OmegaConf.to_container(config, resolve=True)
        return {"backend": str(config.train.backend), "num_pairs": 1}

    monkeypatch.setattr(train_prometheus_judge, "run_train", fake_run_train)
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_prometheus_judge.py",
            "--config",
            str(config_path),
            "--backend",
            "dry_run",
        ],
    )

    train_prometheus_judge.main()

    assert captured["config"]["train"]["backend"] == "dry_run"
    assert json.loads(capsys.readouterr().out) == {"backend": "dry_run", "num_pairs": 1}
