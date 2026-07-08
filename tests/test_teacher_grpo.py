import json
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
from omegaconf import OmegaConf

import scripts.train_teacher_grpo as teacher_grpo
from judge.prometheus import StaticPrometheusScorer
from scripts.train_teacher_grpo import (
    GeneratedQA,
    BlipTeacherPolicy,
    MockTeacherPolicy,
    _parse_generated_qa,
    advantage_weighted_policy_loss,
    run_grpo,
    score_candidate_group,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _make_grpo_config(tmp_path, *, dry_run):
    train_path = tmp_path / "train.json"
    _write_json(
        train_path,
        [
            {"image": "img/a.jpg", "question": "Gold question?", "answer": "gold"},
        ],
    )
    return OmegaConf.create(
        {
            "image_pool": {
                "name": "pathvqa_train",
                "annotations": str(train_path),
                "image_root": str(tmp_path / "images"),
                "use_ground_truth_qa": False,
            },
            "teacher": {
                "output_dir": str(tmp_path / "out"),
                "batch_size": 1,
                "candidates_per_image": 1,
                "max_steps": 1,
                "dry_run": dry_run,
                "device": "cpu",
                "lr": 1e-6,
                "config": str(tmp_path / "teacher.yaml"),
            },
            "generation": {
                "top_p": 0.9,
                "max_length": 10,
                "min_length": 1,
            },
            "reward": {
                "duplicate_question_penalty": 0.0,
                "length_penalty": 0.0,
                "generic_answer_penalty": 0.0,
                "max_question_words": 10,
                "max_answer_words": 4,
            },
            "logging": {
                "candidates_jsonl": str(tmp_path / "candidates" / "candidates.jsonl"),
                "metrics_jsonl": str(tmp_path / "metrics" / "metrics.jsonl"),
            },
            "seed": 42,
        }
    )


def test_score_candidate_group_adds_rewards_and_advantages():
    candidates = [
        GeneratedQA(
            image="img/a.jpg",
            image_path="/tmp/a.jpg",
            question="Q1?",
            answer="mitosis",
            logprob=-1.0,
        ),
        GeneratedQA(
            image="img/a.jpg",
            image_path="/tmp/a.jpg",
            question="Q1?",
            answer="cells",
            logprob=-2.0,
        ),
    ]
    rows = score_candidate_group(
        candidates,
        judge=StaticPrometheusScorer(score=4, feedback="mock"),
        reward_cfg={
            "duplicate_question_penalty": 0.1,
            "length_penalty": 0.0,
            "generic_answer_penalty": 0.1,
            "max_question_words": 10,
            "max_answer_words": 4,
        },
    )
    assert len(rows) == 2
    assert rows[0]["reward"] == 0.75
    assert rows[1]["reward"] == 0.55
    assert rows[0]["advantage"] > rows[1]["advantage"]


class RecordingTeacherPolicy:
    def __init__(self):
        self.updated_rows = None

    def generate(self, item, k):
        return MockTeacherPolicy().generate(item, k)

    def update(self, rows):
        self.updated_rows = list(rows)
        return {"policy_loss": 0.25}


def test_run_grpo_logs_rows_and_metrics_to_split_directories(tmp_path):
    train_path = tmp_path / "train.json"
    _write_json(
        train_path,
        [
            {"image": "img/a.jpg", "question": "Gold question?", "answer": "gold"},
            {"image": "img/b.jpg", "question": "Other?", "answer": "answer"},
        ],
    )
    cfg = OmegaConf.create(
        {
            "image_pool": {
                "name": "pathvqa_train",
                "annotations": str(train_path),
                "image_root": str(tmp_path / "images"),
                "use_ground_truth_qa": False,
            },
            "teacher": {
                "output_dir": str(tmp_path / "out"),
                "batch_size": 1,
                "candidates_per_image": 2,
                "max_steps": 1,
                "dry_run": True,
            },
            "reward": {
                "duplicate_question_penalty": 0.0,
                "length_penalty": 0.0,
                "generic_answer_penalty": 0.0,
                "max_question_words": 10,
                "max_answer_words": 4,
            },
            "logging": {
                "candidates_jsonl": str(tmp_path / "candidates" / "candidates.jsonl"),
                "metrics_jsonl": str(tmp_path / "metrics" / "metrics.jsonl"),
            },
            "seed": 42,
        }
    )
    teacher = RecordingTeacherPolicy()
    result = run_grpo(
        cfg,
        teacher=teacher,
        judge=StaticPrometheusScorer(score=5, feedback="mock"),
    )
    assert result["steps"] == 1
    candidates_path = tmp_path / "candidates" / "candidates.jsonl"
    metrics_path = tmp_path / "metrics" / "metrics.jsonl"
    candidate_rows = _read_jsonl(candidates_path)
    metric_rows = _read_jsonl(metrics_path)
    assert len(candidate_rows) == 2
    assert len(metric_rows) == 1
    assert teacher.updated_rows == candidate_rows
    assert all(row["question"].startswith("Generated question") for row in candidate_rows)
    assert all("Gold question?" not in json.dumps(row) for row in candidate_rows)
    assert candidate_rows[0]["image"] == "img/a.jpg"
    assert candidate_rows[0]["image_path"] == str(tmp_path / "images" / "img/a.jpg")
    assert candidate_rows[0]["judge_score"] == 5
    assert candidate_rows[0]["feedback"] == "mock"
    assert candidate_rows[0]["reward"] == 1.0
    assert "advantage" in candidate_rows[0]
    assert metric_rows[0]["policy_loss"] == 0.25
    assert metric_rows[0]["step"] == 0
    assert "mean_reward" in metric_rows[0]


def test_train_teacher_grpo_cli_runs_from_repo_root_and_prints_json(tmp_path):
    train_path = tmp_path / "train.json"
    _write_json(
        train_path,
        [
            {"image": "img/a.jpg", "question": "Gold question?", "answer": "gold"},
        ],
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "image_pool:",
                f"  annotations: {train_path}",
                f"  image_root: {tmp_path / 'images'}",
                "  use_ground_truth_qa: false",
                "teacher:",
                "  output_dir: ignored",
                "  batch_size: 1",
                "  candidates_per_image: 2",
                "  max_steps: 1",
                "  dry_run: true",
                "reward:",
                "  duplicate_question_penalty: 0.0",
                "  length_penalty: 0.0",
                "  generic_answer_penalty: 0.0",
                "  max_question_words: 10",
                "  max_answer_words: 4",
                "logging:",
                f"  candidates_jsonl: {tmp_path / 'logs' / 'candidates.jsonl'}",
                f"  metrics_jsonl: {tmp_path / 'logs' / 'metrics.jsonl'}",
                "seed: 42",
            ]
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train_teacher_grpo.py",
            "--config",
            str(config_path),
            "--mock-judge-score",
            "5",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload == {"steps": 1}


def test_advantage_weighted_policy_loss_uses_signed_advantages():
    losses = [2.0, 4.0]
    advantages = [1.0, -0.5]
    assert advantage_weighted_policy_loss(losses, advantages) == 0.0


def test_parse_generated_qa_uses_repo_parser_for_spaced_answer_marker():
    question, answer = _parse_generated_qa(
        " what might the person next to the suitcase be doing?. answer : waiting, waiting"
    )
    assert question == "what might the person next to the suitcase be doing?"
    assert answer == "waiting,waiting"


@pytest.mark.parametrize(
    ("text", "expected_question", "expected_answer"),
    [
        (
            ": what does this animal live in?. answer : trees, forest.",
            "what does this animal live in?",
            "trees,forest",
        ),
        (
            " what might the person next to the suitcase be doing?. answer : waiting, waiting",
            "what might the person next to the suitcase be doing?",
            "waiting,waiting",
        ),
    ],
)
def test_parse_generated_qa_handles_repo_style_output_variants(
    text, expected_question, expected_answer
):
    question, answer = _parse_generated_qa(text)
    assert question == expected_question
    assert answer == expected_answer


def test_run_grpo_uses_blip_teacher_and_default_judge_when_not_dry_run(
    tmp_path, monkeypatch
):
    cfg = _make_grpo_config(tmp_path, dry_run=False)
    calls = {}

    class FakeBlipTeacherPolicy:
        def __init__(self, config):
            calls["teacher_config"] = config

        def generate(self, item, k):
            calls["generated_item"] = item
            calls["generated_k"] = k
            return [
                GeneratedQA(
                    image=item.image,
                    image_path=item.image_path,
                    question="Generated question?",
                    answer="mitosis",
                    logprob=-0.25,
                )
            ]

        def update(self, rows):
            calls["updated_rows"] = list(rows)
            return {"policy_loss": 0.5}

    class FakeJudge:
        def score(self, image_path, question, answer):
            calls["judge_inputs"] = (image_path, question, answer)
            return SimpleNamespace(score=5, reward=1.0, feedback="mock")

    def fake_build_judge_from_config(config):
        calls["judge_config"] = config
        return FakeJudge()

    monkeypatch.setattr(teacher_grpo, "BlipTeacherPolicy", FakeBlipTeacherPolicy)
    monkeypatch.setattr(
        teacher_grpo, "build_judge_from_config", fake_build_judge_from_config
    )

    result = run_grpo(cfg, teacher=None, judge=None)

    assert result == {"steps": 1}
    assert calls["teacher_config"] is cfg
    assert calls["judge_config"] is cfg
    assert calls["generated_k"] == 1
    assert calls["judge_inputs"][1:] == ("Generated question?", "mitosis")
    assert calls["updated_rows"][0]["judge_score"] == 5
    assert calls["updated_rows"][0]["feedback"] == "mock"


def test_blip_teacher_generate_uses_eval_mode_for_sampling():
    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeTorch:
        def no_grad(self):
            return FakeNoGrad()

    class FakeTensor:
        def repeat(self, *args):
            return self

    class FakeModel:
        def __init__(self):
            self.training = True
            self.events = []

        def train(self, mode=True):
            self.training = mode
            self.events.append(("train", mode))
            return self

        def eval(self):
            self.training = False
            self.events.append(("eval",))
            return self

        def generate(self, image, **kwargs):
            self.events.append(("generate", self.training, kwargs))
            return [" what might the person next to the suitcase be doing?. answer : waiting"], [-0.5]

    policy = object.__new__(BlipTeacherPolicy)
    policy.torch = FakeTorch()
    policy.model = FakeModel()
    policy.generation = SimpleNamespace(top_p=0.9, max_length=10, min_length=1)
    policy._load_image_tensor = lambda image_path: FakeTensor()

    item = SimpleNamespace(image="img/a.jpg", image_path="/tmp/a.jpg")

    generated = policy.generate(item, k=1)

    assert [event[0] for event in policy.model.events] == ["eval", "generate", "train"]
    assert policy.model.events[1][1] is False
    assert policy.model.training is True
    assert generated == [
        GeneratedQA(
            image="img/a.jpg",
            image_path="/tmp/a.jpg",
            question="what might the person next to the suitcase be doing?",
            answer="waiting",
            logprob=-0.5,
        )
    ]
