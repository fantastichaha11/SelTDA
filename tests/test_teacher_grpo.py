import json
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

from judge.prometheus import StaticPrometheusScorer
from scripts.train_teacher_grpo import (
    GeneratedQA,
    MockTeacherPolicy,
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
