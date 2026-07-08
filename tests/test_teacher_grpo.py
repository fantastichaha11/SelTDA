import json

from omegaconf import OmegaConf

from judge.prometheus import StaticPrometheusScorer
from scripts.train_teacher_grpo import (
    GeneratedQA,
    MockTeacherPolicy,
    run_grpo,
    score_candidate_group,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_score_candidate_group_adds_rewards_and_advantages():
    candidates = [
        GeneratedQA(
            image="img/a.jpg",
            image_path="/tmp/a.jpg",
            question="Q1?",
            answer="nuclei",
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
    assert rows[0]["reward"] == 0.65
    assert rows[1]["reward"] == 0.55
    assert rows[0]["advantage"] > rows[1]["advantage"]


def test_run_grpo_uses_train_images_without_ground_truth_qa(tmp_path):
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
                "candidates_jsonl": str(tmp_path / "out" / "candidates.jsonl"),
                "metrics_jsonl": str(tmp_path / "out" / "metrics.jsonl"),
            },
            "seed": 42,
        }
    )
    result = run_grpo(
        cfg,
        teacher=MockTeacherPolicy(),
        judge=StaticPrometheusScorer(score=5, feedback="mock"),
    )
    assert result["steps"] == 1
    candidates_text = (tmp_path / "out" / "candidates.jsonl").read_text(
        encoding="utf-8"
    )
    assert "Gold question?" not in candidates_text
    assert '"reward": 1.0' in candidates_text
