import json

from omegaconf import OmegaConf

from judge.prometheus import PrometheusResult, StaticPrometheusScorer, score_to_reward
from scripts.score_rank_prometheus_pool import run_filter


def test_filter_requires_exact_two_x_eligible_pool(tmp_path):
    real = tmp_path / "real.json"
    raw = tmp_path / "raw.json"
    images = tmp_path / "images"
    images.mkdir()
    real.write_text(json.dumps([{"image": "r1.jpg"}, {"image": "r2.jpg"}]), encoding="utf-8")
    rows = []
    for index in range(3):
        (images / f"{index}.jpg").touch()
        rows.append(
            {"image": f"{index}.jpg", "question": f"Q{index}?", "answer": ["a"]}
        )
    raw.write_text(json.dumps(rows), encoding="utf-8")
    cfg = OmegaConf.create(
        {
            "data": {
                "raw_pools": [str(raw)],
                "real_annotations": str(real),
                "image_root": str(images),
                "dataset": "pathvqa",
                "target_total": 4,
                "pool_multiplier": 2,
            },
            "output": {"dir": str(tmp_path / "out")},
            "seed": 42,
        }
    )

    try:
        run_filter(cfg, scorer=StaticPrometheusScorer(score=4, feedback="mock"))
    except ValueError as exc:
        assert "required eligible pool=4" in str(exc)
    else:
        raise AssertionError("undersized pool must fail")


class CountingScorer:
    def __init__(self):
        self.calls = 0

    def score(self, image_path: str, question: str, candidate_answer: str):
        del image_path, candidate_answer
        self.calls += 1
        score = 5 if question in {"Q0?", "Q1?"} else 3
        return PrometheusResult(
            score=score,
            reward=score_to_reward(score),
            feedback="counted",
            raw_text=f"Feedback: counted [RESULT] {score}",
        )


def test_filter_resumes_scores_and_writes_manifest(tmp_path):
    real = tmp_path / "real.json"
    raw = tmp_path / "raw.json"
    images = tmp_path / "images"
    output_dir = tmp_path / "out"
    images.mkdir()
    real.write_text(json.dumps([{"image": "r1.jpg"}, {"image": "r2.jpg"}]), encoding="utf-8")
    rows = []
    for index in range(4):
        (images / f"{index}.jpg").touch()
        rows.append(
            {"image": f"{index}.jpg", "question": f"Q{index}?", "answer": ["a"]}
        )
    raw.write_text(json.dumps(rows), encoding="utf-8")
    cfg = OmegaConf.create(
        {
            "data": {
                "raw_pools": [str(raw)],
                "real_annotations": str(real),
                "image_root": str(images),
                "dataset": "pathvqa",
                "target_total": 4,
                "pool_multiplier": 2,
            },
            "judge": {
                "config": "configs/prometheus_judge_pathvqa.yaml",
                "selected_model_path": None,
                "device": "cpu",
            },
            "output": {"dir": str(output_dir)},
            "seed": 42,
        }
    )

    first_scorer = CountingScorer()
    first = run_filter(cfg, scorer=first_scorer)
    second_scorer = CountingScorer()
    second = run_filter(cfg, scorer=second_scorer)

    selected = json.loads((output_dir / "selected.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert first == second == {"real": 2, "eligible_pool": 4, "synthetic": 2, "total": 4}
    assert first_scorer.calls == 4
    assert second_scorer.calls == 0
    assert len(selected) == 2
    assert manifest["pool_multiplier"] == 2
    assert manifest["eligible_pool_count"] == 4
    assert manifest["required_pool_count"] == 4
    assert manifest["synthetic_count"] == 2
    assert manifest["seed"] == 42
    assert manifest["per_image_quota"] is None
