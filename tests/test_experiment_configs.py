from omegaconf import OmegaConf

from judge.factory import resolve_prometheus_config
from judge.prometheus import DEFAULT_RUBRIC


def test_old_pathvqa_rubric_is_explicit_but_unchanged():
    cfg = OmegaConf.load("configs/prometheus_judge_pathvqa.yaml")
    resolved = resolve_prometheus_config("configs/prometheus_judge_pathvqa.yaml")

    assert int(cfg.prompt.criteria_count) == 4
    assert resolved.criteria_count == 4
    assert resolved.rubric == DEFAULT_RUBRIC


def test_old_vizwiz_rubric_is_explicit_and_four_criteria():
    cfg = OmegaConf.load("configs/prometheus_judge_vizwiz.yaml")
    resolved = resolve_prometheus_config("configs/prometheus_judge_vizwiz.yaml")

    assert int(cfg.prompt.criteria_count) == 4
    assert resolved.criteria_count == 4
    assert "blind or low-vision user" in resolved.rubric
    assert "Score 1:" in resolved.rubric and "Score 5:" in resolved.rubric


def test_rubric8_configs_have_eight_explicit_criteria():
    for dataset in ("pathvqa", "vizwiz"):
        resolved = resolve_prometheus_config(
            f"configs/prometheus_judge_{dataset}_rubric8.yaml"
        )
        assert resolved.criteria_count == 8
        assert "eight criteria" in resolved.rubric.lower()
        assert "Score 1:" in resolved.rubric and "Score 5:" in resolved.rubric


def test_grpo_configs_isolate_reward_changes():
    keys = (
        "duplicate_question_penalty",
        "yes_no_answer_penalty",
        "length_penalty",
        "generic_answer_penalty",
    )
    for dataset in ("pathvqa", "vizwiz"):
        judge_only = OmegaConf.load(
            f"configs/grpo_teacher_{dataset}_rubric4_judge_only.yaml"
        )
        rubric8 = OmegaConf.load(
            f"configs/grpo_teacher_{dataset}_rubric8_penalties.yaml"
        )
        assert judge_only.teacher.candidates_per_image == 8
        assert rubric8.teacher.candidates_per_image == 8
        assert judge_only.teacher.max_steps == 200
        assert rubric8.teacher.max_steps == 200
        assert judge_only.teacher.epochs == 3
        assert rubric8.teacher.epochs == 3
        assert [float(judge_only.reward[key]) for key in keys] == [
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        assert [float(rubric8.reward[key]) for key in keys] == [
            0.35,
            0.35,
            0.10,
            0.15,
        ]
        assert str(judge_only.logging.judge_snapshot_json).startswith(
            f"outputs/experiments/{dataset}/pt4_j/teacher/"
        )
        assert str(rubric8.logging.judge_snapshot_json).startswith(
            f"outputs/experiments/{dataset}/pt8_jp/teacher/"
        )
