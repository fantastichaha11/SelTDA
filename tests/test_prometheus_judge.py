import pytest

from judge import DEFAULT_RUBRIC
from judge.prometheus import (
    NO_REFERENCE_TEXT,
    StaticPrometheusScorer,
    build_prometheus_prompt,
    parse_prometheus_score,
    score_to_reward,
)


def test_build_prometheus_prompt_uses_no_reference_template():
    prompt = build_prometheus_prompt(
        question="What abnormality is visible?",
        candidate_answer="A necrotic tumor is present.",
    )

    assert "###Task Description:" in prompt
    assert "###Reference Answer (Score 5):" in prompt
    assert NO_REFERENCE_TEXT in prompt
    assert "Answer the visual question about this pathology image:\nWhat abnormality is visible?" in prompt
    assert "A necrotic tumor is present." in prompt
    assert "ground truth" not in prompt.lower()
    assert "adenocarcinoma" not in prompt


def test_judge_package_reexports_default_rubric():
    assert "Score 5" in DEFAULT_RUBRIC


def test_parse_prometheus_score_parses_result_tag():
    assert parse_prometheus_score("Feedback: The answer is grounded. [RESULT] 4") == 4


def test_parse_prometheus_score_parses_score_label():
    assert parse_prometheus_score("Feedback: too generic. Score: 2") == 2


def test_parse_prometheus_score_raises_when_missing():
    with pytest.raises(ValueError):
        parse_prometheus_score("Feedback only. No numeric result provided.")


@pytest.mark.parametrize(
    ("score", "reward"),
    [
        (1, 0.0),
        (3, 0.5),
        (5, 1.0),
    ],
)
def test_score_to_reward_maps_linearly(score, reward):
    assert score_to_reward(score) == reward


def test_static_prometheus_scorer_returns_deterministic_result():
    result = StaticPrometheusScorer(score=4, feedback="ok").score(
        image_path="/tmp/slide.png",
        question="What is shown?",
        candidate_answer="Tumor cells.",
    )

    assert result.score == 4
    assert result.reward == 0.75
    assert result.feedback == "ok"
