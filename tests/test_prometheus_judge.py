import pytest

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
    assert (
        "An instruction, a response to evaluate, an image, a score rubric, and a neutral no-reference field are given."
        in prompt
    )
    assert "1. Write detailed feedback that assesses the response strictly based on the score rubric." in prompt
    assert "2. After writing feedback, write a score that is an integer between 1 and 5." in prompt
    assert (
        "3. The output format should be: Feedback: (feedback) [RESULT] (integer number between 1 and 5)"
        in prompt
    )
    assert "4. Do not generate any other opening, closing, or explanation." in prompt
    assert "###The instruction to evaluate:" in prompt
    assert "###Response to evaluate:" in prompt
    assert "###Reference Answer (Score 5):" in prompt
    assert "###Score Rubrics:" in prompt
    assert NO_REFERENCE_TEXT in prompt
    assert "Answer the visual question about this pathology image:\nWhat abnormality is visible?" in prompt
    assert "A necrotic tumor is present." in prompt
    assert "ground truth" not in prompt.lower()
    assert prompt.endswith("###Feedback:")


def test_parse_prometheus_score_parses_result_tag():
    assert parse_prometheus_score("Feedback: The answer is grounded. [RESULT] 4") == 4


def test_parse_prometheus_score_parses_score_label():
    assert parse_prometheus_score("Feedback: too generic. Score: 2") == 2


def test_parse_prometheus_score_parses_score_is_form():
    assert parse_prometheus_score("Feedback: grounded, score is 5") == 5


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
    assert result.raw_text == "Feedback: ok [RESULT] 4"
