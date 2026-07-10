import pytest

from judge.reward import (
    apply_reward_penalties,
    batch_diagnostics,
    compute_pairwise_metrics,
    duplicate_question_penalties,
    generic_answer_penalty,
    group_normalized_advantages,
    length_penalty,
    score_distribution,
    yes_no_answer_penalties,
)


def test_pairwise_metrics_count_ties_and_margins():
    metrics = compute_pairwise_metrics(
        positive_scores=[5, 3, 2],
        negative_scores=[1, 3, 4],
    )

    assert metrics["pairwise_accuracy"] == pytest.approx(1 / 3)
    assert metrics["tie_rate"] == pytest.approx(1 / 3)
    assert metrics["mean_margin"] == pytest.approx((4 + 0 - 2) / 3)
    assert metrics["auroc"] == pytest.approx(11 / 18)


def test_pairwise_metrics_raise_on_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        compute_pairwise_metrics(
            positive_scores=[1, 2],
            negative_scores=[1],
        )


def test_pairwise_metrics_return_zeros_on_empty_inputs():
    assert compute_pairwise_metrics([], []) == {
        "pairwise_accuracy": 0.0,
        "tie_rate": 0.0,
        "mean_margin": 0.0,
        "auroc": 0.0,
    }


def test_group_normalized_advantages_zero_mean():
    advantages = group_normalized_advantages([0.2, 0.6, 1.0])

    assert sum(advantages) == pytest.approx(0.0)
    assert advantages[1] == pytest.approx(0.0)


def test_group_normalized_advantages_handles_constant_rewards():
    assert group_normalized_advantages([0.5, 0.5]) == [0.0, 0.0]


def test_penalties_are_explicit_and_bounded():
    assert length_penalty("short answer", max_words=4, penalty=0.2) == 0.0
    assert length_penalty("one two three four five", max_words=4, penalty=0.2) == 0.2
    assert generic_answer_penalty("cells", penalty=0.15) == 0.15
    assert generic_answer_penalty("lymphocytes", penalty=0.15) == 0.0


def test_duplicate_penalty_increases_with_repeated_occurrences():
    penalties = duplicate_question_penalties(["Q?", "Q?", "Other?", "Q?"], penalty=0.2)

    assert penalties == pytest.approx([0.0, 0.2, 0.0, 0.4])


def test_yes_no_penalty_scales_with_group_yes_no_rate():
    penalties = yes_no_answer_penalties(["yes", "no", "cells", "tumor"], penalty=0.4)

    assert penalties == pytest.approx([0.2, 0.2, 0.0, 0.0])


def test_apply_reward_penalties_subtracts_and_clamps():
    rows = [
        {"question": "Is it benign?", "answer": "yes", "reward": 0.8},
        {"question": "Is it benign?", "answer": "cells", "reward": 0.2},
    ]

    rewards = apply_reward_penalties(
        rows,
        duplicate_question_penalty=0.3,
        yes_no_answer_penalty=0.0,
        max_question_words=10,
        max_answer_words=3,
        length_penalty_value=0.1,
        generic_penalty=0.15,
    )

    assert rewards == pytest.approx([0.8, 0.0])


def test_batch_diagnostics_reports_rates():
    rows = [
        {"question": "Is it benign?", "answer": "yes", "reward": 0.8},
        {"question": "Is it benign?", "answer": "cells", "reward": 0.2},
    ]

    diagnostics = batch_diagnostics(rows)

    assert diagnostics["mean_reward"] == pytest.approx(0.5)
    assert diagnostics["duplicate_question_rate"] == pytest.approx(1.0)
    assert diagnostics["yes_no_answer_rate"] == pytest.approx(0.5)
    assert diagnostics["generic_answer_rate"] == pytest.approx(0.5)


def test_score_distribution_groups_by_prefix_and_answer_type():
    rows = [
        {"question_prefix": "is/are", "answer_type": "yes/no", "score": 5},
        {"question_prefix": "is/are", "answer_type": "yes/no", "score": 1},
        {"question_prefix": "what", "answer_type": "phrase", "score": 4},
    ]

    distribution = score_distribution(rows)

    assert distribution["by_question_prefix"]["is/are"] == {"1": 1, "5": 1}
    assert distribution["by_answer_type"]["phrase"] == {"4": 1}


def test_score_distribution_canonicalizes_numeric_score_keys():
    rows = [
        {"question_prefix": "what", "answer_type": "phrase", "score": 4.0},
    ]

    distribution = score_distribution(rows)

    assert distribution["by_question_prefix"]["what"] == {"4": 1}
    assert distribution["by_answer_type"]["phrase"] == {"4": 1}


@pytest.mark.parametrize(
    "row",
    [
        {"question_prefix": "what", "answer_type": "phrase"},
        {"question_prefix": "what", "answer_type": "phrase", "score": "bad"},
        {"question_prefix": "what", "answer_type": "phrase", "score": 4.9},
        {"question_prefix": "what", "answer_type": "phrase", "score": True},
    ],
)
def test_score_distribution_rejects_missing_or_invalid_scores(row):
    with pytest.raises((KeyError, TypeError, ValueError)):
        score_distribution([row])
