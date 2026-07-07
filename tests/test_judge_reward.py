import pytest

from judge.reward import (
    apply_reward_penalties,
    batch_diagnostics,
    compute_pairwise_metrics,
    generic_answer_penalty,
    group_normalized_advantages,
    length_penalty,
    score_distribution,
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


def test_apply_reward_penalties_subtracts_and_clamps():
    rows = [
        {"question": "Is it benign?", "answer": "yes", "reward": 0.8},
        {"question": "Is it benign?", "answer": "cells", "reward": 0.2},
    ]

    rewards = apply_reward_penalties(
        rows,
        duplicate_question_penalty=0.3,
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
    assert diagnostics["duplicate_question_rate"] == pytest.approx(0.5)
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
