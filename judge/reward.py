from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, pstdev
from typing import Mapping, Sequence

from dataset_adapters.generic_vqa import normalize_answer
from judge.data import GENERIC_MEDICAL_ANSWERS


def compute_pairwise_metrics(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
) -> dict[str, float]:
    if len(positive_scores) != len(negative_scores):
        raise ValueError("positive_scores and negative_scores must have the same length")
    if not positive_scores:
        return {
            "pairwise_accuracy": 0.0,
            "tie_rate": 0.0,
            "mean_margin": 0.0,
            "auroc": 0.0,
        }

    margins = [float(pos) - float(neg) for pos, neg in zip(positive_scores, negative_scores)]
    pair_count = len(margins)
    wins = sum(margin > 0 for margin in margins)
    ties = sum(margin == 0 for margin in margins)

    return {
        "pairwise_accuracy": wins / pair_count,
        "tie_rate": ties / pair_count,
        "mean_margin": sum(margins) / pair_count,
        "auroc": _binary_auroc(
            [1] * len(positive_scores) + [0] * len(negative_scores),
            [float(score) for score in positive_scores] + [float(score) for score in negative_scores],
        ),
    }


def _binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = [float(score) for label, score in zip(labels, scores) if int(label) == 1]
    negatives = [float(score) for label, score in zip(labels, scores) if int(label) == 0]
    if not positives or not negatives:
        return 0.0

    wins = 0.0
    total = len(positives) * len(negatives)
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / total


def score_distribution(rows: Sequence[Mapping]) -> dict[str, dict[str, dict[str, int]]]:
    by_question_prefix: dict[str, Counter[str]] = defaultdict(Counter)
    by_answer_type: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        raw_score = row["score"]
        if isinstance(raw_score, bool):
            raise TypeError("score must be an integer-valued number, not bool")
        if isinstance(raw_score, float):
            if not raw_score.is_integer():
                raise ValueError("score must be integer-valued")
            score = str(int(raw_score))
        elif isinstance(raw_score, int):
            score = str(raw_score)
        else:
            score = str(int(raw_score))
        by_question_prefix[str(row.get("question_prefix", "unknown"))][score] += 1
        by_answer_type[str(row.get("answer_type", "unknown"))][score] += 1

    return {
        "by_question_prefix": {key: dict(counter) for key, counter in by_question_prefix.items()},
        "by_answer_type": {key: dict(counter) for key, counter in by_answer_type.items()},
    }


def group_normalized_advantages(rewards: Sequence[float], eps: float = 1e-6) -> list[float]:
    if not rewards:
        return []

    values = [float(reward) for reward in rewards]
    reward_mean = mean(values)
    reward_std = pstdev(values)
    if reward_std <= eps:
        return [0.0 for _ in values]
    return [(value - reward_mean) / (reward_std + eps) for value in values]


def length_penalty(text: str, max_words: int, penalty: float) -> float:
    word_count = len(normalize_answer(text).split())
    return float(penalty) if word_count > int(max_words) else 0.0


def generic_answer_penalty(answer: str, penalty: float) -> float:
    normalized = normalize_answer(answer)
    return float(penalty) if normalized in GENERIC_MEDICAL_ANSWERS else 0.0


def duplicate_question_penalties(questions: Sequence[str], penalty: float) -> list[float]:
    seen_counts: Counter[str] = Counter()
    penalties: list[float] = []

    for question in questions:
        normalized = normalize_answer(question)
        penalties.append(float(penalty) * seen_counts[normalized])
        seen_counts[normalized] += 1

    return penalties


def yes_no_answer_penalties(answers: Sequence[str], penalty: float) -> list[float]:
    normalized_answers = [normalize_answer(answer) for answer in answers]
    if not normalized_answers:
        return []

    yes_no_rate = sum(answer in {"yes", "no"} for answer in normalized_answers) / len(
        normalized_answers
    )
    return [
        float(penalty) * yes_no_rate if answer in {"yes", "no"} else 0.0
        for answer in normalized_answers
    ]


def apply_reward_penalties(
    rows: Sequence[Mapping],
    *,
    duplicate_question_penalty: float = 0.0,
    yes_no_answer_penalty: float = 0.0,
    max_question_words: int = 30,
    max_answer_words: int = 12,
    length_penalty_value: float = 0.0,
    generic_penalty: float = 0.0,
) -> list[float]:
    duplicate_penalties = duplicate_question_penalties(
        [str(row.get("question", "")) for row in rows],
        duplicate_question_penalty,
    )
    yes_no_penalties = yes_no_answer_penalties(
        [str(row.get("answer", "")) for row in rows],
        yes_no_answer_penalty,
    )
    adjusted_rewards: list[float] = []

    for row, duplicate_penalty, yes_no_penalty in zip(
        rows, duplicate_penalties, yes_no_penalties
    ):
        base_reward = float(row.get("reward", 0.0))
        total_penalty = duplicate_penalty + yes_no_penalty
        total_penalty += length_penalty(
            str(row.get("question", "")),
            max_words=max_question_words,
            penalty=length_penalty_value,
        )
        total_penalty += length_penalty(
            str(row.get("answer", "")),
            max_words=max_answer_words,
            penalty=length_penalty_value,
        )
        total_penalty += generic_answer_penalty(
            str(row.get("answer", "")),
            penalty=generic_penalty,
        )
        adjusted_rewards.append(min(1.0, max(0.0, base_reward - total_penalty)))

    return adjusted_rewards


def batch_diagnostics(rows: Sequence[Mapping]) -> dict[str, float]:
    if not rows:
        return {
            "mean_reward": 0.0,
            "reward_std": 0.0,
            "duplicate_question_rate": 0.0,
            "yes_no_answer_rate": 0.0,
            "generic_answer_rate": 0.0,
        }

    rewards = [float(row.get("reward", 0.0)) for row in rows]
    normalized_questions = [normalize_answer(str(row.get("question", ""))) for row in rows]
    normalized_answers = [normalize_answer(str(row.get("answer", ""))) for row in rows]
    question_counts = Counter(normalized_questions)
    duplicate_count = sum(question_counts[question] > 1 for question in normalized_questions)

    return {
        "mean_reward": mean(rewards),
        "reward_std": pstdev(rewards) if len(rewards) > 1 else 0.0,
        "duplicate_question_rate": duplicate_count / len(rows),
        "yes_no_answer_rate": sum(answer in {"yes", "no"} for answer in normalized_answers) / len(rows),
        "generic_answer_rate": sum(answer in GENERIC_MEDICAL_ANSWERS for answer in normalized_answers) / len(rows),
    }
