from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf

from judge.data import ImagePoolItem, load_image_pool
from judge.prometheus import PrometheusVisionScorer, StaticPrometheusScorer
from judge.reward import (
    apply_reward_penalties,
    batch_diagnostics,
    group_normalized_advantages,
)


@dataclass(frozen=True)
class GeneratedQA:
    image: str
    image_path: str
    question: str
    answer: str
    logprob: float


class TeacherPolicy(Protocol):
    def generate(self, item: ImagePoolItem, k: int) -> list[GeneratedQA]:
        ...

    def update(self, rows: Sequence[dict]) -> dict[str, float]:
        ...


class MockTeacherPolicy:
    def generate(self, item: ImagePoolItem, k: int) -> list[GeneratedQA]:
        return [
            GeneratedQA(
                image=item.image,
                image_path=item.image_path,
                question=f"Generated question {idx}?",
                answer="nuclei" if idx == 0 else "cells",
                logprob=-float(idx + 1),
            )
            for idx in range(k)
        ]

    def update(self, rows: Sequence[dict]) -> dict[str, float]:
        del rows
        return {"policy_loss": 0.0}


def build_judge_from_config(config):
    judge_model_path = OmegaConf.select(
        config, "reward.selected_judge_model_path", default=None
    )
    if judge_model_path is None:
        return StaticPrometheusScorer(score=3, feedback="mock")
    return PrometheusVisionScorer(
        model_path=str(judge_model_path),
        device=str(OmegaConf.select(config, "teacher.device", default="cuda")),
    )


def score_candidate_group(candidates: Sequence[GeneratedQA], judge, reward_cfg) -> list[dict]:
    rows: list[dict] = []
    for candidate in candidates:
        result = judge.score(candidate.image_path, candidate.question, candidate.answer)
        row = asdict(candidate)
        row["judge_score"] = result.score
        row["reward"] = result.reward
        row["feedback"] = result.feedback
        rows.append(row)

    adjusted_rewards = apply_reward_penalties(
        rows,
        duplicate_question_penalty=float(reward_cfg.get("duplicate_question_penalty", 0.0)),
        max_question_words=int(reward_cfg.get("max_question_words", 30)),
        max_answer_words=int(reward_cfg.get("max_answer_words", 12)),
        length_penalty_value=float(reward_cfg.get("length_penalty", 0.0)),
        generic_penalty=float(reward_cfg.get("generic_answer_penalty", 0.0)),
    )
    advantages = group_normalized_advantages(adjusted_rewards)

    for row, reward, advantage in zip(rows, adjusted_rewards, advantages):
        row["reward"] = reward
        row["advantage"] = advantage
    return rows


def _append_jsonl(path: str | Path, rows: Sequence[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_grpo(config, teacher: TeacherPolicy | None = None, judge=None) -> dict[str, int]:
    random.seed(int(OmegaConf.select(config, "seed", default=42)))
    pool = load_image_pool(config.image_pool)
    teacher = teacher or MockTeacherPolicy()
    judge = judge or build_judge_from_config(config)

    candidates_path = Path(str(config.logging.candidates_jsonl))
    metrics_path = Path(str(config.logging.metrics_jsonl))
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    candidates_path.write_text("", encoding="utf-8")
    metrics_path.write_text("", encoding="utf-8")

    max_steps = int(config.teacher.max_steps)
    candidate_count = int(config.teacher.candidates_per_image)
    steps = 0

    for item in pool[:max_steps]:
        rows = score_candidate_group(
            teacher.generate(item, k=candidate_count),
            judge=judge,
            reward_cfg=dict(config.reward),
        )
        update_metrics = teacher.update(rows)
        diagnostics = batch_diagnostics(rows)
        diagnostics.update(update_metrics)
        diagnostics["step"] = steps
        _append_jsonl(candidates_path, rows)
        _append_jsonl(metrics_path, [diagnostics])
        steps += 1

    return {"steps": steps}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/grpo_teacher_pathvqa_prometheus.yaml"
    )
    parser.add_argument("--mock-judge-score", type=int, default=None)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    judge = (
        StaticPrometheusScorer(score=args.mock_judge_score, feedback="mock")
        if args.mock_judge_score is not None
        else None
    )
    print(json.dumps(run_grpo(config, judge=judge), indent=2))


if __name__ == "__main__":
    main()
