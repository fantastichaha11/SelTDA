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


def advantage_weighted_policy_loss(
    losses: Sequence[float], advantages: Sequence[float]
) -> float:
    if len(losses) != len(advantages):
        raise ValueError("losses and advantages must have the same length")
    if not losses:
        return 0.0
    return sum(
        float(loss) * float(advantage)
        for loss, advantage in zip(losses, advantages)
    ) / len(losses)


def _parse_generated_qa(text: str) -> tuple[str, str]:
    from generate_questions import (
        AmbiguousBooleanAnswerError,
        ParseModelOutputError,
        VQARecord,
    )

    try:
        record = VQARecord.build_from_raw_model_output(
            text,
            image_path="/synthetic/grpo/image.jpg",
            parse_rationale=False,
        )
    except (AmbiguousBooleanAnswerError, ParseModelOutputError):
        lower = text.lower()
        marker = "answer:"
        if marker in lower:
            index = lower.index(marker)
            before = text[:index]
            after = text[index + len(marker) :]
            question = before.strip()
            if question.lower().startswith("question:"):
                question = question[len("question:") :].strip()
            answer = after.strip().strip(".")
            return question or text.strip(), answer or "unknown"
        return text.strip(), "unknown"
    answer = ",".join(record.answer).strip()
    return record.question, answer or "unknown"


class BlipTeacherPolicy:
    def __init__(self, config):
        import torch
        from PIL import Image
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode

        from models.blip import decoder_from_config

        self.torch = torch
        self.Image = Image
        self.device = str(config.teacher.device)
        teacher_cfg = OmegaConf.load(str(config.teacher.config))
        if OmegaConf.select(config, "teacher.pretrained", default=None):
            teacher_cfg.pretrained = str(config.teacher.pretrained)
        self.model = decoder_from_config(teacher_cfg).to(self.device)
        self.model.train()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=float(config.teacher.lr)
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (int(teacher_cfg.image_size), int(teacher_cfg.image_size)),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.generation = config.generation

    def _load_image_tensor(self, image_path: str):
        image = self.Image.open(image_path).convert("RGB")
        return self.transform(image).unsqueeze(0).to(self.device)

    def generate(self, item: ImagePoolItem, k: int) -> list[GeneratedQA]:
        image = self._load_image_tensor(item.image_path)
        repeated = image.repeat(k, 1, 1, 1)
        with self.torch.no_grad():
            outputs, logprobs = self.model.generate(
                repeated,
                sample=True,
                top_p=float(self.generation.top_p),
                max_length=int(self.generation.max_length),
                min_length=int(self.generation.min_length),
                return_logprob=True,
            )
        generated: list[GeneratedQA] = []
        for output, logprob in zip(outputs, logprobs):
            question, answer = _parse_generated_qa(output)
            generated.append(
                GeneratedQA(
                    image=item.image,
                    image_path=item.image_path,
                    question=question,
                    answer=answer,
                    logprob=float(logprob),
                )
            )
        return generated

    def update(self, rows: Sequence[dict]) -> dict[str, float]:
        if not rows:
            return {"policy_loss": 0.0}
        captions = [
            f"Question: {row['question']} Answer: {row['answer']}" for row in rows
        ]
        images = self.torch.cat(
            [self._load_image_tensor(str(row["image_path"])) for row in rows], dim=0
        )
        advantages = self.torch.tensor(
            [float(row["advantage"]) for row in rows], device=self.device
        )
        losses = []
        for index, caption in enumerate(captions):
            loss = self.model(images[index : index + 1], [caption])
            losses.append(loss)
        stacked = self.torch.stack(losses)
        policy_loss = (stacked * advantages).mean()
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        return {
            "policy_loss": advantage_weighted_policy_loss(
                [float(loss.detach().cpu()) for loss in losses],
                [float(row["advantage"]) for row in rows],
            )
        }


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
    if teacher is None:
        teacher = (
            MockTeacherPolicy()
            if bool(config.teacher.dry_run)
            else BlipTeacherPolicy(config)
        )
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
