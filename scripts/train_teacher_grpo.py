from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf

from judge.data import ImagePoolItem, load_image_pool
from judge.factory import resolve_prometheus_config, write_judge_snapshot
from judge.prometheus import DEFAULT_RUBRIC, PrometheusVisionScorer, StaticPrometheusScorer
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


def _clean_generated_question(question: str) -> str:
    question = question.strip()
    question = re.sub(r"^\s*:+\s*", "", question)
    question = re.sub(r"^\s*question\s*:\s*", "", question, flags=re.IGNORECASE)
    return question.strip()


def _select_generated_answer(answer: str) -> str:
    parts = [part.strip().strip(".") for part in answer.split(",")]
    parts = [part for part in parts if part]
    if not parts:
        return "unknown"

    counts: dict[str, int] = {}
    first_text: dict[str, str] = {}
    first_index: dict[str, int] = {}
    for index, part in enumerate(parts):
        key = part.lower()
        counts[key] = counts.get(key, 0) + 1
        first_text.setdefault(key, part)
        first_index.setdefault(key, index)

    best_key = min(
        counts,
        key=lambda key: (-counts[key], first_index[key]),
    )
    return first_text[best_key]


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
        answer_marker = re.search(r"\banswer\s*:", text, flags=re.IGNORECASE)
        if answer_marker is not None:
            before = text[: answer_marker.start()]
            after = text[answer_marker.end() :]
            question = before.strip()
            if question.lower().startswith("question:"):
                question = question[len("question:") :].strip()
            answer = after.strip().strip(".")
            return (
                _clean_generated_question(question or text.strip()),
                _select_generated_answer(answer),
            )
        return _clean_generated_question(text.strip()), "unknown"
    answer = ",".join(record.answer).strip()
    return _clean_generated_question(record.question), _select_generated_answer(answer)


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
        self.resume_step = 0
        resume_from = OmegaConf.select(config, "teacher.resume_from", default=None)
        if resume_from:
            checkpoint = torch.load(str(resume_from), map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.resume_step = int(checkpoint.get("step", 0))
            print(
                f"[grpo] resumed_checkpoint={resume_from} step={self.resume_step}",
                file=sys.stderr,
                flush=True,
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
        was_training = bool(getattr(self.model, "training", True))
        self.model.eval()
        try:
            with self.torch.no_grad():
                outputs, logprobs = self.model.generate(
                    repeated,
                    sample=True,
                    top_p=float(self.generation.top_p),
                    max_length=int(self.generation.max_length),
                    min_length=int(self.generation.min_length),
                    return_logprob=True,
                )
        finally:
            self.model.train(was_training)
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

    def save_checkpoint(self, path: str | Path, config, steps: int) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": config,
                "step": steps,
            },
            path,
        )


def build_judge_from_config(config):
    judge_model_path = OmegaConf.select(
        config, "reward.selected_judge_model_path", default=None
    )
    if judge_model_path is None:
        return StaticPrometheusScorer(score=3, feedback="mock")
    judge_config_path = OmegaConf.select(config, "reward.judge_config", default=None)
    if judge_config_path is None:
        raise ValueError("reward.judge_config is required when selected_judge_model_path is set")
    resolved = resolve_prometheus_config(
        judge_config_path,
        selected_model_path=str(judge_model_path),
        device=str(OmegaConf.select(config, "teacher.device", default="cuda")),
    )
    snapshot_path = OmegaConf.select(config, "logging.judge_snapshot_json", default=None)
    if snapshot_path:
        write_judge_snapshot(snapshot_path, resolved)

    return PrometheusVisionScorer(
        model_path=resolved.model_path,
        model_base=resolved.model_base,
        conv_mode=resolved.conv_mode,
        device=resolved.device,
        temperature=resolved.temperature,
        max_new_tokens=resolved.max_new_tokens,
        rubric=resolved.rubric,
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
        yes_no_answer_penalty=float(reward_cfg.get("yes_no_answer_penalty", 0.0)),
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


def _print_grpo_progress(
    *,
    step: int,
    total_steps: int,
    epoch: int,
    epochs: int,
    candidate_count: int,
    diagnostics: dict,
) -> None:
    print(
        "[grpo] "
        f"step={step + 1}/{total_steps} "
        f"epoch={epoch + 1}/{epochs} "
        f"candidates={candidate_count} "
        f"mean_reward={diagnostics.get('mean_reward', 0.0):.4f} "
        f"reward_std={diagnostics.get('reward_std', 0.0):.4f} "
        f"policy_loss={diagnostics.get('policy_loss', 0.0):.6f} "
        f"dup_rate={diagnostics.get('duplicate_question_rate', 0.0):.2f} "
        f"yes_no_rate={diagnostics.get('yes_no_answer_rate', 0.0):.2f} "
        f"generic_rate={diagnostics.get('generic_answer_rate', 0.0):.2f}",
        file=sys.stderr,
        flush=True,
    )


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
    epochs = int(OmegaConf.select(config, "teacher.epochs", default=1))
    if epochs < 1:
        raise ValueError("teacher.epochs must be >= 1")
    candidate_count = int(config.teacher.candidates_per_image)
    epoch_items = pool[:max_steps]
    total_steps = len(epoch_items) * epochs
    steps = int(getattr(teacher, "resume_step", 0))
    save_every = int(OmegaConf.select(config, "teacher.save_every", default=0))
    print(
        "[grpo] "
        f"start images_per_epoch={len(epoch_items)} "
        f"epochs={epochs} "
        f"total_steps={total_steps} "
        f"candidates_per_image={candidate_count} "
        f"metrics={metrics_path} "
        f"candidates={candidates_path}",
        file=sys.stderr,
        flush=True,
    )

    for epoch in range(epochs):
        for image_index, item in enumerate(epoch_items):
            global_index = epoch * len(epoch_items) + image_index
            if global_index < steps:
                continue
            rows = score_candidate_group(
                teacher.generate(item, k=candidate_count),
                judge=judge,
                reward_cfg=dict(config.reward),
            )
            update_metrics = teacher.update(rows)
            diagnostics = batch_diagnostics(rows)
            diagnostics.update(update_metrics)
            diagnostics["step"] = steps
            diagnostics["epoch"] = epoch
            diagnostics["image_index"] = image_index
            _append_jsonl(candidates_path, rows)
            _append_jsonl(metrics_path, [diagnostics])
            _print_grpo_progress(
                step=steps,
                total_steps=total_steps,
                epoch=epoch,
                epochs=epochs,
                candidate_count=len(rows),
                diagnostics=diagnostics,
            )
            steps += 1
            if (
                save_every > 0
                and steps % save_every == 0
                and hasattr(teacher, "save_checkpoint")
            ):
                checkpoint_path = (
                    Path(str(config.teacher.output_dir)) / f"checkpoint_step_{steps:06d}.pth"
                )
                teacher.save_checkpoint(checkpoint_path, config, steps)
                print(
                    f"[grpo] saved_checkpoint={checkpoint_path}",
                    file=sys.stderr,
                    flush=True,
                )

    final_checkpoint = Path(str(config.teacher.output_dir)) / "checkpoint_final.pth"
    final_already_saved = save_every > 0 and steps > 0 and steps % save_every == 0
    if hasattr(teacher, "save_checkpoint") and not final_already_saved:
        teacher.save_checkpoint(final_checkpoint, config, steps)
        print(f"[grpo] saved_checkpoint={final_checkpoint}", file=sys.stderr, flush=True)

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
