from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol


NO_REFERENCE_TEXT = (
    "No reference answer is provided. Judge only from the image, question, and candidate answer."
)

DEFAULT_RUBRIC = """Score 1: The answer is incorrect, unsupported by the image, or does not address the question.
Score 2: The answer is mostly incorrect, overly generic, or weakly grounded in the image.
Score 3: The answer is partially correct and somewhat grounded, but incomplete or imprecise.
Score 4: The answer is mostly correct, grounded in the image, and addresses the question well.
Score 5: The answer is fully correct, specific, and well grounded in the image."""


@dataclass(frozen=True)
class PrometheusResult:
    score: int
    reward: float
    feedback: str
    raw_text: str | None = None


class PrometheusScorer(Protocol):
    def score(
        self,
        image_path: str,
        question: str,
        candidate_answer: str,
    ) -> PrometheusResult: ...


def build_prometheus_prompt(
    question: str,
    candidate_answer: str,
    rubric: str = DEFAULT_RUBRIC,
) -> str:
    return (
        "###Task Description:\n"
        "Answer the visual question about this pathology image:\n"
        f"{question}\n\n"
        "###Scoring Rubric:\n"
        f"{rubric}\n\n"
        "###Reference Answer (Score 5):\n"
        f"{NO_REFERENCE_TEXT}\n\n"
        "###Candidate Answer:\n"
        f"{candidate_answer}\n\n"
        "###Feedback:"
    )


def parse_prometheus_score(raw_text: str) -> int:
    patterns = (
        r"\[RESULT\]\s*([1-5])\b",
        r"\bScore\s*:\s*([1-5])\b",
        r"\bscore\s+is\s+([1-5])\b",
    )
    for pattern in patterns:
        match = re.search(pattern, raw_text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    raise ValueError("Unable to parse Prometheus score from response.")


def score_to_reward(score: int) -> float:
    if score < 1 or score > 5:
        raise ValueError(f"Prometheus score must be in [1, 5], got {score}.")
    return (score - 1) / 4


def _feedback_from_raw(raw_text: str) -> str:
    text = raw_text.strip()
    result_match = re.search(r"\[RESULT\]\s*[1-5]\b", text, flags=re.IGNORECASE)
    if result_match:
        text = text[: result_match.start()].rstrip()
    text = re.sub(r"^Feedback:\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


class StaticPrometheusScorer:
    def __init__(self, score: int, feedback: str = ""):
        self._score = int(score)
        self._feedback = feedback

    def score(
        self,
        image_path: str,
        question: str,
        candidate_answer: str,
    ) -> PrometheusResult:
        del image_path, question, candidate_answer
        return PrometheusResult(
            score=self._score,
            reward=score_to_reward(self._score),
            feedback=self._feedback,
            raw_text=self._feedback,
        )


class PrometheusVisionScorer:
    def __init__(
        self,
        model_name: str = "kaist-ai/prometheus-vision-13b-v1.0",
        rubric: str = DEFAULT_RUBRIC,
        max_new_tokens: int = 256,
    ):
        self.model_name = model_name
        self.rubric = rubric
        self.max_new_tokens = max_new_tokens
        self._model: Any | None = None
        self._processor: Any | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        from PIL import Image  # noqa: F401
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self._model = AutoModelForVision2Seq.from_pretrained(self.model_name)
        self._processor = AutoProcessor.from_pretrained(self.model_name)

    def _generate_raw(self, image_path: str, prompt: str) -> str:
        self._ensure_loaded()

        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, text=prompt, return_tensors="pt")
        output_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        decoded = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        if "###Feedback:" in decoded:
            return decoded.rsplit("###Feedback:", maxsplit=1)[-1].strip()
        return decoded.strip()

    def score(
        self,
        image_path: str,
        question: str,
        candidate_answer: str,
    ) -> PrometheusResult:
        prompt = build_prometheus_prompt(
            question=question,
            candidate_answer=candidate_answer,
            rubric=self.rubric,
        )
        raw_text = self._generate_raw(image_path=image_path, prompt=prompt)
        score = parse_prometheus_score(raw_text)
        return PrometheusResult(
            score=score,
            reward=score_to_reward(score),
            feedback=_feedback_from_raw(raw_text),
            raw_text=raw_text,
        )


__all__ = [
    "DEFAULT_RUBRIC",
    "NO_REFERENCE_TEXT",
    "PrometheusResult",
    "PrometheusScorer",
    "PrometheusVisionScorer",
    "StaticPrometheusScorer",
    "build_prometheus_prompt",
    "parse_prometheus_score",
    "score_to_reward",
]
