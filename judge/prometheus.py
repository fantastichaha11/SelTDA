from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol


NO_REFERENCE_TEXT = (
    "No reference answer is provided. Judge only from the image, question, and candidate answer."
)

DEFAULT_RUBRIC = """You are evaluating a candidate answer for a pathology image question.
Judge the answer using only the pathology image, the question, and the candidate answer.
Focus on visual grounding in the image, medical correctness, specificity, and practical usefulness for the question.
Penalize hallucinated details, unrelated content, vague answers, and medically misleading claims.

Score 1: The answer is completely incorrect, unrelated to the question, not grounded in the image, or medically misleading.
Score 2: The answer has slight relevance but is mostly incorrect, weakly grounded, too generic, hallucinated, or substantially misleading.
Score 3: The answer is partially correct and somewhat grounded in the image, but it is incomplete, imprecise, or misses important medical detail.
Score 4: The answer is mostly correct, well grounded in the image, medically sound, and useful, with only minor omissions or imprecision.
Score 5: The answer is fully correct, clearly grounded in the image, medically accurate, specific, and maximally useful for the question."""


@dataclass(frozen=True)
class PrometheusResult:
    score: int
    reward: float
    feedback: str
    raw_text: str


class PrometheusScorer(Protocol):
    def score(
        self,
        image_path: str,
        question: str,
        candidate_answer: str,
    ) -> PrometheusResult: ...


def build_prometheus_prompt(
    *,
    question: str,
    candidate_answer: str,
    rubric: str = DEFAULT_RUBRIC,
    reference_answer: str = NO_REFERENCE_TEXT,
) -> str:
    instruction = "Answer the visual question about this pathology image:\n" + question
    return (
        "###Task Description:\n"
        "An instruction, a response to evaluate, an image, a score rubric, and a neutral no-reference field are given.\n"
        "1. Write detailed feedback that assesses the response strictly based on the score rubric.\n"
        "2. After writing feedback, write a score that is an integer between 1 and 5.\n"
        "3. The output format should be: Feedback: (feedback) [RESULT] (integer number between 1 and 5)\n"
        "4. Do not generate any other opening, closing, or explanation.\n\n"
        "###The instruction to evaluate:\n"
        f"{instruction}\n\n"
        "###Response to evaluate:\n"
        f"{candidate_answer}\n\n"
        "###Reference Answer (Score 5):\n"
        f"{reference_answer}\n\n"
        "###Score Rubrics:\n"
        f"{rubric}\n\n"
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
    text = re.sub(r"^\s*Feedback:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\s*(?:\[\s*RESULT\s*\]\s*[1-5]\b|Score\s*:\s*[1-5]\b|score\s+is\s+[1-5]\b)\s*$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip()


class StaticPrometheusScorer:
    def __init__(self, score: int = 3, feedback: str = "static"):
        self.static_score = int(score)
        self.feedback = feedback

    def score(
        self,
        image_path: str,
        question: str,
        candidate_answer: str,
    ) -> PrometheusResult:
        del image_path, question, candidate_answer
        raw = f"Feedback: {self.feedback} [RESULT] {self.static_score}"
        return PrometheusResult(
            score=self.static_score,
            reward=score_to_reward(self.static_score),
            feedback=self.feedback,
            raw_text=raw,
        )


class PrometheusVisionScorer:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        model_base: str | None = None,
        conv_mode: str = "vicuna_v1",
        temperature: float = 0.0,
        max_new_tokens: int = 512,
    ):
        self.model_path = model_path
        self.device = device
        self.model_base = model_base
        self.conv_mode = conv_mode
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self._loaded = False
        self._tokenizer: Any = None
        self._model: Any = None
        self._image_processor: Any = None
        self._context_len: Any = None
        self._torch: Any = None
        self._Image: Any = None
        self._IMAGE_TOKEN_INDEX: Any = None
        self._DEFAULT_IMAGE_TOKEN: Any = None
        self._conv_templates: Any = None
        self._get_model_name_from_path: Any = None
        self._process_images: Any = None
        self._tokenizer_image_token: Any = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        import torch
        from PIL import Image
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import (
            get_model_name_from_path,
            process_images,
            tokenizer_image_token,
        )
        from llava.model.builder import load_pretrained_model

        model_name = get_model_name_from_path(self.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            self.model_path,
            self.model_base,
            model_name,
            device=self.device,
        )

        self._torch = torch
        self._Image = Image
        self._IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self._DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self._conv_templates = conv_templates
        self._get_model_name_from_path = get_model_name_from_path
        self._process_images = process_images
        self._tokenizer_image_token = tokenizer_image_token
        self._tokenizer = tokenizer
        self._model = model
        self._image_processor = image_processor
        self._context_len = context_len
        self._loaded = True

    def generate(self, image_path: str, prompt: str) -> str:
        self._ensure_loaded()

        image = self._Image.open(image_path).convert("RGB")
        image_tensor = self._process_images(
            [image],
            self._image_processor,
            self._model.config,
        )
        if isinstance(image_tensor, list):
            image_tensor = [
                tensor.to(self._model.device, dtype=self._torch.float16)
                for tensor in image_tensor
            ]
        else:
            image_tensor = image_tensor.to(self._model.device, dtype=self._torch.float16)

        conv = self._conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], f"{self._DEFAULT_IMAGE_TOKEN}\n{prompt}")
        conv.append_message(conv.roles[1], None)
        full_prompt = conv.get_prompt()

        input_ids = self._tokenizer_image_token(
            full_prompt,
            self._tokenizer,
            self._IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(self._model.device)

        with self._torch.inference_mode():
            output_ids = self._model.generate(
                input_ids,
                images=image_tensor,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )

        generated_ids = output_ids
        input_token_count = input_ids.shape[1]
        if output_ids.shape[1] >= input_token_count:
            output_prefix = output_ids[:, :input_token_count]
            same_prefix = False
            if hasattr(output_prefix, "equal"):
                same_prefix = output_prefix.equal(input_ids)
            else:
                try:
                    same_prefix = output_prefix.tolist() == input_ids.tolist()
                except AttributeError:
                    same_prefix = False
            if same_prefix:
                generated_ids = output_ids[:, input_token_count:]

        return self._tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    def score(
        self,
        image_path: str,
        question: str,
        candidate_answer: str,
    ) -> PrometheusResult:
        prompt = build_prometheus_prompt(
            question=question,
            candidate_answer=candidate_answer,
        )
        raw_text = self.generate(image_path=image_path, prompt=prompt)
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
    "_feedback_from_raw",
    "build_prometheus_prompt",
    "parse_prometheus_score",
    "score_to_reward",
]
