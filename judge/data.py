from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from dataset_adapters.generic_vqa import normalize_answer


YES_NO = {"yes", "no"}
GENERIC_MEDICAL_ANSWERS = (
    "tissue",
    "cells",
    "abnormality",
    "lesion",
    "nuclei",
)


@dataclass(frozen=True)
class AnswerPools:
    all_answers: list[str]
    by_prefix: dict[str, list[str]]
    by_type: dict[str, list[str]]
    by_image: dict[str, list[str]]


@dataclass(frozen=True)
class JudgeCandidate:
    image: str
    question: str
    answer: str
    label: int
    question_id: object | None = None


@dataclass(frozen=True)
class JudgePair:
    positive: JudgeCandidate
    negative: JudgeCandidate


@dataclass(frozen=True)
class ImagePoolItem:
    image: str
    image_path: str
    question: str | None = None
    answer: str | None = None


def load_records(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list")
    return data


def answer_text(answer: object) -> str:
    if isinstance(answer, list):
        return normalize_answer(answer[0] if answer else "")
    return normalize_answer(answer)


def question_prefix(question: str) -> str:
    q = normalize_answer(question)
    if q.startswith("how many"):
        return "how many"
    if q.startswith("where"):
        return "where"
    if q.startswith("what"):
        return "what"
    if q.startswith("is ") or q.startswith("are "):
        return "is/are"
    if q.startswith("does ") or q.startswith("do "):
        return "does/do"
    return "other"


def infer_answer_type(answer: object) -> str:
    text = answer_text(answer)
    if text in YES_NO:
        return "yes/no"
    if re.fullmatch(r"\d+", text):
        return "count"
    if len(text.split()) <= 4:
        return "phrase"
    return "long"


def _sorted_unique(values: Iterable[str]) -> list[str]:
    return sorted({value for value in values if value})


def build_answer_pools(records: Sequence[Mapping]) -> AnswerPools:
    all_answers: list[str] = []
    by_prefix: dict[str, list[str]] = defaultdict(list)
    by_type: dict[str, list[str]] = defaultdict(list)
    by_image: dict[str, list[str]] = defaultdict(list)

    for record in records:
        answer = answer_text(record.get("answer", ""))
        if not answer:
            continue
        prefix = question_prefix(str(record.get("question", "")))
        answer_type = infer_answer_type(answer)
        image = str(record.get("image", ""))
        all_answers.append(answer)
        by_prefix[prefix].append(answer)
        by_type[answer_type].append(answer)
        by_image[image].append(answer)

    return AnswerPools(
        all_answers=_sorted_unique(all_answers),
        by_prefix={key: _sorted_unique(values) for key, values in by_prefix.items()},
        by_type={key: _sorted_unique(values) for key, values in by_type.items()},
        by_image={key: _sorted_unique(values) for key, values in by_image.items()},
    )


class NegativeSampler:
    def __init__(self, pools: AnswerPools, seed: int = 42):
        self.pools = pools
        self.rng = random.Random(seed)

    def sample(self, record: Mapping) -> str:
        gold = answer_text(record.get("answer", ""))
        if gold == "yes":
            return "no"
        if gold == "no":
            return "yes"

        image = str(record.get("image", ""))
        prefix = question_prefix(str(record.get("question", "")))
        answer_type = infer_answer_type(gold)
        candidate_groups = [
            self.pools.by_image.get(image, []),
            self.pools.by_prefix.get(prefix, []),
            self.pools.by_type.get(answer_type, []),
            list(GENERIC_MEDICAL_ANSWERS),
            self.pools.all_answers,
        ]
        for group in candidate_groups:
            choices = [candidate for candidate in group if candidate and candidate != gold]
            if choices:
                return self.rng.choice(choices)
        return "unknown"


def _candidate(record: Mapping, answer: str, label: int) -> JudgeCandidate:
    return JudgeCandidate(
        image=str(record.get("image", "")),
        question=str(record.get("question", "")),
        answer=answer,
        label=label,
        question_id=record.get("question_id"),
    )


def build_judge_pairs(
    records: Sequence[Mapping],
    pools: AnswerPools,
    seed: int = 42,
    negatives_per_positive: int = 1,
) -> list[JudgePair]:
    sampler = NegativeSampler(pools, seed=seed)
    pairs: list[JudgePair] = []
    negative_count = max(0, int(negatives_per_positive))

    for record in records:
        positive_answer = answer_text(record.get("answer", ""))
        if not positive_answer:
            continue
        for _ in range(negative_count):
            negative_answer = sampler.sample(record)
            pairs.append(
                JudgePair(
                    positive=_candidate(record, positive_answer, 1),
                    negative=_candidate(record, negative_answer, 0),
                )
            )
    return pairs


def load_image_pool(image_pool_config: Mapping) -> list[ImagePoolItem]:
    annotations = load_records(image_pool_config["annotations"])
    image_root = Path(str(image_pool_config["image_root"]))
    use_ground_truth_qa = bool(image_pool_config.get("use_ground_truth_qa", False))
    items: list[ImagePoolItem] = []
    seen: set[str] = set()

    for record in annotations:
        image = str(record["image"])
        if image in seen:
            continue
        seen.add(image)
        items.append(
            ImagePoolItem(
                image=image,
                image_path=str(image_root / image),
                question=normalize_answer(record.get("question", "")) if use_ground_truth_qa else None,
                answer=answer_text(record.get("answer", "")) if use_ground_truth_qa else None,
            )
        )
    return items
