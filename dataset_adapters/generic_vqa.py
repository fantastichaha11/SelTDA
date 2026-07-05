from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence


_PUNCT_RE = re.compile(r"[\t\n\r\f\v]+|[\"'`.,!?;:()\[\]{}]")
_SPACE_RE = re.compile(r"\s+")


def normalize_answer(answer: object) -> str:
    """Normalize short VQA answers without applying dataset-specific stemming."""
    text = "" if answer is None else str(answer)
    text = text.strip().lower()
    text = _PUNCT_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _iter_answers(records: Iterable[Mapping]) -> Iterable[str]:
    for record in records:
        answers = record.get("answer", [])
        if isinstance(answers, str):
            answers = [answers]
        for answer in answers:
            normalized = normalize_answer(answer)
            if normalized:
                yield normalized


def build_answer_list(*record_groups: Iterable[Mapping]) -> list[str]:
    answers: set[str] = set()
    for records in record_groups:
        answers.update(_iter_answers(records))
    return sorted(answers)


def vqa_soft_accuracy(prediction: str, references: Sequence[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    normalized_references = [normalize_answer(ref) for ref in references]
    matches = sum(ref == normalized_prediction for ref in normalized_references)
    return min(1.0, matches / 3.0)


def exact_match_accuracy(
    predictions: Mapping[int, str],
    references: Mapping[int, Sequence[str]],
) -> float:
    if not references:
        return 0.0
    correct = 0
    for question_id, ref_answers in references.items():
        prediction = normalize_answer(predictions.get(question_id, ""))
        normalized_refs = {normalize_answer(answer) for answer in ref_answers}
        if prediction in normalized_refs:
            correct += 1
    return correct / len(references)


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
    return "other"
