from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from data.utils import join_image_root_path
from dataset_adapters.generic_vqa import normalize_answer
from judge.data import GENERIC_MEDICAL_ANSWERS, infer_answer_type, question_prefix


@dataclass(frozen=True)
class EligibilityResult:
    records: list[dict]
    rejections: dict[str, int]


def _as_answer_list(value: object) -> list[object]:
    return value if isinstance(value, list) else [value]


def canonical_answer(value: object) -> str:
    normalized = [normalize_answer(item) for item in _as_answer_list(value)]
    normalized = [item for item in normalized if item]
    if not normalized:
        return ""
    counts = Counter(normalized)
    return min(counts, key=lambda item: (-counts[item], normalized.index(item)))


def canonical_image_key(value: object, image_root: str | Path | None = None) -> str:
    raw = str(value or "").replace("\\", "/").lstrip("/")
    if not raw or image_root is None:
        return Path(raw).as_posix()

    root = Path(image_root).resolve()
    resolved = Path(join_image_root_path(str(root), raw)).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"image escapes image_root: {raw}") from exc


def record_identity(
    record: Mapping, image_root: str | Path | None = None
) -> tuple[str, str, str]:
    return (
        canonical_image_key(record.get("image", ""), image_root),
        normalize_answer(record.get("question", "")),
        canonical_answer(record.get("answer", "")),
    )


def eligible_records(
    records: Iterable[Mapping],
    image_root: str | Path,
    *,
    dataset: str,
) -> EligibilityResult:
    root = Path(image_root)
    accepted: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    rejected: Counter[str] = Counter()

    for source_index, raw in enumerate(records):
        try:
            identity = record_identity(raw, root)
        except ValueError:
            rejected["image_outside_root"] += 1
            continue

        image, question, answer = identity
        if not image or not question or not answer:
            rejected["empty_field"] += 1
            continue
        if not Path(join_image_root_path(str(root), image)).is_file():
            rejected["missing_image"] += 1
            continue
        if identity in seen:
            rejected["duplicate"] += 1
            continue

        seen.add(identity)
        accepted.append(
            {
                **dict(raw),
                "image": image,
                "question": str(raw.get("question", "")).strip(),
                "answer": [answer],
                "dataset": dataset,
                "_source_index": source_index,
            }
        )

    return EligibilityResult(records=accepted, rejections=dict(rejected))


def synthetic_quota(*, real_count: int, target_total: int) -> int:
    quota = int(target_total) - int(real_count)
    if quota < 0:
        raise ValueError(f"real_count={real_count} exceeds target_total={target_total}")
    return quota


def _tie_key(record: Mapping, seed: int) -> str:
    payload = f"{seed}|{'|'.join(record_identity(record))}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def rank_global(records: Sequence[Mapping], *, quota: int, seed: int) -> list[dict]:
    if len(records) < quota:
        raise ValueError(f"eligible records {len(records)} < quota {quota}")
    ordered = sorted(
        records,
        key=lambda row: (-int(row["judge_score"]), _tie_key(row, seed)),
    )
    return [dict(row) for row in ordered[:quota]]


def build_nested_synthetic(
    base: Sequence[Mapping],
    extra: Sequence[Mapping],
    *,
    target_synthetic: int,
    image_root: str | Path | None = None,
) -> list[dict]:
    target = int(target_synthetic)
    if len(base) > target:
        raise ValueError(f"base synthetic count {len(base)} exceeds target {target}")

    nested = [dict(row) for row in base]
    seen = {record_identity(row, image_root) for row in base}
    for row in extra:
        identity = record_identity(row, image_root)
        if identity in seen:
            continue
        seen.add(identity)
        nested.append(dict(row))
        if len(nested) == target:
            return nested

    raise ValueError(f"available extra records cannot reach target_synthetic={target}")


def _distribution(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _rate(count: int, total: int) -> float:
    return 0.0 if total == 0 else count / total


def _length_summary(lengths: list[int]) -> dict[str, float]:
    if not lengths:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    ordered = sorted(lengths)

    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
        return float(ordered[index])

    return {
        "mean": float(sum(ordered) / len(ordered)),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
    }


def dataset_diagnostics(records: Sequence[Mapping]) -> dict:
    count = len(records)
    identities = [record_identity(row) for row in records]
    questions = [identity[1] for identity in identities]
    answers = [identity[2] for identity in identities]
    images = [identity[0] for identity in identities]
    prefixes = [question_prefix(str(row.get("question", ""))) for row in records]
    answer_types = [infer_answer_type(row.get("answer", "")) for row in records]
    generic_answers = set(GENERIC_MEDICAL_ANSWERS)

    return {
        "count": count,
        "unique_images": len(set(images)),
        "duplicate_question_rate": _rate(count - len(set(questions)), count),
        "duplicate_qa_rate": _rate(count - len(set(identities)), count),
        "question_prefix_distribution": _distribution(prefixes),
        "answer_type_distribution": _distribution(answer_types),
        "yes_no_rate": _rate(sum(answer in {"yes", "no"} for answer in answers), count),
        "generic_answer_rate": _rate(sum(answer in generic_answers for answer in answers), count),
        "unanswerable_rate": _rate(
            sum(answer in {"unanswerable", "unknown", "cannot determine"} for answer in answers),
            count,
        ),
        "question_length_words": _length_summary(
            [len(question.split()) for question in questions]
        ),
        "answer_length_words": _length_summary([len(answer.split()) for answer in answers]),
    }


def sha256_json(payload: object) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
