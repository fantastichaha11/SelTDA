from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any, Iterable

from dataset_adapters.generic_vqa import build_answer_list, load_json, normalize_answer, write_json


VAL_ID_OFFSET = 1_000_000

_IMAGE_KEYS = ("image", "image_id", "img")
_QUESTION_KEYS = ("question", "question_str")
_ANSWER_KEYS = ("answer", "answers")
_RAW_IMAGE_RE = re.compile(r"\b(image\d+)\b", re.IGNORECASE)
_RAW_IMAGE_SUFFIX_RE = re.compile(r"\s+in\s+the\s+image\d+\s*\?\s*$", re.IGNORECASE)


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"DAQUAR {label} file not found: {path}")


def _first_value(record: dict[str, Any], keys: Iterable[str], label: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    raise ValueError(f"DAQUAR record missing required {label} field: {record}")


def _answer_value(record: dict[str, Any]) -> str:
    raw = _first_value(record, _ANSWER_KEYS, "answer")
    if isinstance(raw, list):
        for item in raw:
            normalized = normalize_answer(item)
            if normalized:
                return normalized
        raise ValueError(f"DAQUAR record has no usable answer: {record}")
    normalized = normalize_answer(raw)
    if not normalized:
        raise ValueError(f"DAQUAR record has no usable answer: {record}")
    return normalized


def _image_candidates(raw_image: Any) -> list[str]:
    image = str(raw_image).strip()
    if not image:
        raise ValueError("DAQUAR record has an empty image field")
    path = Path(image)
    if path.suffix:
        return [image]
    suffixes = (".png", ".jpg", ".jpeg")
    candidates = [f"{image}{suffix}" for suffix in suffixes]
    if image.isdigit():
        candidates = [f"image{image}{suffix}" for suffix in suffixes] + candidates
    return candidates


def _resolve_image_name(raw_image: Any, image_root: Path) -> str:
    candidates = _image_candidates(raw_image)
    for candidate in candidates:
        if (image_root / candidate).is_file():
            return candidate
    return candidates[0]


def _dialect_for(path: Path, sample: str) -> csv.Dialect:
    if path.suffix.lower() == ".tsv":
        return csv.excel_tab
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t")
    except csv.Error:
        return csv.excel


def _has_header(row: list[str]) -> bool:
    fields = {field.strip().lower() for field in row}
    return (
        bool(fields.intersection(_IMAGE_KEYS))
        and bool(fields.intersection(_QUESTION_KEYS))
        and bool(fields.intersection(_ANSWER_KEYS))
    )


def _clean_raw_question(question: str) -> str:
    cleaned = _RAW_IMAGE_SUFFIX_RE.sub("?", question.strip())
    return cleaned.strip()


def _looks_like_raw_daquar(lines: list[str]) -> bool:
    return (
        len(lines) >= 2
        and len(lines) % 2 == 0
        and bool(_RAW_IMAGE_RE.search(lines[0]))
        and not _RAW_IMAGE_RE.search(lines[1])
    )


def _load_raw_text_records(path: Path, lines: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx in range(0, len(lines), 2):
        question = lines[idx].strip()
        answer = lines[idx + 1].strip()
        image_match = _RAW_IMAGE_RE.search(question)
        if image_match is None:
            raise ValueError(f"DAQUAR raw question missing image id in {path}: {question}")
        records.append(
            {
                "image": image_match.group(1).lower(),
                "question": _clean_raw_question(question),
                "answer": answer,
            }
        )
    return records


def _load_table_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if _looks_like_raw_daquar(lines):
        return _load_raw_text_records(path, lines)

    dialect = _dialect_for(path, text[:2048])
    rows = list(csv.reader(text.splitlines(), dialect=dialect))
    rows = [row for row in rows if any(cell.strip() for cell in row)]
    if not rows:
        return []

    if _has_header(rows[0]):
        reader = csv.DictReader(text.splitlines(), dialect=dialect)
        return [
            {str(key).strip().lower(): value for key, value in row.items() if key is not None}
            for row in reader
            if row and any((value or "").strip() for value in row.values())
        ]

    records: list[dict[str, Any]] = []
    for row in rows:
        if len(row) < 3:
            raise ValueError(f"DAQUAR no-header row requires image, question, answer columns: {row}")
        records.append({"image": row[0], "question": row[1], "answer": row[2]})
    return records


def _load_qa_records(path: Path) -> list[dict[str, Any]]:
    _require_file(path, "QA")
    if path.suffix.lower() == ".json":
        data = load_json(path)
        if not isinstance(data, list):
            raise ValueError(f"DAQUAR JSON QA file must contain a list: {path}")
        if not all(isinstance(record, dict) for record in data):
            raise ValueError(f"DAQUAR JSON QA records must be objects: {path}")
        return data
    return _load_table_records(path)


def _convert_split(
    raw_records: list[dict[str, Any]],
    *,
    image_root: Path,
    id_offset: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    converted: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, str]] = {}

    if not image_root.is_dir():
        raise FileNotFoundError(f"DAQUAR image directory not found: {image_root}")

    for idx, raw in enumerate(raw_records):
        image = _resolve_image_name(_first_value(raw, _IMAGE_KEYS, "image"), image_root)
        question = str(_first_value(raw, _QUESTION_KEYS, "question")).strip()
        if not question:
            raise ValueError(f"DAQUAR record has an empty question field: {raw}")
        answer = _answer_value(raw)

        image_path = image_root / image
        if not image_path.is_file():
            raise FileNotFoundError(f"DAQUAR image missing: {image_path}")

        question_id = id_offset + idx
        converted.append(
            {
                "dataset": "daquar",
                "image": image,
                "question": question,
                "question_id": question_id,
                "answer": [answer],
            }
        )
        metadata[str(question_id)] = {"answer": answer, "question": question}

    return converted, metadata


def convert_daquar_dataset(
    daquar_root: str | Path,
    *,
    train_qa: str | Path | None = None,
    test_qa: str | Path | None = None,
    image_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> None:
    root = Path(daquar_root)
    train_qa = Path(train_qa) if train_qa is not None else root / "qa_train.json"
    test_qa = Path(test_qa) if test_qa is not None else root / "qa_test.json"
    image_root = Path(image_root) if image_root is not None else root / "images"
    output_root = Path(output_root) if output_root is not None else root

    train_raw = _load_qa_records(train_qa)
    test_raw = _load_qa_records(test_qa)
    if not train_raw:
        raise ValueError(f"DAQUAR train split is empty: {train_qa}")
    if not test_raw:
        raise ValueError(f"DAQUAR test split is empty: {test_qa}")

    train_records, _ = _convert_split(train_raw, image_root=image_root, id_offset=0)
    val_records, val_metadata = _convert_split(
        test_raw,
        image_root=image_root,
        id_offset=VAL_ID_OFFSET,
    )

    write_json(output_root / "train.json", train_records)
    write_json(output_root / "val.json", val_records)
    write_json(output_root / "answer_list.json", build_answer_list(train_records, val_records))
    write_json(output_root / "daquar_val_metadata.json", val_metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DAQUAR to SelTDA generic VQA JSON.")
    parser.add_argument("--daquar-root", type=Path, required=True)
    parser.add_argument("--train-qa", type=Path, default=None)
    parser.add_argument("--test-qa", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_daquar_dataset(
        args.daquar_root,
        train_qa=args.train_qa,
        test_qa=args.test_qa,
        image_root=args.image_root,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
