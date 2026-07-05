from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from dataset_adapters.generic_vqa import build_answer_list, load_json, normalize_answer, write_json


VAL_ID_OFFSET = 1_000_000


def _answers(raw: dict[str, Any]) -> list[str]:
    answers = raw.get("answers", [])
    if not isinstance(answers, list):
        raise ValueError(f"VizWiz record answers must be a list: {raw}")
    normalized = []
    for answer in answers:
        if isinstance(answer, dict):
            value = answer.get("answer", "")
        else:
            value = answer
        normalized_answer = normalize_answer(value)
        if normalized_answer:
            normalized.append(normalized_answer)
    if not normalized:
        raise ValueError(f"VizWiz record has no usable answers: {raw}")
    return normalized


def _majority_answer(answers: list[str]) -> str:
    counts: dict[str, int] = {}
    for answer in answers:
        counts[answer] = counts.get(answer, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _convert_split(
    raw_records: list[dict[str, Any]],
    split: str,
    image_dir: Path,
    id_offset: int,
    include_unanswerable: bool,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    converted: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}

    if not image_dir.is_dir():
        raise FileNotFoundError(f"VizWiz image directory not found: {image_dir}")

    for idx, raw in enumerate(raw_records):
        image_name = raw.get("image")
        question = raw.get("question")
        if not image_name or not question:
            raise ValueError(f"VizWiz record missing image or question: {raw}")
        answers = _answers(raw)
        if not include_unanswerable and _majority_answer(answers) == "unanswerable":
            continue
        image_path = image_dir / image_name
        if not image_path.is_file():
            raise FileNotFoundError(f"VizWiz image missing: {image_path}")
        question_id = id_offset + idx
        converted.append(
            {
                "dataset": "vizwiz",
                "image": f"{split}/{image_name}",
                "question": str(question).strip(),
                "question_id": question_id,
                "answer": answers,
            }
        )
        metadata[str(question_id)] = {
            "answerable": int(raw.get("answerable", 1)),
            "answer_type": str(raw.get("answer_type", "unknown")),
        }

    if not converted:
        raise ValueError(f"VizWiz {split} split is empty after conversion")
    return converted, metadata


def convert_vizwiz_dataset(
    vizwiz_root: str | Path,
    *,
    train_annotations: str | Path | None = None,
    val_annotations: str | Path | None = None,
    train_image_dir: str | Path | None = None,
    val_image_dir: str | Path | None = None,
    output_root: str | Path | None = None,
    include_unanswerable: bool = True,
) -> None:
    root = Path(vizwiz_root)
    output_root = Path(output_root) if output_root is not None else root
    train_annotations = Path(train_annotations) if train_annotations else root / "annotations" / "train.json"
    val_annotations = Path(val_annotations) if val_annotations else root / "annotations" / "val.json"
    train_image_dir = Path(train_image_dir) if train_image_dir else root / "images" / "train"
    val_image_dir = Path(val_image_dir) if val_image_dir else root / "images" / "val"

    train_raw = load_json(train_annotations)
    val_raw = load_json(val_annotations)
    if not isinstance(train_raw, list) or not isinstance(val_raw, list):
        raise ValueError("VizWiz annotations must be JSON lists")

    train_records, _ = _convert_split(
        train_raw,
        "train",
        train_image_dir,
        id_offset=0,
        include_unanswerable=include_unanswerable,
    )
    val_records, val_metadata = _convert_split(
        val_raw,
        "val",
        val_image_dir,
        id_offset=VAL_ID_OFFSET,
        include_unanswerable=include_unanswerable,
    )

    write_json(output_root / "train.json", train_records)
    write_json(output_root / "val.json", val_records)
    write_json(output_root / "answer_list.json", build_answer_list(train_records, val_records))
    write_json(output_root / "vizwiz_val_metadata.json", val_metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert VizWiz-VQA to SelTDA generic VQA JSON.")
    parser.add_argument("--vizwiz-root", type=Path, required=True)
    parser.add_argument("--train-annotations", type=Path, default=None)
    parser.add_argument("--val-annotations", type=Path, default=None)
    parser.add_argument("--train-image-dir", type=Path, default=None)
    parser.add_argument("--val-image-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--exclude-unanswerable",
        action="store_true",
        help="Drop records whose majority answer is unanswerable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_vizwiz_dataset(
        args.vizwiz_root,
        train_annotations=args.train_annotations,
        val_annotations=args.val_annotations,
        train_image_dir=args.train_image_dir,
        val_image_dir=args.val_image_dir,
        output_root=args.output_root,
        include_unanswerable=not args.exclude_unanswerable,
    )


if __name__ == "__main__":
    main()
