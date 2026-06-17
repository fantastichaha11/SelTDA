#!/usr/bin/env python3
"""Download PathVQA from HuggingFace and write SelTDA-compatible all_data.json + images."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

SPLIT_MAP = {
    "train": "train",
    "validation": "val",
    "test": "test",
}


def infer_answer_type(answer: str) -> str:
    normalized = answer.lower().strip().rstrip(".")
    if normalized in {"yes", "no"}:
        return "yes/no"
    if re.fullmatch(r"[\d,.]+", normalized):
        return "number"
    return "other"


def infer_question_type(question: str, answer_type: str) -> str:
    if answer_type == "yes/no":
        return "yes/no"

    lowered = question.lower().strip()
    if lowered.startswith("how many") or lowered.startswith("how much"):
        return "how many"
    for prefix in ("what", "where", "why", "how", "when", "whose", "which", "who"):
        if lowered.startswith(prefix):
            return prefix
    return "other"


def split_ready(root: Path) -> bool:
    marker = root / "all_data.json"
    images_root = root / "images"
    if not marker.exists():
        return False
    for split in SPLIT_MAP.values():
        if not any((images_root / split).glob("*.jpg")):
            return False
    return True


def build_split_records(
    hf_split,
    split_name: str,
    images_root: Path,
    question_id_start: int,
) -> Tuple[List[dict], List[dict], int]:
    qa_records: List[dict] = []
    vqa_records: List[dict] = []
    question_id = question_id_start

    split_dir = images_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in enumerate(hf_split):
        question = row["question"].strip()
        answer = row["answer"].strip()
        image_stem = f"{split_name}_{idx:06d}"
        image_path = split_dir / f"{image_stem}.jpg"

        if not image_path.exists():
            image = row["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(image_path, format="JPEG", quality=95)

        answer_type = infer_answer_type(answer)
        question_type = infer_question_type(question, answer_type)

        qa_records.append(
            {
                "image": image_stem,
                "question": question,
                "answer": answer,
            }
        )
        vqa_records.append(
            {
                "answer_type": answer_type,
                "img_id": image_stem,
                "label": {answer: 1},
                "question_id": question_id,
                "question_type": question_type,
                "sent": question,
            }
        )
        question_id += 1

    return qa_records, vqa_records, question_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/pathvqa"),
        help="Directory for all_data.json and images/",
    )
    args = parser.parse_args()

    root = args.output_root.resolve()
    if split_ready(root):
        print(f"[skip] PathVQA already present under {root}")
        return

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pip install datasets huggingface-hub"
        ) from exc

    print(f"Downloading PathVQA from HuggingFace into {root}...")
    dataset = load_dataset("flaviagiammarino/path-vqa")
    images_root = root / "images"
    images_root.mkdir(parents=True, exist_ok=True)

    dump: Dict[str, List[dict]] = {}
    question_id = 0
    for hf_split, split_name in SPLIT_MAP.items():
        qa_records, vqa_records, question_id = build_split_records(
            dataset[hf_split],
            split_name,
            images_root,
            question_id,
        )
        dump[f"{split_name}_qa"] = qa_records
        dump[f"{split_name}_vqa"] = vqa_records
        print(f"  {split_name}: {len(qa_records)} QA pairs")

    root.mkdir(parents=True, exist_ok=True)
    with open(root / "all_data.json", "w") as f:
        json.dump(dump, f)

    print(f"Wrote {root / 'all_data.json'}")


if __name__ == "__main__":
    main()
