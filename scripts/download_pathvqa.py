#!/usr/bin/env python3
"""Download PathVQA from HuggingFace and write SelTDA-compatible all_data.json + images."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Tuple

SPLIT_MAP = {
    "train": "train",
    "validation": "val",
    "test": "test",
}

PATHVQA_TEACHER_CHECKPOINT_FILE_ID = "1l_VOx3GL6_MQoatmrCEfjF47VV_ihpCq"
PATHVQA_SYNTHETIC_DATA_FILE_ID = "14ZQcf5NRhE0_3ZsZf6adzrqrkrVak9W8"


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


def resolve_auxiliary_asset_paths(
    output_root: Path,
    *,
    teacher_checkpoint_output: Path | None = None,
    synthetic_data_output: Path | None = None,
) -> dict[str, Path]:
    return {
        "teacher_checkpoint": teacher_checkpoint_output
        or Path("cache/pathvqa_teacher_weights/checkpoint_04.pth"),
        "synthetic_data": synthetic_data_output or (output_root / "synthetic_data_raw.json"),
    }


def download_google_drive_file(
    file_id: str,
    output_path: Path,
    *,
    force: bool = False,
    runner: Callable[..., object] = subprocess.run,
) -> bool:
    output_path = Path(output_path)
    if output_path.exists() and not force:
        print(f"[skip] {output_path} already exists")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    runner(
        [
            "gdown",
            f"https://drive.google.com/uc?id={file_id}",
            "-O",
            str(output_path),
        ],
        check=True,
    )
    return True


def download_auxiliary_assets(
    output_root: Path,
    *,
    teacher_checkpoint_output: Path | None = None,
    synthetic_data_output: Path | None = None,
    force: bool = False,
) -> None:
    paths = resolve_auxiliary_asset_paths(
        output_root,
        teacher_checkpoint_output=teacher_checkpoint_output,
        synthetic_data_output=synthetic_data_output,
    )
    download_google_drive_file(
        PATHVQA_TEACHER_CHECKPOINT_FILE_ID,
        paths["teacher_checkpoint"],
        force=force,
    )
    download_google_drive_file(
        PATHVQA_SYNTHETIC_DATA_FILE_ID,
        paths["synthetic_data"],
        force=force,
    )


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
        default=Path("/teamspace/uploads/pathvqa"),
        help="Directory for all_data.json and images/",
    )
    parser.add_argument(
        "--include-aux-assets",
        action="store_true",
        help="Also download the PathVQA teacher checkpoint and generated synthetic data.",
    )
    parser.add_argument(
        "--teacher-checkpoint-output",
        type=Path,
        default=None,
        help="Override output path for the downloaded PathVQA teacher checkpoint.",
    )
    parser.add_argument(
        "--synthetic-data-output",
        type=Path,
        default=None,
        help="Override output path for the downloaded PathVQA synthetic data JSON.",
    )
    parser.add_argument(
        "--force-aux-download",
        action="store_true",
        help="Re-download auxiliary assets even if target files already exist.",
    )
    args = parser.parse_args()

    root = args.output_root.resolve()
    if split_ready(root):
        print(f"[skip] PathVQA already present under {root}")
        if args.include_aux_assets:
            download_auxiliary_assets(
                root,
                teacher_checkpoint_output=args.teacher_checkpoint_output,
                synthetic_data_output=args.synthetic_data_output,
                force=args.force_aux_download,
            )
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
    if args.include_aux_assets:
        download_auxiliary_assets(
            root,
            teacher_checkpoint_output=args.teacher_checkpoint_output,
            synthetic_data_output=args.synthetic_data_output,
            force=args.force_aux_download,
        )


if __name__ == "__main__":
    main()
