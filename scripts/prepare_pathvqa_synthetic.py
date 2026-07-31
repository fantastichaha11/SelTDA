#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.synthetic_data import assign_missing_question_ids, clean_question


def normalize_question(question: str) -> str:
    return clean_question(question)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert generated PathVQA QA records into trainable annotations."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--question-id-start", type=int, default=10000000)
    args = parser.parse_args()

    with args.input.open("r") as f:
        records = json.load(f)

    converted = assign_missing_question_ids(
        records,
        start=args.question_id_start,
        dataset="pathvqa",
    )
    for output, source in zip(converted, records):
        output["rationales"] = source.get("rationales")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(converted, f)

    print(f"wrote {len(converted)} records to {args.output}")


if __name__ == "__main__":
    main()
