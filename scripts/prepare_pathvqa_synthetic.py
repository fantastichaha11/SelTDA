#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


QUESTION_PREFIX = re.compile(r"^\s*question\s*:\s*", re.IGNORECASE)


def normalize_question(question: str) -> str:
    return QUESTION_PREFIX.sub("", question).strip()


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

    converted = []
    for idx, record in enumerate(records):
        converted.append(
            {
                **record,
                "question_id": args.question_id_start + idx,
                "question": normalize_question(record["question"]),
                "dataset": "pathvqa",
                "rationales": record.get("rationales"),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(converted, f)

    print(f"wrote {len(converted)} records to {args.output}")


if __name__ == "__main__":
    main()
