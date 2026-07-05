from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any

from dataset_adapters.generic_vqa import load_json, vqa_soft_accuracy, write_json


METRIC_DECIMALS = 4


def _prediction_lookup(results: list[dict[str, Any]]) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for row in results:
        if not isinstance(row, dict) or "question_id" not in row or "answer" not in row:
            raise ValueError(f"Result row must contain question_id and answer: {row}")
        lookup[int(row["question_id"])] = str(row["answer"])
    return lookup


def _mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _metric(values: list[float]) -> float:
    return round(_mean(values), METRIC_DECIMALS)


def evaluate_vizwiz(
    result_file: str | Path,
    annotation_file: str | Path,
    metadata_file: str | Path,
) -> dict[str, Any]:
    annotations = load_json(annotation_file)
    results = load_json(result_file)
    metadata = load_json(metadata_file)
    if not isinstance(annotations, list) or not isinstance(results, list):
        raise ValueError("VizWiz annotations and results must be JSON lists")
    if not isinstance(metadata, dict):
        raise ValueError("VizWiz metadata must be a JSON object")

    predictions = _prediction_lookup(results)
    overall_scores: list[float] = []
    answerable_scores: list[float] = []
    unanswerable_scores: list[float] = []
    by_answer_type: dict[str, list[float]] = {}

    for ann in annotations:
        if not isinstance(ann, dict):
            raise ValueError(f"VizWiz annotation row must be a JSON object: {ann}")
        question_id = int(ann["question_id"])
        if question_id not in predictions:
            raise ValueError(f"Missing prediction for question_id={question_id}")
        meta = metadata.get(str(question_id))
        if meta is None:
            raise ValueError(f"Missing VizWiz metadata for question_id={question_id}")
        if not isinstance(meta, dict):
            raise ValueError(f"VizWiz metadata row must be a JSON object: {meta}")

        score = vqa_soft_accuracy(predictions[question_id], ann.get("answer", []))
        overall_scores.append(score)
        answer_type = str(meta.get("answer_type", "unknown"))
        by_answer_type.setdefault(answer_type, []).append(score)
        if int(meta.get("answerable", 1)) == 1:
            answerable_scores.append(score)
        else:
            unanswerable_scores.append(score)

    return {
        "overall": _metric(overall_scores),
        "answerable": _metric(answerable_scores),
        "unanswerable": _metric(unanswerable_scores),
        "by_answer_type": {
            answer_type: _metric(scores)
            for answer_type, scores in sorted(by_answer_type.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VizWiz generic VQA predictions.")
    parser.add_argument("result_file", type=Path)
    parser.add_argument("--annotation-file", type=Path, default=Path("datasets/vizwiz/val.json"))
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=Path("datasets/vizwiz/vizwiz_val_metadata.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_vizwiz(args.result_file, args.annotation_file, args.metadata_file)
    print(metrics)
    write_json(Path(args.result_file).parent / "vizwiz_eval.json", metrics)


if __name__ == "__main__":
    main()
