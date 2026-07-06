from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any

from dataset_adapters.generic_vqa import (
    load_json,
    normalize_answer,
    question_prefix,
    write_json,
)
import wandb_utils


METRIC_DECIMALS = 4


def _prediction_lookup(results: list[dict[str, Any]]) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for row in results:
        if not isinstance(row, dict) or "question_id" not in row or "answer" not in row:
            raise ValueError(f"Result row must contain question_id and answer: {row}")
        question_id = int(row["question_id"])
        answer = normalize_answer(row["answer"])
        if question_id in lookup and lookup[question_id] != answer:
            raise ValueError(f"Conflicting duplicate prediction for question_id={question_id}")
        lookup[question_id] = answer
    return lookup


def _answers(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value]


def _mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _metric(values: list[float]) -> float:
    return round(_mean(values), METRIC_DECIMALS)


def evaluate_daquar(
    result_file: str | Path,
    annotation_file: str | Path,
) -> dict[str, Any]:
    annotations = load_json(annotation_file)
    results = load_json(result_file)
    if not isinstance(annotations, list) or not isinstance(results, list):
        raise ValueError("DAQUAR annotations and results must be JSON lists")

    predictions = _prediction_lookup(results)
    scores: list[float] = []
    by_prefix: dict[str, list[float]] = {}

    for ann in annotations:
        if not isinstance(ann, dict):
            raise ValueError(f"DAQUAR annotation row must be a JSON object: {ann}")
        question_id = int(ann["question_id"])
        if question_id not in predictions:
            raise ValueError(f"Missing prediction for question_id={question_id}")

        refs = {normalize_answer(answer) for answer in _answers(ann.get("answer", []))}
        score = 1.0 if predictions[question_id] in refs else 0.0
        scores.append(score)

        prefix = question_prefix(str(ann.get("question", "")))
        by_prefix.setdefault(prefix, []).append(score)

    return {
        "overall": _metric(scores),
        "by_question_prefix": {
            prefix: _metric(prefix_scores)
            for prefix, prefix_scores in sorted(by_prefix.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DAQUAR generic VQA predictions.")
    parser.add_argument("result_file", type=Path)
    parser.add_argument("--annotation-file", type=Path, default=Path("datasets/daquar/val.json"))
    parser.add_argument("--wandb", action="store_true", help="Log eval metrics and JSON artifact to W&B.")
    parser.add_argument("--wandb-project", default="seltda")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default="daquar-eval")
    parser.add_argument("--wandb-group", default="daquar")
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--wandb-tags", nargs="*", default=["daquar", "eval"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_daquar(args.result_file, args.annotation_file)
    print(metrics)
    eval_path = Path(args.result_file).parent / "daquar_eval.json"
    write_json(eval_path, metrics)
    if args.wandb:
        logger = wandb_utils.init_wandb(
            argparse.Namespace(output_dir=str(Path(args.result_file).parent), evaluate=True),
            {
                "wandb": True,
                "wandb_project": args.wandb_project,
                "wandb_entity": args.wandb_entity,
                "wandb_name": args.wandb_name,
                "wandb_group": args.wandb_group,
                "wandb_mode": args.wandb_mode,
                "wandb_tags": args.wandb_tags,
                "dataset_name": "daquar",
            },
            job_type="daquar-eval",
        )
        logger.log_metrics(metrics, prefix="eval")
        logger.update_summary(metrics, prefix="eval")
        logger.log_artifact(eval_path, name="daquar-eval", artifact_type="eval", metadata=metrics)
        logger.finish()


if __name__ == "__main__":
    main()
