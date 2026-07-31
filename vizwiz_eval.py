from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import Any

from dataset_adapters.generic_vqa import write_json
from experiments.analysis import vizwiz_per_example
import wandb_utils


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
    rows = vizwiz_per_example(result_file, annotation_file, metadata_file)
    overall_scores: list[float] = []
    answerable_scores: list[float] = []
    unanswerable_scores: list[float] = []
    by_answer_type: dict[str, list[float]] = {}

    for row in rows:
        score = float(row["score"])
        overall_scores.append(score)
        answer_type = str(row.get("answer_type", "unknown"))
        by_answer_type.setdefault(answer_type, []).append(score)
        if int(row.get("answerable", 1)) == 1:
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
    parser.add_argument("--wandb", action="store_true", help="Log eval metrics and JSON artifact to W&B.")
    parser.add_argument("--wandb-project", default="seltda")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-name", default="vizwiz-eval")
    parser.add_argument("--wandb-group", default="vizwiz")
    parser.add_argument("--wandb-mode", default="online")
    parser.add_argument("--wandb-tags", nargs="*", default=["vizwiz", "eval"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_vizwiz(args.result_file, args.annotation_file, args.metadata_file)
    print(metrics)
    eval_path = Path(args.result_file).parent / "vizwiz_eval.json"
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
                "dataset_name": "vizwiz",
            },
            job_type="vizwiz-eval",
        )
        logger.log_metrics(metrics, prefix="eval")
        logger.update_summary(metrics, prefix="eval")
        logger.log_artifact(eval_path, name="vizwiz-eval", artifact_type="eval", metadata=metrics)
        logger.finish()


if __name__ == "__main__":
    main()
