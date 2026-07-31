from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

import schemas
from dataset_adapters.generic_vqa import load_json, vqa_soft_accuracy
from experiments.synthetic_data import dataset_diagnostics, sha256_json
from judge.data import load_records


REGISTERED_COMPARISONS = (
    "pt4_j-filter4",
    "pt4_jp-pt4_j",
    "pt8_jp-pt4_jp",
    "pt4_j-seltda",
    "pt4_jp_60k-pt4_jp",
)


def paired_bootstrap_delta(
    left: dict[int, float],
    right: dict[int, float],
    *,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> dict:
    ids = sorted(set(left) & set(right))
    if set(ids) != set(left) or set(ids) != set(right):
        raise ValueError("paired conditions must contain identical question IDs")
    if not ids:
        raise ValueError("paired conditions must not be empty")
    differences = np.asarray([float(left[item]) - float(right[item]) for item in ids])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), size=(n_resamples, len(ids)))
    samples = differences[draws].mean(axis=1)
    return {
        "n": len(ids),
        "delta": float(differences.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "resamples": n_resamples,
        "seed": seed,
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2
        for original_index, _ in indexed[index:end]:
            ranks[original_index] = average_rank
        index = end
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Spearman inputs must have equal length")
    if not left:
        raise ValueError("Spearman inputs must not be empty")
    left_ranks = np.asarray(_average_ranks([float(value) for value in left]))
    right_ranks = np.asarray(_average_ranks([float(value) for value in right]))
    if np.all(left_ranks == left_ranks[0]) or np.all(right_ranks == right_ranks[0]):
        return 0.0
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _prediction_lookup(results: list[dict]) -> dict[int, str]:
    lookup = {}
    for row in results:
        if "question_id" not in row or "answer" not in row:
            raise ValueError(f"result row requires question_id and answer: {row}")
        lookup[int(row["question_id"])] = str(row["answer"])
    return lookup


def pathvqa_per_example(annotation_file: str | Path, result_file: str | Path) -> list[dict]:
    record_model = schemas.MinimalEvaluationRecord
    annotations = [
        record_model.model_validate(row)
        if hasattr(record_model, "model_validate")
        else record_model.parse_obj(row)
        for row in load_json(annotation_file)
    ]
    predictions = _prediction_lookup(load_json(result_file))
    rows = []
    for annotation in annotations:
        question_id = int(annotation.question_id)
        if question_id not in predictions:
            raise ValueError(f"Missing prediction for question_id={question_id}")
        prediction = predictions[question_id]
        true_answer = annotation.answer
        rows.append(
            {
                "question_id": question_id,
                "score": 1.0 if prediction == true_answer else 0.0,
                "prediction": prediction,
                "true_answer": true_answer,
                "question_type": annotation.question_type,
                "answer_type": annotation.answer_type,
            }
        )
    return rows


def vizwiz_per_example(
    result_file: str | Path,
    annotation_file: str | Path,
    metadata_file: str | Path,
) -> list[dict]:
    annotations = load_json(annotation_file)
    results = load_json(result_file)
    metadata = load_json(metadata_file)
    if not isinstance(annotations, list) or not isinstance(results, list):
        raise ValueError("VizWiz annotations and results must be JSON lists")
    if not isinstance(metadata, dict):
        raise ValueError("VizWiz metadata must be a JSON object")

    predictions = _prediction_lookup(results)
    rows = []
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ValueError(f"VizWiz annotation row must be a JSON object: {annotation}")
        question_id = int(annotation["question_id"])
        if question_id not in predictions:
            raise ValueError(f"Missing prediction for question_id={question_id}")
        meta = metadata.get(str(question_id))
        if meta is None:
            raise ValueError(f"Missing VizWiz metadata for question_id={question_id}")
        if not isinstance(meta, dict):
            raise ValueError(f"VizWiz metadata row must be a JSON object: {meta}")
        rows.append(
            {
                "question_id": question_id,
                "score": vqa_soft_accuracy(predictions[question_id], annotation.get("answer", [])),
                "prediction": predictions[question_id],
                "answer_type": str(meta.get("answer_type", "unknown")),
                "answerable": int(meta.get("answerable", 1)),
            }
        )
    return rows


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _condition_summary(rows: list[dict], *, metric_decimals: int = 4) -> dict:
    strata: dict[str, dict[str, list[float]]] = {
        "answer_type": defaultdict(list),
        "question_type": defaultdict(list),
        "answerable": defaultdict(list),
    }
    for row in rows:
        for key in ("answer_type", "question_type", "answerable"):
            if row.get(key) is not None:
                strata[key][str(row[key])].append(float(row["score"]))
    return {
        "overall": round(_mean([float(row["score"]) for row in rows]), metric_decimals),
        "n": len(rows),
        "strata": {
            key: {
                stratum: round(_mean(scores), metric_decimals)
                for stratum, scores in sorted(groups.items())
            }
            for key, groups in strata.items()
            if groups
        },
    }


def _scores_by_id(rows: list[dict]) -> dict[int, float]:
    return {int(row["question_id"]): float(row["score"]) for row in rows}


def _load_json_if_present(path: str | Path | None):
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def analyze_matrix(config) -> dict:
    dataset = str(config["dataset"])
    annotations = config["annotations"]
    metadata = config.get("metadata")
    condition_paths = dict(config["conditions"])
    rows_by_condition = {}
    for name, prediction_path in condition_paths.items():
        if dataset == "vizwiz":
            rows = vizwiz_per_example(prediction_path, annotations, metadata)
        else:
            rows = pathvqa_per_example(annotations, prediction_path)
        rows_by_condition[name] = rows

    comparisons = {}
    for left, right in config.get("comparisons", []):
        comparisons[f"{left}-{right}"] = paired_bootstrap_delta(
            _scores_by_id(rows_by_condition[left]),
            _scores_by_id(rows_by_condition[right]),
            n_resamples=int(config.get("bootstrap_resamples", 10_000)),
            seed=int(config.get("seed", 42)),
        )

    data_quality = {}
    for name, synthetic_path in dict(config.get("synthetics", {})).items():
        records = load_records(synthetic_path)
        data_quality[name] = {
            "sha256": sha256_json(records),
            **dataset_diagnostics(records),
        }

    compute_root = config.get("compute_root")
    efficiency = {}
    if compute_root:
        root = Path(compute_root)
        for name in condition_paths:
            efficiency[name] = _load_json_if_present(root / name / "compute.json")

    summary = {
        "dataset": dataset,
        "metric": "soft_accuracy" if dataset == "vizwiz" else "exact_match",
        "conditions": {
            name: _condition_summary(rows)
            for name, rows in sorted(rows_by_condition.items())
        },
        "comparisons": comparisons,
        "data_quality": data_quality,
        "efficiency": efficiency,
        "rubric_diagnostic": _load_json_if_present(config.get("rubric_diagnostic")),
        "limitations": [
            "Fixed training seed 42; bootstrap reflects test-example uncertainty only."
        ],
    }
    output = config.get("output")
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _comparison_label(
    summaries: Mapping[str, dict],
    comparison: str,
    positive_label: str,
    negative_label: str,
) -> dict:
    dataset_results = {
        dataset: summary["comparisons"][comparison]
        for dataset, summary in sorted(summaries.items())
    }
    positive = [
        result["delta"] > 0 and result["ci95_low"] > 0
        for result in dataset_results.values()
    ]
    if all(positive):
        label = positive_label
    elif any(result["delta"] > 0 for result in dataset_results.values()):
        label = "mixed"
    else:
        label = negative_label
    return {"label": label, "datasets": dataset_results}


def _claim_assessment(summaries: Mapping[str, dict]) -> dict:
    return {
        "posttraining_vs_filtering": _comparison_label(
            summaries, "pt4_j-filter4", "supported", "not_supported"
        ),
        "penalty_contribution": _comparison_label(
            summaries, "pt4_jp-pt4_j", "positive", "neutral_or_harmful"
        ),
        "rubric8_contribution": _comparison_label(
            summaries, "pt8_jp-pt4_jp", "positive", "not_supported"
        ),
        "data60k_contribution": _comparison_label(
            summaries,
            "pt4_jp_60k-pt4_jp",
            "positive_practical_scaling",
            "not_supported",
        ),
        "seed_limitation": "All training conditions use seed 42; no across-seed robustness claim.",
    }


def combine_dataset_summaries(
    summary_paths: dict[str, Path],
    claim_output: Path | None = None,
) -> dict:
    if len(summary_paths) != len(set(summary_paths)):
        raise ValueError("duplicate dataset names in summaries")
    summaries = {
        dataset: json.loads(Path(path).read_text(encoding="utf-8"))
        for dataset, path in summary_paths.items()
    }
    for dataset, summary in summaries.items():
        missing = [
            comparison
            for comparison in REGISTERED_COMPARISONS
            if comparison not in summary.get("comparisons", {})
        ]
        if missing:
            raise ValueError(f"{dataset} summary missing comparisons: {missing}")
    combined = {
        "datasets": summaries,
        "limitations": [
            "PathVQA and VizWiz metrics are not averaged.",
            "Fixed training seed 42; bootstrap reflects test-example uncertainty only.",
        ],
    }
    if claim_output is not None:
        claim = _claim_assessment(summaries)
        claim_output.parent.mkdir(parents=True, exist_ok=True)
        claim_output.write_text(json.dumps(claim, indent=2), encoding="utf-8")
    return combined
