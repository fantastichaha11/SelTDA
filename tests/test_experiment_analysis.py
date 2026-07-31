import json

from dataset_adapters.generic_vqa import write_json
from experiments.analysis import (
    paired_bootstrap_delta,
    pathvqa_per_example,
    spearman_correlation,
    vizwiz_per_example,
)
from pathvqa_eval import exact_match_eval
from vizwiz_eval import evaluate_vizwiz


def test_paired_bootstrap_is_deterministic_and_absolute():
    left = {1: 1.0, 2: 1.0, 3: 0.0, 4: 1.0}
    right = {1: 0.0, 2: 1.0, 3: 0.0, 4: 0.0}

    first = paired_bootstrap_delta(left, right, n_resamples=1000, seed=42)
    second = paired_bootstrap_delta(left, right, n_resamples=1000, seed=42)

    assert first == second
    assert first["delta"] == 0.5
    assert first["n"] == 4


def test_spearman_handles_tied_scores():
    assert spearman_correlation([1, 2, 2, 4], [1, 3, 3, 4]) == 1.0


def test_pathvqa_per_example_preserves_legacy_aggregate(tmp_path):
    annotations = [
        {
            "dataset": "pathvqa",
            "image": "a.jpg",
            "question": "Is this benign?",
            "question_id": 1,
            "answer": "yes",
            "question_type": "yes/no",
            "answer_type": "closed",
        },
        {
            "dataset": "pathvqa",
            "image": "b.jpg",
            "question": "What is shown?",
            "question_id": 2,
            "answer": "cells",
            "question_type": "open",
            "answer_type": "open",
        },
    ]
    results = [
        {"question_id": 1, "answer": "yes"},
        {"question_id": 2, "answer": "wrong"},
    ]
    ann_path = tmp_path / "test.json"
    result_path = tmp_path / "vqa_result.json"
    write_json(ann_path, annotations)
    write_json(result_path, results)

    rows = pathvqa_per_example(ann_path, result_path)
    metrics = exact_match_eval(ann_path, result_path)

    assert rows == [
        {
            "question_id": 1,
            "score": 1.0,
            "prediction": "yes",
            "true_answer": "yes",
            "question_type": "yes/no",
            "answer_type": "closed",
        },
        {
            "question_id": 2,
            "score": 0.0,
            "prediction": "wrong",
            "true_answer": "cells",
            "question_type": "open",
            "answer_type": "open",
        },
    ]
    assert metrics["overall"] == 0.5
    assert "per_example" not in metrics


def test_vizwiz_per_example_preserves_legacy_aggregate(tmp_path):
    annotations = [
        {
            "dataset": "vizwiz",
            "image": "val/a.jpg",
            "question": "What is this?",
            "question_id": 10,
            "answer": ["can", "can", "can"],
        },
        {
            "dataset": "vizwiz",
            "image": "val/b.jpg",
            "question": "Can this be answered?",
            "question_id": 11,
            "answer": ["unanswerable", "unanswerable", "text"],
        },
    ]
    metadata = {
        "10": {"answerable": 1, "answer_type": "other"},
        "11": {"answerable": 0, "answer_type": "unanswerable"},
    }
    results = [
        {"question_id": 10, "answer": "can"},
        {"question_id": 11, "answer": "unanswerable"},
    ]
    ann_path = tmp_path / "val.json"
    meta_path = tmp_path / "metadata.json"
    result_path = tmp_path / "result.json"
    write_json(ann_path, annotations)
    write_json(meta_path, metadata)
    write_json(result_path, results)

    rows = vizwiz_per_example(result_path, ann_path, meta_path)
    metrics = evaluate_vizwiz(result_path, ann_path, meta_path)

    assert rows == [
        {
            "question_id": 10,
            "score": 1.0,
            "prediction": "can",
            "answer_type": "other",
            "answerable": 1,
        },
        {
            "question_id": 11,
            "score": 2 / 3,
            "prediction": "unanswerable",
            "answer_type": "unanswerable",
            "answerable": 0,
        },
    ]
    assert metrics["overall"] == 0.8333
    assert "per_example" not in metrics
