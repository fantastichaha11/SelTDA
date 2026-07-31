from experiments.synthetic_data import (
    build_nested_synthetic,
    dataset_diagnostics,
    eligible_records,
    rank_global,
    record_identity,
    sha256_json,
    synthetic_quota,
)


def test_eligibility_deduplicates_only_within_same_image(tmp_path):
    (tmp_path / "a.jpg").touch()
    (tmp_path / "b.jpg").touch()
    rows = [
        {"image": f"{tmp_path.name}/a.jpg", "question": "What?", "answer": ["Cells"]},
        {"image": "a.jpg", "question": " what ", "answer": "cells"},
        {"image": "b.jpg", "question": "What?", "answer": "cells"},
        {"image": "missing.jpg", "question": "Q?", "answer": "A"},
    ]

    result = eligible_records(rows, tmp_path, dataset="pathvqa")

    assert len(result.records) == 2
    assert result.rejections == {"duplicate": 1, "missing_image": 1}


def test_global_ties_are_independent_of_input_order():
    rows = [
        {"image": f"{i}.jpg", "question": "Q?", "answer": ["a"], "judge_score": 4}
        for i in range(4)
    ]

    first = rank_global(rows, quota=2, seed=42)
    second = rank_global(list(reversed(rows)), quota=2, seed=42)

    assert [record_identity(row) for row in first] == [
        record_identity(row) for row in second
    ]


def test_nested_synthetic_keeps_prefix_and_exact_deficit():
    base = [{"image": "a.jpg", "question": "Q1?", "answer": ["a"]}]
    extra = [
        {"image": "a.jpg", "question": "Q1?", "answer": ["a"]},
        {"image": "b.jpg", "question": "Q2?", "answer": ["b"]},
        {"image": "c.jpg", "question": "Q3?", "answer": ["c"]},
    ]

    nested = build_nested_synthetic(base, extra, target_synthetic=3)

    assert nested[:1] == base
    assert len(nested) == 3
    assert synthetic_quota(real_count=12, target_total=40) == 28


def test_diagnostics_and_hash_are_deterministic():
    rows = [
        {"image": "a.jpg", "question": "Is it red?", "answer": ["yes"]},
        {"image": "a.jpg", "question": "Is it red?", "answer": ["yes"]},
        {"image": "b.jpg", "question": "What is shown?", "answer": ["cells"]},
        {"image": "c.jpg", "question": "Can this be read?", "answer": ["unanswerable"]},
    ]

    diagnostics = dataset_diagnostics(rows)

    assert diagnostics["count"] == 4
    assert diagnostics["unique_images"] == 3
    assert diagnostics["duplicate_question_rate"] == 0.25
    assert diagnostics["duplicate_qa_rate"] == 0.25
    assert diagnostics["question_prefix_distribution"]["is/are"] == 2
    assert diagnostics["answer_type_distribution"]["yes/no"] == 2
    assert diagnostics["generic_answer_rate"] == 0.25
    assert diagnostics["unanswerable_rate"] == 0.25
    assert sha256_json({"b": 1, "a": 2}) == sha256_json({"a": 2, "b": 1})
