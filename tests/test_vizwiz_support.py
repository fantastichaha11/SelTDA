import json
from pathlib import Path

from dataset_adapters.generic_vqa import (
    build_answer_list,
    exact_match_accuracy,
    normalize_answer,
    question_prefix,
    vqa_soft_accuracy,
    write_json,
)


def test_normalize_answer_basic_punctuation_and_case():
    assert normalize_answer("  The Bottle!!! ") == "the bottle"
    assert normalize_answer("NO.") == "no"
    assert normalize_answer("") == ""


def test_build_answer_list_is_deterministic():
    records = [
        {"answer": ["zebra", "apple"]},
        {"answer": ["apple", "bottle"]},
    ]
    assert build_answer_list(records) == ["apple", "bottle", "zebra"]


def test_vqa_soft_accuracy_uses_reference_count_over_three():
    refs = ["cat", "cat", "dog", "cat", "horse"]
    assert vqa_soft_accuracy("cat", refs) == 1.0
    assert vqa_soft_accuracy("dog", refs) == 1 / 3
    assert vqa_soft_accuracy("bird", refs) == 0.0


def test_exact_match_accuracy_normalizes_answers():
    predictions = {
        1: "Bottle!",
        2: "wrong",
    }
    references = {
        1: ["bottle"],
        2: ["chair"],
    }
    assert exact_match_accuracy(predictions, references) == 0.5


def test_question_prefix_buckets_common_forms():
    assert question_prefix("How many cans are there?") == "how many"
    assert question_prefix("Where is the label?") == "where"
    assert question_prefix("Is this readable?") == "is/are"
    assert question_prefix("Name the object") == "other"


def test_write_json_creates_parent_directory(tmp_path):
    out = tmp_path / "nested" / "data.json"
    write_json(out, [{"a": 1}])
    assert json.loads(out.read_text()) == [{"a": 1}]
