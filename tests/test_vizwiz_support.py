import json
from pathlib import Path

from PIL import Image

from dataset_adapters.generic_vqa import (
    build_answer_list,
    exact_match_accuracy,
    normalize_answer,
    question_prefix,
    vqa_soft_accuracy,
    write_json,
)
from convert_vizwiz import convert_vizwiz_dataset
from vizwiz_eval import evaluate_vizwiz


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


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), color=(255, 255, 255)).save(path)


def test_convert_vizwiz_dataset_outputs_generic_records(tmp_path):
    root = tmp_path / "vizwiz"
    train_img = root / "images" / "train" / "VizWiz_train_00000001.jpg"
    val_img = root / "images" / "val" / "VizWiz_val_00000001.jpg"
    _write_image(train_img)
    _write_image(val_img)

    ann_dir = root / "annotations"
    ann_dir.mkdir(parents=True)
    train_annotations = [
        {
            "image": "VizWiz_train_00000001.jpg",
            "question": "What drink is this?",
            "answerable": 1,
            "answer_type": "other",
            "answers": [
                {"answer": "Soda", "answer_confidence": "yes"},
                {"answer": "soda", "answer_confidence": "yes"},
                {"answer": "can", "answer_confidence": "maybe"},
            ],
        }
    ]
    val_annotations = [
        {
            "image": "VizWiz_val_00000001.jpg",
            "question": "Can this be answered?",
            "answerable": 0,
            "answer_type": "unanswerable",
            "answers": [
                {"answer": "unanswerable", "answer_confidence": "yes"},
                {"answer": "unanswerable", "answer_confidence": "yes"},
                {"answer": "text", "answer_confidence": "no"},
            ],
        }
    ]
    (ann_dir / "train.json").write_text(json.dumps(train_annotations))
    (ann_dir / "val.json").write_text(json.dumps(val_annotations))

    convert_vizwiz_dataset(root, output_root=root)

    train = json.loads((root / "train.json").read_text())
    val = json.loads((root / "val.json").read_text())
    answer_list = json.loads((root / "answer_list.json").read_text())
    metadata = json.loads((root / "vizwiz_val_metadata.json").read_text())

    assert train == [
        {
            "dataset": "vizwiz",
            "image": "train/VizWiz_train_00000001.jpg",
            "question": "What drink is this?",
            "question_id": 0,
            "answer": ["soda", "soda", "can"],
        }
    ]
    assert val[0]["image"] == "val/VizWiz_val_00000001.jpg"
    assert val[0]["answer"] == ["unanswerable", "unanswerable", "text"]
    assert answer_list == ["can", "soda", "text", "unanswerable"]
    assert metadata["1000000"]["answerable"] == 0
    assert metadata["1000000"]["answer_type"] == "unanswerable"


def test_vizwiz_eval_soft_accuracy_and_strata(tmp_path):
    annotations = [
        {
            "dataset": "vizwiz",
            "image": "val/VizWiz_val_00000001.jpg",
            "question": "What is this?",
            "question_id": 1000000,
            "answer": ["can", "can", "can"],
        },
        {
            "dataset": "vizwiz",
            "image": "val/VizWiz_val_00000002.jpg",
            "question": "Can this be answered?",
            "question_id": 1000001,
            "answer": ["unanswerable", "unanswerable", "text"],
        },
    ]
    metadata = {
        "1000000": {"answerable": 1, "answer_type": "other"},
        "1000001": {"answerable": 0, "answer_type": "unanswerable"},
    }
    results = [
        {"question_id": 1000000, "answer": "can"},
        {"question_id": 1000001, "answer": "unanswerable"},
    ]

    ann_path = tmp_path / "val.json"
    meta_path = tmp_path / "vizwiz_val_metadata.json"
    result_path = tmp_path / "result.json"
    write_json(ann_path, annotations)
    write_json(meta_path, metadata)
    write_json(result_path, results)

    metrics = evaluate_vizwiz(result_path, ann_path, meta_path)

    assert metrics["overall"] == 0.8333
    assert metrics["answerable"] == 1.0
    assert metrics["unanswerable"] == 0.6667
    assert metrics["by_answer_type"]["other"] == 1.0
    assert metrics["by_answer_type"]["unanswerable"] == 0.6667
