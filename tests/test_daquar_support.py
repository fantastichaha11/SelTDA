import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from convert_daquar import VAL_ID_OFFSET, convert_daquar_dataset
from daquar_eval import evaluate_daquar


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), color=(255, 255, 255)).save(path)


def test_convert_daquar_dataset_outputs_generic_records_from_json(tmp_path):
    root = tmp_path / "daquar"
    image_root = root / "images"
    _write_image(image_root / "train_room.png")
    _write_image(image_root / "val_room.jpg")

    train_records = [
        {
            "image": "train_room",
            "question": "What is on the table?",
            "answer": "The Chair!",
        }
    ]
    test_records = [
        {
            "img": "val_room.jpg",
            "question_str": "How many lamps are there?",
            "answers": ["Two"],
        }
    ]
    (root / "qa_train.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "qa_train.json").write_text(json.dumps(train_records))
    (root / "qa_test.json").write_text(json.dumps(test_records))

    convert_daquar_dataset(root)

    train = json.loads((root / "train.json").read_text())
    val = json.loads((root / "val.json").read_text())
    answer_list = json.loads((root / "answer_list.json").read_text())
    metadata = json.loads((root / "daquar_val_metadata.json").read_text())

    assert train == [
        {
            "dataset": "daquar",
            "image": "train_room.png",
            "question": "What is on the table?",
            "question_id": 0,
            "answer": ["the chair"],
        }
    ]
    assert val == [
        {
            "dataset": "daquar",
            "image": "val_room.jpg",
            "question": "How many lamps are there?",
            "question_id": VAL_ID_OFFSET,
            "answer": ["two"],
        }
    ]
    assert answer_list == ["the chair", "two"]
    assert metadata == {
        str(VAL_ID_OFFSET): {
            "answer": "two",
            "question": "How many lamps are there?",
        }
    }


def test_convert_daquar_dataset_outputs_generic_records_from_csv(tmp_path):
    root = tmp_path / "daquar"
    image_root = root / "images"
    _write_image(image_root / "train_image.png")
    _write_image(image_root / "test_image.png")

    train_qa = root / "custom_train.csv"
    test_qa = root / "custom_test.csv"
    for path, row in [
        (train_qa, ["train_image", "What color is the bed?", "Blue."]),
        (test_qa, ["test_image", "Is there a window?", "yes"]),
    ]:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Image", "Question", "Answer"])
            writer.writerow(row)

    output_root = tmp_path / "converted"
    convert_daquar_dataset(root, train_qa=train_qa, test_qa=test_qa, output_root=output_root)

    train = json.loads((output_root / "train.json").read_text())
    val = json.loads((output_root / "val.json").read_text())

    assert train[0]["image"] == "train_image.png"
    assert train[0]["answer"] == ["blue"]
    assert train[0]["question_id"] == 0
    assert val[0]["image"] == "test_image.png"
    assert val[0]["answer"] == ["yes"]
    assert val[0]["question_id"] == VAL_ID_OFFSET


def test_convert_daquar_dataset_resolves_numeric_image_ids(tmp_path):
    root = tmp_path / "daquar"
    image_root = root / "images"
    _write_image(image_root / "image3.png")
    _write_image(image_root / "image4.png")

    train_records = [{"image_id": 3, "question": "What is visible?", "answer": "chair"}]
    test_records = [{"image_id": 4, "question": "Where is the desk?", "answer": "office"}]
    (root / "qa_train.json").write_text(json.dumps(train_records))
    (root / "qa_test.json").write_text(json.dumps(test_records))

    convert_daquar_dataset(root)

    train = json.loads((root / "train.json").read_text())
    val = json.loads((root / "val.json").read_text())

    assert train[0]["image"] == "image3.png"
    assert val[0]["image"] == "image4.png"


def test_convert_daquar_dataset_outputs_generic_records_from_raw_text(tmp_path):
    root = tmp_path / "daquar"
    image_root = root / "images"
    _write_image(image_root / "image3.png")
    _write_image(image_root / "image4.png")

    train_qa = root / "qa.37.raw.train.txt"
    test_qa = root / "qa.37.raw.test.txt"
    train_qa.write_text(
        "\n".join(
            [
                "what is on the right side of the black telephone in the image3 ?",
                "desk",
            ]
        )
    )
    test_qa.write_text(
        "\n".join(
            [
                "how many bottles are on the desk in the image4 ?",
                "11",
            ]
        )
    )

    convert_daquar_dataset(root, train_qa=train_qa, test_qa=test_qa)

    train = json.loads((root / "train.json").read_text())
    val = json.loads((root / "val.json").read_text())

    assert train[0] == {
        "dataset": "daquar",
        "image": "image3.png",
        "question": "what is on the right side of the black telephone?",
        "question_id": 0,
        "answer": ["desk"],
    }
    assert val[0]["image"] == "image4.png"
    assert val[0]["question"] == "how many bottles are on the desk?"
    assert val[0]["question_id"] == VAL_ID_OFFSET
    assert val[0]["answer"] == ["11"]


def test_daquar_eval_exact_match_and_question_prefix(tmp_path):
    annotations = [
        {
            "question_id": 1,
            "question": "What is on the table?",
            "image": "a.png",
            "dataset": "daquar",
            "answer": ["book"],
        },
        {
            "question_id": 2,
            "question": "Where is the chair?",
            "image": "b.png",
            "dataset": "daquar",
            "answer": ["kitchen"],
        },
    ]
    results = [
        {"question_id": 1, "answer": "Book!"},
        {"question_id": 2, "answer": "bedroom"},
    ]
    ann_path = tmp_path / "val.json"
    result_path = tmp_path / "vqa_result.json"
    ann_path.write_text(json.dumps(annotations))
    result_path.write_text(json.dumps(results))

    metrics = evaluate_daquar(result_path, ann_path)

    assert metrics["overall"] == 0.5
    assert metrics["by_question_prefix"]["what"] == 1.0
    assert metrics["by_question_prefix"]["where"] == 0.0


def test_daquar_eval_allows_duplicate_predictions_with_same_answer(tmp_path):
    annotations = [
        {
            "question_id": 1,
            "question": "What is on the table?",
            "image": "a.png",
            "dataset": "daquar",
            "answer": ["book"],
        }
    ]
    results = [
        {"question_id": 1, "answer": "Book!"},
        {"question_id": 1, "answer": "book"},
    ]
    ann_path = tmp_path / "val.json"
    result_path = tmp_path / "vqa_result.json"
    ann_path.write_text(json.dumps(annotations))
    result_path.write_text(json.dumps(results))

    assert evaluate_daquar(result_path, ann_path)["overall"] == 1.0


def test_daquar_eval_rejects_conflicting_duplicate_predictions(tmp_path):
    annotations = [
        {
            "question_id": 1,
            "question": "What is on the table?",
            "image": "a.png",
            "dataset": "daquar",
            "answer": ["book"],
        }
    ]
    results = [
        {"question_id": 1, "answer": "book"},
        {"question_id": 1, "answer": "chair"},
    ]
    ann_path = tmp_path / "val.json"
    result_path = tmp_path / "vqa_result.json"
    ann_path.write_text(json.dumps(annotations))
    result_path.write_text(json.dumps(results))

    with pytest.raises(ValueError, match="Conflicting duplicate prediction"):
        evaluate_daquar(result_path, ann_path)
