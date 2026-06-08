import json
import tempfile
from pathlib import Path

from scripts.convert_aokvqa_to_clip_flant5 import (
    coco_image_relpath,
    convert_records,
    pick_answer,
    record_to_llava,
)


def test_pick_answer_mode():
    assert pick_answer(["bus", "taxi", "bus"]) == "bus"
    assert pick_answer("yes") == "yes"
    assert pick_answer([]) == ""


def test_coco_image_relpath():
    assert coco_image_relpath("0000001.jpg", image_prefix="coco/train2017") == (
        "coco/train2017/0000001.jpg"
    )


def test_record_to_llava_vqa_mode():
    rec = {
        "question": "What color?",
        "answer": ["red", "red", "blue"],
        "image": "0000001.jpg",
        "question_id": "abc",
    }
    out = record_to_llava(
        rec,
        mode="vqa",
        image_prefix="coco/train2017",
        vqascore_template='Does this figure show "{qa}"?',
    )
    assert out is not None
    assert out["image"] == "coco/train2017/0000001.jpg"
    assert out["conversations"][0]["value"].startswith("<image>\nWhat color?")
    assert out["conversations"][1]["value"] == "red"


def test_record_to_llava_vqascore_mode():
    rec = {
        "question": "Is it red?",
        "answer": ["yes"],
        "image": "0000002.jpg",
        "question_id": 1,
    }
    out = record_to_llava(
        rec,
        mode="vqascore",
        image_prefix="coco/train2017",
        vqascore_template='Does this figure show "{qa}"? Please answer yes or no.',
    )
    assert "Is it red? yes" in out["conversations"][0]["value"]
    assert out["conversations"][1]["value"] == "Yes"


def test_convert_records_skips_missing_images(tmp_path):
    vqa_root = tmp_path / "coco"
    img_dir = vqa_root / "train2017"
    img_dir.mkdir(parents=True)
    (img_dir / "exists.jpg").write_bytes(b"fake")

    records = [
        {
            "question": "Q1",
            "answer": ["a"],
            "image": "exists.jpg",
            "question_id": 1,
        },
        {
            "question": "Q2",
            "answer": ["b"],
            "image": "missing.jpg",
            "question_id": 2,
        },
    ]
    out, skipped = convert_records(
        records,
        mode="vqa",
        image_prefix="coco/train2017",
        vqascore_template="",
        vqa_root=vqa_root,
        skip_missing_images=True,
    )
    assert len(out) == 1
    assert skipped == 1


def test_convert_script_roundtrip(tmp_path):
    ann = tmp_path / "ann"
    ann.mkdir()
    vqa_root = tmp_path / "images"
    vqa_root.mkdir()
    (vqa_root / "img.jpg").write_bytes(b"x")
    ann_file = ann / "train.json"
    ann_file.write_text(
        json.dumps(
            [
                {
                    "dataset": "aokvqa",
                    "image": "img.jpg",
                    "question": "How many?",
                    "question_id": "1",
                    "answer": ["two"],
                }
            ]
        )
    )
    out, skipped = convert_records(
        json.loads(ann_file.read_text()),
        mode="vqa",
        image_prefix="",
        vqascore_template="",
        vqa_root=vqa_root,
        skip_missing_images=True,
    )
    assert len(out) == 1
    assert out[0]["conversations"][1]["value"] == "two"
