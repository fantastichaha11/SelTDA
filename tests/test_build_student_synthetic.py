import json

from omegaconf import OmegaConf

from experiments.synthetic_data import sha256_json
from scripts.build_student_synthetic import build_student_synthetic, verify_manifest


def test_builder_outputs_loader_compatible_exact_total(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    real = [{"image": "real.jpg", "question": "R?", "answer": ["r"], "question_id": 1}]
    raw = []
    for index in range(4):
        (images / f"{index}.jpg").touch()
        raw.append({"image": f"{index}.jpg", "question": f"Q{index}?", "answer": "yes"})
    (tmp_path / "real.json").write_text(json.dumps(real), encoding="utf-8")
    (tmp_path / "raw.json").write_text(json.dumps(raw), encoding="utf-8")
    cfg = OmegaConf.create(
        {
            "real_annotations": str(tmp_path / "real.json"),
            "source_pools": [str(tmp_path / "raw.json")],
            "image_root": str(images),
            "dataset": "vizwiz",
            "target_total": 4,
            "output": str(tmp_path / "synthetic.json"),
            "manifest": str(tmp_path / "manifest.json"),
            "question_id_start": 10_000_000,
        }
    )

    result = build_student_synthetic(cfg)

    synthetic = json.loads((tmp_path / "synthetic.json").read_text(encoding="utf-8"))
    assert result == {"real": 1, "synthetic": 3, "total": 4}
    assert all(isinstance(row["answer"], list) for row in synthetic)
    assert [row["question_id"] for row in synthetic] == [
        10_000_000,
        10_000_001,
        10_000_002,
    ]


def test_nested_builder_preserves_parent_prefix_and_hash(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    real = [{"image": "real.jpg", "question": "R?", "answer": ["r"], "question_id": 1}]
    base = [
        {"image": "a.jpg", "question": "Q1?", "answer": ["a"], "question_id": 7},
        {"image": "b.jpg", "question": "Q2?", "answer": ["b"], "question_id": 8},
    ]
    extras = [
        {"image": "a.jpg", "question": "Q1?", "answer": ["a"]},
        {"image": "c.jpg", "question": "Q3?", "answer": "c"},
        {"image": "d.jpg", "question": "Q4?", "answer": "d"},
        {"image": "e.jpg", "question": "Q5?", "answer": "e"},
    ]
    for name in ("c.jpg", "d.jpg", "e.jpg"):
        (images / name).touch()
    (tmp_path / "real.json").write_text(json.dumps(real), encoding="utf-8")
    (tmp_path / "base.json").write_text(json.dumps(base), encoding="utf-8")
    (tmp_path / "extras.json").write_text(json.dumps(extras), encoding="utf-8")
    cfg = OmegaConf.create(
        {
            "real_annotations": str(tmp_path / "real.json"),
            "source_pools": [str(tmp_path / "extras.json")],
            "base_synthetic": str(tmp_path / "base.json"),
            "image_root": str(images),
            "dataset": "pathvqa",
            "target_total": 5,
            "output": str(tmp_path / "synthetic_60k.json"),
            "manifest": str(tmp_path / "manifest.json"),
            "question_id_start": 10_000_000,
        }
    )

    result = build_student_synthetic(cfg)

    synthetic = json.loads((tmp_path / "synthetic_60k.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert result == {"real": 1, "synthetic": 4, "total": 5}
    assert synthetic[:2] == base
    assert len(synthetic[2:]) == 2
    assert [row["question_id"] for row in synthetic] == [7, 8, 9, 10]
    assert manifest["parent_sha256"] == sha256_json(base)
    assert manifest["base_synthetic_count"] == 2
    assert manifest["appended_synthetic_count"] == 2


def test_verify_manifest_rejects_output_tampering(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "0.jpg").touch()
    (tmp_path / "real.json").write_text(json.dumps([]), encoding="utf-8")
    (tmp_path / "raw.json").write_text(
        json.dumps([{"image": "0.jpg", "question": "Q?", "answer": "yes"}]),
        encoding="utf-8",
    )
    cfg = OmegaConf.create(
        {
            "real_annotations": str(tmp_path / "real.json"),
            "source_pools": [str(tmp_path / "raw.json")],
            "image_root": str(images),
            "dataset": "pathvqa",
            "target_total": 1,
            "output": str(tmp_path / "synthetic.json"),
            "manifest": str(tmp_path / "manifest.json"),
            "question_id_start": 10,
        }
    )
    build_student_synthetic(cfg)
    synthetic = json.loads((tmp_path / "synthetic.json").read_text(encoding="utf-8"))
    synthetic[0]["question"] = "Tampered?"
    (tmp_path / "synthetic.json").write_text(json.dumps(synthetic), encoding="utf-8")

    try:
        verify_manifest(tmp_path / "manifest.json")
    except ValueError as exc:
        assert "output_sha256" in str(exc)
    else:
        raise AssertionError("tampered output must fail verification")


def test_verify_manifest_rejects_parent_prefix_tampering(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    real = [{"image": "real.jpg"}]
    base = [{"image": "a.jpg", "question": "Q1?", "answer": ["a"], "question_id": 7}]
    extras = [{"image": "b.jpg", "question": "Q2?", "answer": "b"}]
    (images / "b.jpg").touch()
    (tmp_path / "real.json").write_text(json.dumps(real), encoding="utf-8")
    (tmp_path / "base.json").write_text(json.dumps(base), encoding="utf-8")
    (tmp_path / "extras.json").write_text(json.dumps(extras), encoding="utf-8")
    cfg = OmegaConf.create(
        {
            "real_annotations": str(tmp_path / "real.json"),
            "source_pools": [str(tmp_path / "extras.json")],
            "base_synthetic": str(tmp_path / "base.json"),
            "image_root": str(images),
            "dataset": "pathvqa",
            "target_total": 3,
            "output": str(tmp_path / "synthetic_60k.json"),
            "manifest": str(tmp_path / "manifest.json"),
            "question_id_start": 10,
        }
    )
    build_student_synthetic(cfg)
    synthetic = json.loads((tmp_path / "synthetic_60k.json").read_text(encoding="utf-8"))
    synthetic[0]["question"] = "Tampered parent?"
    (tmp_path / "synthetic_60k.json").write_text(json.dumps(synthetic), encoding="utf-8")

    try:
        verify_manifest(tmp_path / "manifest.json")
    except ValueError as exc:
        assert "parent_prefix_sha256" in str(exc)
    else:
        raise AssertionError("tampered parent prefix must fail verification")
