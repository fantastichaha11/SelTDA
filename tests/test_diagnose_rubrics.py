import json

from omegaconf import OmegaConf

from judge.prometheus import StaticPrometheusScorer
from scripts.diagnose_rubrics import (
    prepare_probe_manifest,
    run_diagnostic,
    select_probe_images,
    summarize_rubric_scores,
)


def test_summary_reports_changed_group_winner():
    rows = [
        {"group_id": "a", "rubric4_score": 5, "rubric8_score": 3},
        {"group_id": "a", "rubric4_score": 4, "rubric8_score": 5},
        {"group_id": "b", "rubric4_score": 5, "rubric8_score": 5},
        {"group_id": "b", "rubric4_score": 4, "rubric8_score": 4},
    ]

    summary = summarize_rubric_scores(rows, expected_group_size=2)

    assert summary["group_count"] == 2
    assert summary["winner_change_rate"] == 0.5


def test_run_diagnostic_rejects_wrong_group_shape(tmp_path):
    pairs = [
        {"group_id": "a", "image": "a.jpg", "question": "Q1?", "answer": "yes"},
        {"group_id": "a", "image": "a.jpg", "question": "Q2?", "answer": "no"},
        {"group_id": "b", "image": "b.jpg", "question": "Q3?", "answer": "yes"},
    ]
    pair_path = tmp_path / "pairs.json"
    pair_path.write_text(json.dumps(pairs), encoding="utf-8")
    cfg = OmegaConf.create(
        {
            "pairs_json": str(pair_path),
            "image_root": str(tmp_path),
            "output_dir": str(tmp_path / "out"),
            "expected_groups": 2,
            "expected_group_size": 2,
            "seed": 42,
        }
    )

    try:
        run_diagnostic(
            cfg,
            scorer4=StaticPrometheusScorer(score=4),
            scorer8=StaticPrometheusScorer(score=5),
        )
    except ValueError as exc:
        assert "expected 2 groups of 2" in str(exc)
    else:
        raise AssertionError("wrong probe group shape must fail")


def test_probe_image_selection_is_stable_and_manifest_is_analysis_only(tmp_path):
    annotations = [
        {
            "image": f"{index}.jpg",
            "question": f"Q{index}?",
            "answer": "yes",
            "question_id": index,
        }
        for index in range(30)
    ]
    annotations.append({**annotations[0], "question_id": 999})
    ann_path = tmp_path / "val.json"
    ckpt_path = tmp_path / "teacher.pth"
    output_path = tmp_path / "probe_images.json"
    ann_path.write_text(json.dumps(annotations), encoding="utf-8")
    ckpt_path.write_text("checkpoint", encoding="utf-8")

    selected = select_probe_images(annotations, image_count=25, seed=42)
    selected_reversed = select_probe_images(list(reversed(annotations)), image_count=25, seed=42)
    manifest = prepare_probe_manifest(
        validation_annotations=ann_path,
        teacher_checkpoint=ckpt_path,
        output=output_path,
        image_count=25,
        seed=42,
    )

    assert [row["image"] for row in selected] == [
        row["image"] for row in selected_reversed
    ]
    assert len({row["image"] for row in selected}) == 25
    assert manifest["analysis_only"] is True
    assert manifest["seed"] == 42
    assert len(manifest["selected_images"]) == 25
    assert "validation_annotations_sha256" in manifest
    assert "teacher_checkpoint_sha256" in manifest
    assert json.loads(output_path.read_text(encoding="utf-8")) == manifest
