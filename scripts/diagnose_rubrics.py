from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf

from data.utils import join_image_root_path
from experiments.analysis import spearman_correlation
from experiments.synthetic_data import canonical_answer, sha256_json
from judge.data import infer_answer_type, load_records, question_prefix
from judge.factory import (
    build_prometheus_scorer,
    resolve_prometheus_config,
    write_judge_snapshot,
)


def _write_json(path: str | Path, payload) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_key(image: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{image}".encode("utf-8")).hexdigest()


def select_probe_images(records: Sequence[Mapping], image_count: int = 25, seed: int = 42) -> list[dict]:
    by_image = {}
    for record in records:
        image = str(record.get("image", ""))
        if image and image not in by_image:
            by_image[image] = dict(record)
    if len(by_image) < image_count:
        raise ValueError(f"need {image_count} unique images, found {len(by_image)}")
    ordered = sorted(by_image.values(), key=lambda row: _selection_key(str(row["image"]), seed))
    return ordered[:image_count]


def prepare_probe_manifest(
    *,
    validation_annotations: str | Path,
    teacher_checkpoint: str | Path,
    output: str | Path,
    image_count: int = 25,
    seed: int = 42,
) -> dict:
    records = load_records(validation_annotations)
    selected = select_probe_images(records, image_count=image_count, seed=seed)
    manifest = {
        "analysis_only": True,
        "seed": seed,
        "image_count": image_count,
        "validation_annotations": str(validation_annotations),
        "validation_annotations_sha256": _file_sha256(validation_annotations),
        "teacher_checkpoint": str(teacher_checkpoint),
        "teacher_checkpoint_sha256": _file_sha256(teacher_checkpoint),
        "selected_images": [
            {
                "image": str(row["image"]),
                "question_id": row.get("question_id"),
                "question": row.get("question"),
            }
            for row in selected
        ],
    }
    _write_json(output, manifest)
    return manifest


def _validate_groups(rows: Sequence[Mapping], expected_groups: int, expected_group_size: int) -> None:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get("group_id"))].append(row)
    valid = len(groups) == expected_groups and all(
        len(group) == expected_group_size for group in groups.values()
    )
    if not valid:
        raise ValueError(
            f"expected {expected_groups} groups of {expected_group_size}; "
            f"found {len(groups)} groups with sizes {[len(group) for group in groups.values()]}"
        )


def _winner_index(group: Sequence[Mapping], score_key: str) -> int:
    return max(range(len(group)), key=lambda index: (int(group[index][score_key]), -index))


def summarize_rubric_scores(rows: Sequence[Mapping], expected_group_size: int) -> dict:
    groups = defaultdict(list)
    rubric4_scores = []
    rubric8_scores = []
    by_prefix = defaultdict(lambda: {"rubric4": [], "rubric8": []})
    by_answer_type = defaultdict(lambda: {"rubric4": [], "rubric8": []})

    for row in rows:
        groups[str(row["group_id"])].append(row)
        score4 = int(row["rubric4_score"])
        score8 = int(row["rubric8_score"])
        rubric4_scores.append(score4)
        rubric8_scores.append(score8)
        prefix = question_prefix(str(row.get("question", "")))
        answer_type = infer_answer_type(row.get("answer", ""))
        by_prefix[prefix]["rubric4"].append(score4)
        by_prefix[prefix]["rubric8"].append(score8)
        by_answer_type[answer_type]["rubric4"].append(score4)
        by_answer_type[answer_type]["rubric8"].append(score8)

    if any(len(group) != expected_group_size for group in groups.values()):
        raise ValueError("all probe groups must have the expected group size")

    changed = sum(
        _winner_index(group, "rubric4_score") != _winner_index(group, "rubric8_score")
        for group in groups.values()
    )

    def mean(values: Sequence[int]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    def grouped_mean(groups_by_name):
        return {
            name: {
                "mean_rubric4": mean(values["rubric4"]),
                "mean_rubric8": mean(values["rubric8"]),
            }
            for name, values in sorted(groups_by_name.items())
        }

    return {
        "probe_count": len(rows),
        "group_count": len(groups),
        "group_size": expected_group_size,
        "mean_rubric4": mean(rubric4_scores),
        "mean_rubric8": mean(rubric8_scores),
        "mean_paired_delta": mean([b - a for a, b in zip(rubric4_scores, rubric8_scores)]),
        "spearman": spearman_correlation(rubric4_scores, rubric8_scores),
        "winner_change_rate": 0.0 if not groups else changed / len(groups),
        "by_question_prefix": grouped_mean(by_prefix),
        "by_answer_type": grouped_mean(by_answer_type),
    }


def _score_rows(rows: Sequence[Mapping], *, scorer4, scorer8, image_root: str | Path) -> list[dict]:
    scored = []
    for row in rows:
        answer = canonical_answer(row.get("answer", ""))
        result4 = scorer4.score(
            join_image_root_path(str(image_root), str(row.get("image", ""))),
            str(row.get("question", "")),
            answer,
        )
        result8 = scorer8.score(
            join_image_root_path(str(image_root), str(row.get("image", ""))),
            str(row.get("question", "")),
            answer,
        )
        scored.append(
            {
                **dict(row),
                "rubric4_score": int(result4.score),
                "rubric4_feedback": result4.feedback,
                "rubric8_score": int(result8.score),
                "rubric8_feedback": result8.feedback,
            }
        )
    return scored


def run_diagnostic(config, scorer4=None, scorer8=None) -> dict:
    rows = load_records(config.pairs_json)
    expected_groups = int(config.expected_groups)
    expected_group_size = int(config.expected_group_size)
    _validate_groups(rows, expected_groups, expected_group_size)

    output_dir = Path(str(config.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    if scorer4 is None:
        rubric4 = resolve_prometheus_config(
            config.rubric4_config,
            selected_model_path=OmegaConf.select(config, "selected_model_path", default=None),
        )
        write_judge_snapshot(output_dir / "rubric4_judge_snapshot.json", rubric4)
        scorer4 = build_prometheus_scorer(rubric4)
    if scorer8 is None:
        rubric8 = resolve_prometheus_config(
            config.rubric8_config,
            selected_model_path=OmegaConf.select(config, "selected_model_path", default=None),
        )
        write_judge_snapshot(output_dir / "rubric8_judge_snapshot.json", rubric8)
        scorer8 = build_prometheus_scorer(rubric8)

    scored = _score_rows(rows, scorer4=scorer4, scorer8=scorer8, image_root=config.image_root)
    summary = summarize_rubric_scores(scored, expected_group_size=expected_group_size)
    summary["analysis_only"] = True
    summary["seed"] = int(OmegaConf.select(config, "seed", default=42))
    summary["pairs_sha256"] = sha256_json(rows)
    _write_json(output_dir / "scored_pairs.json", scored)
    _write_json(output_dir / "summary.json", summary)
    return summary


def _prepare_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-annotations", required=True)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-count", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    return args


def _score_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-json", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--rubric4-config", required=True)
    parser.add_argument("--rubric8-config", required=True)
    parser.add_argument("--selected-model-path", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-groups", type=int, required=True)
    parser.add_argument("--expected-group-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"prepare", "score"}:
        raise SystemExit("usage: diagnose_rubrics.py prepare|score ...")
    command, argv = sys.argv[1], sys.argv[2:]
    if command == "prepare":
        args = _prepare_args(argv)
        del args.image_root
        manifest = prepare_probe_manifest(
            validation_annotations=args.validation_annotations,
            teacher_checkpoint=args.teacher_checkpoint,
            output=args.output,
            image_count=args.image_count,
            seed=args.seed,
        )
        print(json.dumps(manifest, indent=2))
        return
    args = _score_args(argv)
    summary = run_diagnostic(
        OmegaConf.create(
            {
                "pairs_json": args.pairs_json,
                "image_root": args.image_root,
                "rubric4_config": args.rubric4_config,
                "rubric8_config": args.rubric8_config,
                "selected_model_path": args.selected_model_path,
                "output_dir": args.output_dir,
                "expected_groups": args.expected_groups,
                "expected_group_size": args.expected_group_size,
                "seed": args.seed,
            }
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
