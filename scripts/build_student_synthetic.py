from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf

from experiments.synthetic_data import (
    assign_missing_question_ids,
    build_nested_synthetic,
    dataset_diagnostics,
    eligible_records,
    sha256_json,
    synthetic_quota,
)
from judge.data import load_records


def _write_json(path: str | Path, payload) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _max_question_id(records: list[dict], default_start: int) -> int:
    ids = [int(row["question_id"]) for row in records if row.get("question_id") is not None]
    return max(ids, default=default_start - 1)


def _source_records(paths) -> list[dict]:
    return [row for path in list(paths) for row in load_records(path)]


def _manifest_payload(
    *,
    config,
    real_count: int,
    finalized: list[dict],
    rejections: dict,
    base: list[dict] | None,
) -> dict:
    output_path = Path(str(config.output))
    target_total = int(config.target_total)
    manifest = {
        "target_total": target_total,
        "real_count": real_count,
        "synthetic_count": len(finalized),
        "total_count": real_count + len(finalized),
        "output_path": str(output_path),
        "output_sha256": sha256_json(finalized),
        "source_pools": [str(path) for path in list(config.source_pools)],
        "rejections": dict(rejections),
        "diagnostics": dataset_diagnostics(finalized),
    }
    if base is not None:
        base_count = len(base)
        manifest.update(
            {
                "base_synthetic_path": str(config.base_synthetic),
                "base_synthetic_count": base_count,
                "appended_synthetic_count": len(finalized) - base_count,
                "parent_sha256": sha256_json(base),
                "parent_prefix_sha256": sha256_json(finalized[:base_count]),
            }
        )
    return manifest


def build_student_synthetic(config) -> dict:
    real = load_records(config.real_annotations)
    target = int(config.target_total)
    quota = synthetic_quota(real_count=len(real), target_total=target)
    raw = _source_records(config.source_pools)
    eligibility = eligible_records(raw, config.image_root, dataset=str(config.dataset))
    base_path = OmegaConf.select(config, "base_synthetic", default=None)

    base = None
    if base_path:
        base = load_records(base_path)
        selected = build_nested_synthetic(
            base,
            eligibility.records,
            target_synthetic=quota,
            image_root=config.image_root,
        )
        base_count = len(base)
        next_id = _max_question_id(base, int(config.question_id_start)) + 1
        finalized = [dict(row) for row in selected[:base_count]]
        finalized.extend(
            assign_missing_question_ids(
                selected[base_count:],
                start=next_id,
                dataset=str(config.dataset),
            )
        )
    else:
        if len(eligibility.records) < quota:
            raise ValueError(
                f"eligible synthetic {len(eligibility.records)} < required quota {quota}"
            )
        finalized = assign_missing_question_ids(
            eligibility.records[:quota],
            start=int(config.question_id_start),
            dataset=str(config.dataset),
        )

    if len(real) + len(finalized) != target:
        raise AssertionError("loader-visible count invariant failed")

    _write_json(config.output, finalized)
    manifest = _manifest_payload(
        config=config,
        real_count=len(real),
        finalized=finalized,
        rejections=eligibility.rejections,
        base=base,
    )
    _write_json(config.manifest, manifest)
    return {"real": len(real), "synthetic": len(finalized), "total": target}


def verify_manifest(manifest_path: str | Path) -> dict:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    output = load_records(manifest["output_path"])

    if len(output) != int(manifest["synthetic_count"]):
        raise ValueError("synthetic_count mismatch")
    if int(manifest["real_count"]) + len(output) != int(manifest["target_total"]):
        raise ValueError("target_total mismatch")

    if "parent_sha256" in manifest:
        base = load_records(manifest["base_synthetic_path"])
        if sha256_json(base) != manifest["parent_sha256"]:
            raise ValueError("parent_sha256 mismatch")
        base_count = int(manifest["base_synthetic_count"])
        if sha256_json(output[:base_count]) != manifest["parent_prefix_sha256"]:
            raise ValueError("parent_prefix_sha256 mismatch")
        appended = len(output) - base_count
        if appended != int(manifest["appended_synthetic_count"]):
            raise ValueError("appended_synthetic_count mismatch")

    if sha256_json(output) != manifest["output_sha256"]:
        raise ValueError("output_sha256 mismatch")

    return {
        "real": int(manifest["real_count"]),
        "synthetic": len(output),
        "total": int(manifest["target_total"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--verify-manifest", default=None)
    parser.add_argument("--overrides", nargs="+", default=[])
    args = parser.parse_args()

    if args.verify_manifest:
        print(json.dumps(verify_manifest(args.verify_manifest), indent=2))
        return
    if args.config is None:
        parser.error("--config is required unless --verify-manifest is used")

    config = OmegaConf.load(args.config)
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))
    print(json.dumps(build_student_synthetic(config), indent=2))


if __name__ == "__main__":
    main()
