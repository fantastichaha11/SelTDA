from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf

from data.utils import join_image_root_path
from experiments.synthetic_data import (
    dataset_diagnostics,
    eligible_records,
    rank_global,
    record_identity,
    sha256_json,
    synthetic_quota,
)
from judge.data import load_records
from judge.factory import build_prometheus_scorer, resolve_prometheus_config


def _write_json(path: str | Path, payload) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _output_path(config, key: str, default_name: str) -> Path:
    output_dir = Path(str(config.output.dir))
    value = OmegaConf.select(config, f"output.{key}", default=None)
    return Path(str(value)) if value else output_dir / default_name


def _public_record(row: Mapping) -> dict:
    return {key: value for key, value in dict(row).items() if not str(key).startswith("_")}


def load_score_jsonl(path: str | Path) -> dict[str, dict]:
    score_path = Path(path)
    if not score_path.exists():
        return {}
    rows = {}
    for line in score_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[str(row["identity_hash"])] = row
    return rows


def _append_score(path: Path, row: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        handle.flush()


def _score_key(row: Mapping) -> tuple[str, tuple[str, str, str]]:
    identity = record_identity(row)
    return sha256_json(identity), identity


def score_missing_records(
    records: Sequence[Mapping],
    cache: dict[str, dict],
    scorer,
    *,
    image_root: str | Path,
    scores_jsonl: str | Path,
) -> list[dict]:
    scored = []
    score_path = Path(scores_jsonl)
    for row in records:
        identity_hash, identity = _score_key(row)
        cached = cache.get(identity_hash)
        if cached is None:
            result = scorer.score(
                join_image_root_path(str(image_root), str(row["image"])),
                str(row["question"]),
                str(row["answer"][0] if isinstance(row.get("answer"), list) else row.get("answer", "")),
            )
            cached = {
                "identity_hash": identity_hash,
                "identity": list(identity),
                "source_index": row.get("_source_index"),
                "judge_score": int(result.score),
                "reward": float(result.reward),
                "feedback": result.feedback,
                "raw_text": result.raw_text,
            }
            cache[identity_hash] = cached
            _append_score(score_path, cached)
        scored.append({**dict(row), **cached})
    return scored


def _score_distribution(rows: Sequence[Mapping]) -> dict[str, int]:
    return dict(sorted(Counter(str(int(row["judge_score"])) for row in rows).items()))


def _runtime_config(config):
    judge_config = OmegaConf.select(config, "judge.config", default=None)
    if judge_config is None:
        return None
    return resolve_prometheus_config(
        judge_config,
        selected_model_path=OmegaConf.select(
            config, "judge.selected_model_path", default=None
        ),
        device=str(OmegaConf.select(config, "judge.device", default="cuda")),
    )


def run_filter(config, scorer=None) -> dict:
    real = load_records(config.data.real_annotations)
    quota = synthetic_quota(
        real_count=len(real), target_total=int(config.data.target_total)
    )
    required = int(config.data.pool_multiplier) * quota
    raw = [
        row
        for path in list(config.data.raw_pools)
        for row in load_records(path)
    ]
    eligibility = eligible_records(
        raw, config.data.image_root, dataset=str(config.data.dataset)
    )
    if len(eligibility.records) < required:
        deficit = required - len(eligibility.records)
        raise ValueError(
            f"required eligible pool={required}; "
            f"available={len(eligibility.records)}; deficit={deficit}"
        )

    output_dir = Path(str(config.output.dir))
    selected_path = _output_path(config, "synthetic_json", "selected.json")
    manifest_path = _output_path(config, "manifest", "manifest.json")
    diagnostics_path = _output_path(config, "diagnostics", "diagnostics.json")
    scores_path = _output_path(config, "scores_jsonl", "scores.jsonl")
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = _runtime_config(config)
    if scorer is None and runtime is None:
        raise ValueError("judge.config is required when scorer is not injected")
    scorer = scorer or build_prometheus_scorer(runtime)
    pool = eligibility.records[:required]
    cache = load_score_jsonl(scores_path)
    scored = score_missing_records(
        pool,
        cache,
        scorer,
        image_root=config.data.image_root,
        scores_jsonl=scores_path,
    )
    selected = rank_global(scored, quota=quota, seed=int(config.seed))
    selected_public = [_public_record(row) for row in selected]
    diagnostics = dataset_diagnostics(selected_public)

    _write_json(selected_path, selected_public)
    _write_json(diagnostics_path, diagnostics)
    manifest = {
        "target_total": int(config.data.target_total),
        "real_count": len(real),
        "synthetic_count": len(selected_public),
        "total_count": len(real) + len(selected_public),
        "output_path": str(selected_path),
        "output_sha256": sha256_json(selected_public),
        "eligible_pool_count": len(pool),
        "required_pool_count": required,
        "pool_multiplier": int(config.data.pool_multiplier),
        "seed": int(config.seed),
        "per_image_quota": None,
        "raw_pool_paths": [str(path) for path in list(config.data.raw_pools)],
        "rejections": eligibility.rejections,
        "score_distribution": _score_distribution(scored),
        "diagnostics_path": str(diagnostics_path),
        "scores_jsonl": str(scores_path),
        "judge": None if runtime is None else runtime.to_dict(),
    }
    _write_json(manifest_path, manifest)
    return {
        "real": len(real),
        "eligible_pool": required,
        "synthetic": len(selected_public),
        "total": len(real) + len(selected_public),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--overrides", nargs="+", default=[])
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    if args.overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))
    print(json.dumps(run_filter(config), indent=2))


if __name__ == "__main__":
    main()
