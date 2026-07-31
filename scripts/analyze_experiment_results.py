from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.analysis import analyze_matrix, combine_dataset_summaries


def _key_value(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {value}")
    key, path = value.split("=", 1)
    if not key:
        raise argparse.ArgumentTypeError(f"empty name in {value}")
    return key, path


def _comparison(value: str) -> tuple[str, str]:
    if "," not in value:
        raise argparse.ArgumentTypeError(f"expected LEFT,RIGHT, got {value}")
    left, right = value.split(",", 1)
    return left, right


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["pathvqa", "vizwiz"], default=None)
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--metadata-file", default=None)
    parser.add_argument("--condition", action="append", type=_key_value, default=[])
    parser.add_argument("--synthetic", action="append", type=_key_value, default=[])
    parser.add_argument("--comparison", action="append", type=_comparison, default=[])
    parser.add_argument("--rubric-diagnostic", default=None)
    parser.add_argument("--compute-root", default=None)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--combine-summary", action="append", type=_key_value, default=[])
    parser.add_argument("--claim-output", default=None)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.combine_summary:
        combined = combine_dataset_summaries(
            {dataset: Path(path) for dataset, path in args.combine_summary},
            claim_output=Path(args.claim_output) if args.claim_output else None,
        )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(combined, indent=2), encoding="utf-8")
        return

    if args.dataset is None or args.annotations is None:
        raise SystemExit("--dataset and --annotations are required outside combine mode")
    metadata = args.metadata_file
    if args.dataset == "vizwiz" and metadata is None:
        metadata = "datasets/vizwiz/vizwiz_test_metadata.json"
    summary = analyze_matrix(
        {
            "dataset": args.dataset,
            "annotations": args.annotations,
            "metadata": metadata,
            "conditions": dict(args.condition),
            "synthetics": dict(args.synthetic),
            "comparisons": list(args.comparison),
            "rubric_diagnostic": args.rubric_diagnostic,
            "compute_root": args.compute_root,
            "bootstrap_resamples": args.bootstrap_resamples,
            "seed": args.seed,
            "output": args.output,
        }
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
