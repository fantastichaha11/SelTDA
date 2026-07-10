from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf

from judge.data import build_answer_pools, build_judge_pairs, load_records
from judge.prometheus import build_prometheus_prompt


def _write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sft_row(example_id: str, image: str, question: str, answer: str, score: int) -> dict:
    prompt = build_prometheus_prompt(question=question, candidate_answer=answer)
    return {
        "id": example_id,
        "image": image,
        "conversations": [
            {"from": "human", "value": "<image>\n" + prompt},
            {"from": "gpt", "value": f"Feedback: PathVQA train-derived supervision. [RESULT] {score}"},
        ],
    }


def build_training_artifacts(config) -> dict[str, int]:
    train_records = load_records(config.data.train_annotations)
    pools = build_answer_pools(train_records)
    pairs = build_judge_pairs(
        train_records,
        pools=pools,
        seed=int(config.train.seed),
        negatives_per_positive=int(config.data.negatives_per_positive),
    )
    pair_rows: list[dict] = []
    sft_rows: list[dict] = []
    for idx, pair in enumerate(pairs):
        pair_rows.append(
            {
                "id": f"pathvqa-train-{idx}",
                "image": pair.positive.image,
                "question": pair.positive.question,
                "positive_answer": pair.positive.answer,
                "negative_answer": pair.negative.answer,
                "positive_score": 5,
                "negative_score": 1,
            }
        )
        sft_rows.append(
            _sft_row(
                f"pathvqa-train-{idx}-pos",
                pair.positive.image,
                pair.positive.question,
                pair.positive.answer,
                5,
            )
        )
        sft_rows.append(
            _sft_row(
                f"pathvqa-train-{idx}-neg",
                pair.negative.image,
                pair.negative.question,
                pair.negative.answer,
                1,
            )
        )

    _write_jsonl(config.train.train_pairs_jsonl, pair_rows)
    sft_path = Path(str(config.train.prometheus_sft_json))
    sft_path.parent.mkdir(parents=True, exist_ok=True)
    sft_path.write_text(json.dumps(sft_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"num_pairs": len(pairs), "num_sft_examples": len(sft_rows)}


def build_external_command(config) -> list[str]:
    resolved = OmegaConf.to_container(config, resolve=True)
    command_cfg = resolved["train"]["external_command"]
    return [
        str(command_cfg["executable"]),
        str(command_cfg["script"]),
        *[str(item) for item in command_cfg.get("extra_args", [])],
    ]


def _command_arg(command: list[str], name: str) -> str | None:
    try:
        return command[command.index(name) + 1]
    except (ValueError, IndexError):
        return None


def validate_external_llava_command(command: list[str]) -> None:
    version = _command_arg(command, "--version")
    if version in {"plain", "v0_plain"}:
        raise ValueError(
            "Prometheus judge SFT data needs a chat template such as vicuna_v1; "
            "LLaVA plain mode drops the rubric and question prompt."
        )


def run_train(config) -> dict[str, object]:
    backend = str(config.train.backend)
    if backend not in {"dry_run", "external_llava"}:
        raise ValueError("train.backend must be dry_run or external_llava")

    summary = build_training_artifacts(config)
    if backend == "dry_run":
        return {"backend": backend, **summary}

    command = build_external_command(config)
    validate_external_llava_command(command)
    subprocess.run(command, check=True)
    return {"backend": backend, "command": command, **summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/prometheus_judge_pathvqa.yaml")
    parser.add_argument("--backend", default=None, choices=["dry_run", "external_llava"])
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    if args.backend is not None:
        config.train.backend = args.backend

    result = run_train(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
