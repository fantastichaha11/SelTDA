"""IT-1: multi-round SelTDA orchestrator via subprocesses."""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from orchestration.skill_gap import SkillGapReport, compute_skill_gap_from_results

logger = logging.getLogger(__name__)


@dataclass
class RoundState:
    round_id: int
    val_accuracy: float
    skill_gap_path: Path
    synthetic_path: Path
    student_ckpt: Path


def save_round_state(state: RoundState, path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "round_id": state.round_id,
                "val_accuracy": state.val_accuracy,
                "skill_gap_path": str(state.skill_gap_path),
                "synthetic_path": str(state.synthetic_path),
                "student_ckpt": str(state.student_ckpt),
            },
            indent=2,
        )
    )


def load_round_state(path: Path) -> RoundState:
    data = json.loads(path.read_text())
    return RoundState(
        round_id=int(data["round_id"]),
        val_accuracy=float(data["val_accuracy"]),
        skill_gap_path=Path(data["skill_gap_path"]),
        synthetic_path=Path(data["synthetic_path"]),
        student_ckpt=Path(data["student_ckpt"]),
    )


def run_subprocess(cmd: list[str], dry_run: bool = False) -> int:
    logger.info("run: %s", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd, check=False).returncode


def run_round(
    round_id: int,
    config: dict,
    *,
    subprocess_runner=run_subprocess,
) -> tuple[float, SkillGapReport]:
    """Execute one iterative round (mockable subprocess runner)."""
    project = Path(config["project_root"])
    python = config.get("python", "python")

    if round_id > 0 and config.get("train_vqg_on_weak", True):
        vqg_cfg = config.get("train_vqg_config")
        if vqg_cfg:
            subprocess_runner(
                [python, str(project / "train_vqg.py"), "--config", vqg_cfg]
            )

    gen_cmd = [
        python,
        str(project / "generate_questions.py"),
        "--config",
        config["generate_config"],
    ]
    if config.get("weak_types_file"):
        gen_cmd.extend(["--overrides", f"weak_types_file={config['weak_types_file']}"])
    subprocess_runner(gen_cmd)

    subprocess_runner(
        [
            python,
            str(project / "filter_pseudo.py"),
            "--config",
            config.get("filter_config", "configs/filter_pseudo_stratified.yaml"),
        ]
    )

    if config.get("staged_curriculum", False):
        from orchestration.merge_pools import merge_with_weights

        easy = json.loads(Path(config["easy_pool"]).read_text())
        hard = json.loads(Path(config["hard_pool"]).read_text())
        merged = merge_with_weights(
            {"easy": (easy, config.get("easy_weight", 1)), "hard": (hard, config.get("hard_weight", 3))}
        )
        out = Path(config["synthetic_path"])
        out.write_text(json.dumps(merged, indent=2))
    else:
        out = Path(config["synthetic_path"])

    subprocess_runner(
        [
            python,
            str(project / "train_vqa.py"),
            "--config",
            config["train_vqa_config"],
            "--overrides",
            f"synthetic_data={out}",
        ]
    )

    results_path = Path(config["results_template"].format(round=round_id))
    subprocess_runner(
        [python, str(project / "evaluate.py"), "--config", config["eval_config"]]
    )

    report = compute_skill_gap_from_results(results_path)
    acc = sum(report.per_type_accuracy.values()) / max(len(report.per_type_accuracy), 1)
    gap_path = Path(config["skill_gap_template"].format(round=round_id))
    gap_path.write_text(
        json.dumps(
            {
                "per_type_accuracy": report.per_type_accuracy,
                "per_type_error_rate": report.per_type_error_rate,
                "weak_types": report.weak_types,
            },
            indent=2,
        )
    )
    return acc, report


def run_iterative_loop(config: dict, *, subprocess_runner=run_subprocess) -> list[RoundState]:
    max_rounds = int(config.get("max_rounds", 2))
    early_stop_delta = float(config.get("early_stop_delta", 0.5))
    states: list[RoundState] = []
    acc_prev = 0.0

    for r in range(max_rounds):
        acc, report = run_round(r, config, subprocess_runner=subprocess_runner)
        state = RoundState(
            round_id=r,
            val_accuracy=acc,
            skill_gap_path=Path(config["skill_gap_template"].format(round=r)),
            synthetic_path=Path(config["synthetic_path"]),
            student_ckpt=Path(config.get("student_ckpt", "cache/student_weights/checkpoint_09.pth")),
        )
        states.append(state)
        if r > 0 and acc - acc_prev < early_stop_delta:
            logger.info("Early stop at round %d (gain %.3f)", r, acc - acc_prev)
            break
        acc_prev = acc
        if report.weak_types:
            config["weak_types_file"] = str(state.skill_gap_path)
    return states
