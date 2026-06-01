"""IT loop: accumulating-teacher GRPO + restarting-student, round-level stop."""

from __future__ import annotations

import json
from pathlib import Path


def should_continue(acc_curr: float, acc_prev: float, delta: float) -> bool:
    return (acc_curr - acc_prev) >= delta


def kl_beta_for_round(beta0: float, gamma: float, r: int) -> float:
    """kl_beta_r = beta0 * gamma^(r-1). Round index r is 1-based."""
    return beta0 * (gamma ** (r - 1))


def run_round(
    r,
    cfg,
    run_subprocess=None,
    eval_fn=None,
    state_dir="orchestration/state",
):
    """Generate (no filter) -> train student (restart) -> eval. Returns acc_r."""
    import subprocess

    if run_subprocess is None:
        run_subprocess = subprocess.run

    state = Path(state_dir)
    state.mkdir(parents=True, exist_ok=True)
    teacher_ckpt = str(state / f"teacher_{r}.pth")

    run_subprocess(
        [
            "python",
            "generate_questions.py",
            "--config",
            "configs/generate_questions_aokvqa.yaml",
            "--overrides",
            f"pretrained={teacher_ckpt}",
            f"output_annotations_name=pool_{r}.json",
        ],
        check=True,
    )
    run_subprocess(
        [
            "python",
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=1",
            "train_vqa.py",
            "--output_dir",
            f"cache/rl_student_{r}",
            "--config",
            "configs/aokvqa.yaml",
            "--overrides",
            f"train_files=[train,pool_{r}]",
            "wandb=false",
        ],
        check=True,
    )
    acc = eval_fn(r) if eval_fn else 0.0
    (state / f"round_{r}.json").write_text(
        json.dumps({"round": r, "acc": acc, "teacher_ckpt": teacher_ckpt}, indent=2)
    )
    return acc
