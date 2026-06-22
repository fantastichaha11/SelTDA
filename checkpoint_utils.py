"""Checkpoint helpers for resuming SelTDA training scripts."""

from __future__ import annotations

from pathlib import Path

import torch


def resolve_resume_checkpoint(resume, output_dir):
    """Resolve --resume to a checkpoint path, or None when not resuming."""
    if resume is None:
        return None

    if resume == "auto":
        checkpoints = list_checkpoints(output_dir)
        if not checkpoints:
            raise FileNotFoundError(
                f"--resume auto: no checkpoint_*.pth found in {output_dir}"
            )
        return str(checkpoints[-1])

    checkpoint_path = Path(resume)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
    return str(checkpoint_path)


def resolve_training_resume(resume, output_dir, *, auto: bool = True) -> str | None:
    """Pick a checkpoint for training/eval.

  - Explicit ``--resume`` / ``--resume auto`` / path → unchanged behavior.
  - ``resume is None`` and ``auto=True`` → latest ``checkpoint_*.pth`` in output_dir.
  - ``auto=False`` (``--no-resume``) → always start from pretrained.
    """
    if resume is not None:
        return resolve_resume_checkpoint(resume, output_dir)

    if not auto:
        return None

    checkpoints = list_checkpoints(output_dir)
    if not checkpoints:
        return None

    latest = str(checkpoints[-1])
    print(f"Auto-resume: found checkpoint in {output_dir} → {latest}")
    return latest


def load_training_checkpoint(checkpoint_path, optimizer):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "optimizer" not in checkpoint:
        raise KeyError(
            f"{checkpoint_path} has no optimizer state; cannot resume training"
        )
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = int(checkpoint.get("epoch", -1)) + 1
    print(f"Resuming from {checkpoint_path} at epoch {start_epoch}")
    return start_epoch


def list_checkpoints(output_dir) -> list[Path]:
    return sorted(
        Path(output_dir).glob("checkpoint_*.pth"),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )


def prune_checkpoints(output_dir, max_checkpoints: int | None) -> list[Path]:
    """Delete oldest checkpoint_*.pth files, keeping at most max_checkpoints."""
    if max_checkpoints is None or max_checkpoints <= 0:
        return []

    checkpoints = list_checkpoints(output_dir)
    removed: list[Path] = []
    while len(checkpoints) > max_checkpoints:
        oldest = checkpoints.pop(0)
        oldest.unlink()
        removed.append(oldest)
        print(f"Removed old checkpoint: {oldest}")
    return removed
