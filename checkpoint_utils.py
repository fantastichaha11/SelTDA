"""Checkpoint helpers for resuming SelTDA training scripts."""

from __future__ import annotations

from pathlib import Path

import torch


def resolve_resume_checkpoint(resume, output_dir):
    """Resolve --resume to a checkpoint path, or None when not resuming."""
    if resume is None:
        return None

    if resume == "auto":
        checkpoints = sorted(
            Path(output_dir).glob("checkpoint_*.pth"),
            key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
        )
        if not checkpoints:
            raise FileNotFoundError(
                f"--resume auto: no checkpoint_*.pth found in {output_dir}"
            )
        return str(checkpoints[-1])

    checkpoint_path = Path(resume)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
    return str(checkpoint_path)


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
