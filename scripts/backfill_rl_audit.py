#!/usr/bin/env python3
"""Backfill epoch audits for RL teacher (e.g. epoch 0 after a crashed post-epoch audit)."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_vqg_rl import (
    _backfill_missing_audits,
    _cfg_namespace,
    _read_epoch_rows,
    build_reward_fn,
    build_teacher,
)

logger = logging.getLogger(__name__)


def _training_running() -> bool:
    try:
        subprocess.check_output(
            ["pgrep", "-f", "python train_vqg_rl.py"],
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/rl_teacher_aokvqa.yaml",
        help="RL teacher config path",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=None,
        help="Epoch indices to audit (default: audit.backfill_epochs from config)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cuda or cpu (default: cpu if train_vqg_rl.py is running, else cuda)",
    )
    args_cli = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    config = OmegaConf.load(ROOT / args_cli.config)

    import torch

    if args_cli.epochs is not None:
        config.audit.backfill_epochs = list(args_cli.epochs)

    device = args_cli.device
    if device is None:
        device = "cpu" if _training_running() else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Audit device: %s (training_running=%s)", device, _training_running())

    round_id = int(config.grpo.get("round", 1))
    out_dir = __import__("pathlib").Path(config.grpo.get("output_dir", "orchestration/state"))
    log_path = out_dir / f"round_{round_id}_epochs.jsonl"

    cfg = _cfg_namespace(config)
    completed = _read_epoch_rows(log_path, round_id, min_steps=cfg.steps_per_epoch)
    if not completed:
        raise SystemExit(f"No completed epochs in {log_path}")

    last = completed[-1]
    cfg.group_size = int(last.get("group_size", cfg.group_size))
    cfg.batch_size = int(last.get("batch_size", cfg.batch_size))
    cfg.kl_beta = float(last.get("kl_beta", cfg.kl_beta))

    # Load from first target epoch checkpoint (not latest teacher_*.pth).
    targets = [int(e) for e in config.audit.get("backfill_epochs", [])]
    if not targets:
        raise SystemExit("No epochs to audit (set --epochs or audit.backfill_epochs)")

    from train_vqg_rl import _resolve_backfill_ckpt

    ep0 = min(targets)
    ckpt_path = _resolve_backfill_ckpt(ep0, round_id, out_dir, config)
    if ckpt_path is None:
        raise SystemExit(f"No checkpoint for epoch {ep0}")

    policy = build_teacher(config, device=device, trainable=False)
    ckpt = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(ckpt["model"])
    logger.info("Loaded policy for audit from %s (epoch %s)", ckpt_path, ckpt.get("epoch"))

    probe_peak = int(config.grpo.get("resume_probe_peak_mib", 20515))
    reward_fn, reward_cfg, audit_terms_fn = build_reward_fn(
        config, device=device, probe_peak_mib=probe_peak
    )
    del reward_fn

    from orchestration.hrp import HrpState

    hrp_state = HrpState(kl_beta=cfg.kl_beta, w_itm=float(reward_cfg.w_itm))

    _backfill_missing_audits(
        policy,
        config,
        round_id=round_id,
        out_dir=out_dir,
        epoch_rows=completed,
        reward_terms_fn=audit_terms_fn,
        device=device,
        cfg=cfg,
        hrp_state=hrp_state,
        reward_cfg=reward_cfg,
    )
    logger.info("Backfill done. See %s", out_dir / f"round_{round_id}_audit.jsonl")


if __name__ == "__main__":
    main()
