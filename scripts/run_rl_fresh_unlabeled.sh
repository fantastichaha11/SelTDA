#!/usr/bin/env bash
# Fresh GRPO round 1 on COCO unlabeled2017 (no --resume).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs cache/judges orchestration/state
LOG=logs/train_vqg_rl_full.log
echo "=== Fresh unlabeled run $(date -Is) ===" >> "$LOG"
exec python train_vqg_rl.py \
  --config configs/rl_teacher_aokvqa.yaml \
  --output_dir orchestration/state \
  "$@" >> "$LOG" 2>&1
