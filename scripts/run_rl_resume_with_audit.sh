#!/usr/bin/env bash
# Resume GRPO from orchestration/state/teacher_1.pth (epochs 0-1 done) with epoch audit.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs cache/judges
LOG=logs/train_vqg_rl_full.log
exec python train_vqg_rl.py \
  --config configs/rl_teacher_aokvqa.yaml \
  --output_dir orchestration/state \
  --resume auto \
  "$@" >> "$LOG" 2>&1
