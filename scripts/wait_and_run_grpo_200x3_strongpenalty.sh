#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_ROOT}"

LOG_PATH="${1:-outputs/logs/pathvqa_grpo_teacher_base7b_200x3epochs_strongpenalty_restart_20260710.log}"
mkdir -p "$(dirname "${LOG_PATH}")"

export PYTHONPATH="external/LLaVA:${PYTHONPATH:-}"
export SELECTED_PROMETHEUS_JUDGE="${SELECTED_PROMETHEUS_JUDGE:-cache/prometheus-vision-7b-v1.0}"
export PROMETHEUS_VISION_MODEL="${PROMETHEUS_VISION_MODEL:-cache/prometheus-vision-7b-v1.0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

while true; do
  if python3 - <<'PY' >>"${LOG_PATH}" 2>&1
import torch
if not torch.cuda.is_available():
    raise SystemExit(1)
torch.zeros(1, device="cuda")
print("[cuda-watch] cuda_available=True")
PY
  then
    break
  fi
  echo "[cuda-watch] CUDA unavailable; retrying in 60s" | tee -a "${LOG_PATH}"
  sleep 60
done

python3 scripts/train_teacher_grpo.py \
  --config configs/grpo_teacher_pathvqa_prometheus_200x3epochs.yaml \
  2>&1 | tee -a "${LOG_PATH}"
