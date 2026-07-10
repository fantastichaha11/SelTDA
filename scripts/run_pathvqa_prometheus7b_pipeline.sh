#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_ROOT}"

export PROMETHEUS_VISION_MODEL="${PROMETHEUS_VISION_MODEL:-cache/prometheus-vision-7b-v1.0}"
export SELECTED_PROMETHEUS_JUDGE="${SELECTED_PROMETHEUS_JUDGE:-outputs/prometheus_judge/pathvqa_llava_lora_adapted}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONPATH="${PYTHONPATH:-external/LLaVA}"

python3 scripts/train_prometheus_judge.py \
  --config configs/prometheus_judge_pathvqa.yaml \
  --backend external_llava

python3 scripts/train_teacher_grpo.py \
  --config configs/grpo_teacher_pathvqa_prometheus.yaml
