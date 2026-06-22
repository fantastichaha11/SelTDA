#!/bin/bash
# Self-train student on two synthetic pools (2 checkpoints):
#   1) xcons only — synthetic_data_xcons_full.json (from dataset.sh, no filter)
#   2) gates 1+2+4 — synthetic_data_g124.json (from filter_gates_1_2_4.sh)
#
# Prerequisites:
#   conda activate blip
#   export PYTHONNOUSERSITE=1
#   datasets/aokvqa/synthetic_data_xcons_full.json  (bash dataset.sh)
#   datasets/aokvqa/synthetic_data_g124.json        (bash examples/filter_gates_1_2_4.sh)
#   datasets/aokvqa/train.json, val.json
#
# Optional env:
#   NUM_GPUS=1
#   TRUNCATE=34000
#   SKIP_XCONS_TRAIN=1
#   SKIP_G124_TRAIN=1
#   FRESH=1                  start from pretrained (pass --no-resume)
#   RESUME_XCONS=path        optional explicit checkpoint (overrides auto)
#   RESUME_G124=path
#
# Auto-resume: if output_dir already has checkpoint_*.pth, training continues
# from the latest file without passing --resume.
#
# Checkpoints:
#   cache/self_trained_weights_xcons/
#   cache/self_trained_weights_g124/

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONNOUSERSITE=1

DATASETS_DIR="${PROJECT_ROOT}/datasets"
COCO_DIR="${DATASETS_DIR}/coco2017"
AOKVQA_DIR="${DATASETS_DIR}/aokvqa"

OUT_XCONS="${PROJECT_ROOT}/cache/self_trained_weights_xcons"
OUT_G124="${PROJECT_ROOT}/cache/self_trained_weights_g124"

NUM_GPUS="${NUM_GPUS:-1}"
TRUNCATE="${TRUNCATE:-34000}"

[ -f "${AOKVQA_DIR}/synthetic_data_xcons_full.json" ] || {
  echo "ERROR: Missing ${AOKVQA_DIR}/synthetic_data_xcons_full.json" >&2
  echo "  Run: bash dataset.sh" >&2
  exit 1
}

[ -f "${AOKVQA_DIR}/synthetic_data_g124.json" ] || {
  echo "ERROR: Missing ${AOKVQA_DIR}/synthetic_data_g124.json" >&2
  echo "  Run: bash examples/filter_gates_1_2_4.sh" >&2
  exit 1
}

COMMON_TRAIN_OVERRIDES=(
  "vqa_root='${COCO_DIR}'"
  "ann_root='${AOKVQA_DIR}'"
  "truncate_train_dataset_to=${TRUNCATE}"
  wandb=false
  use_validation_set_as_test_set=true
)

run_train() {
  local label="$1"
  local out_dir="$2"
  local synth_name="$3"
  local resume="${4:-}"

  mkdir -p "${out_dir}"
  echo "========== Train: ${label} (train + ${synth_name}) → ${out_dir} =========="

  local cmd=(
    python -m torch.distributed.run --nproc_per_node="${NUM_GPUS}" train_vqa.py
    --output_dir="${out_dir}"
    --config configs/aokvqa.yaml
    --overrides
      "${COMMON_TRAIN_OVERRIDES[@]}"
      "train_files=[train,${synth_name}]"
  )
  if [ "${FRESH:-0}" = "1" ]; then
    cmd+=(--no-resume)
  elif [ -n "${resume}" ]; then
    cmd+=(--resume "${resume}")
  fi
  "${cmd[@]}"
}

if [ "${SKIP_XCONS_TRAIN:-0}" != "1" ]; then
  run_train "xcons (gate 4)" "${OUT_XCONS}" "synthetic_data_xcons_full" "${RESUME_XCONS:-}"
else
  echo "SKIP_XCONS_TRAIN=1 — skipping xcons checkpoint."
fi

if [ "${SKIP_G124_TRAIN:-0}" != "1" ]; then
  run_train "gates 1+2+4" "${OUT_G124}" "synthetic_data_g124" "${RESUME_G124:-}"
else
  echo "SKIP_G124_TRAIN=1 — skipping g124 checkpoint."
fi

echo "========== Training finished =========="
echo "  xcons checkpoint: ${OUT_XCONS}"
echo "  g124 checkpoint:  ${OUT_G124}"
