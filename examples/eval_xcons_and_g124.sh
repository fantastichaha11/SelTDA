#!/bin/bash
# Evaluate A-OKVQA validation set for both self-trained students:
#   1) xcons  — cache/self_trained_weights_xcons
#   2) g124   — cache/self_trained_weights_g124
#
# Prerequisites:
#   conda activate blip
#   export PYTHONNOUSERSITE=1
#   Training finished (bash examples/train_xcons_and_g124.sh)
#
# Optional env:
#   NUM_GPUS=1
#   CHECKPOINT_XCONS=path/to/checkpoint_XX.pth   (default: latest in train dir)
#   CHECKPOINT_G124=path/to/checkpoint_XX.pth
#   SKIP_XCONS_EVAL=1
#   SKIP_G124_EVAL=1
#
# Results:
#   cache/evals_xcons/result/vqa_result.json
#   cache/evals_g124/result/vqa_result.json

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONNOUSERSITE=1

DATASETS_DIR="${PROJECT_ROOT}/datasets"
COCO_DIR="${DATASETS_DIR}/coco2017"
AOKVQA_DIR="${DATASETS_DIR}/aokvqa"

TRAIN_XCONS="${PROJECT_ROOT}/cache/self_trained_weights_xcons"
TRAIN_G124="${PROJECT_ROOT}/cache/self_trained_weights_g124"
EVAL_XCONS="${PROJECT_ROOT}/cache/evals_xcons"
EVAL_G124="${PROJECT_ROOT}/cache/evals_g124"

NUM_GPUS="${NUM_GPUS:-1}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

latest_checkpoint() {
  local train_dir="$1"
  ls -1 "${train_dir}"/checkpoint_*.pth 2>/dev/null | sort -V | tail -1
}

resolve_checkpoint() {
  local explicit="${1:-}"
  local train_dir="$2"
  local label="$3"

  if [ -n "${explicit}" ]; then
    [ -f "${explicit}" ] || die "${label} checkpoint not found: ${explicit}"
    echo "${explicit}"
    return
  fi

  local ckpt
  ckpt="$(latest_checkpoint "${train_dir}")"
  [ -n "${ckpt}" ] || die "No checkpoint_*.pth in ${train_dir} (run train first)"
  echo "${ckpt}"
}

run_eval() {
  local label="$1"
  local ckpt="$2"
  local eval_dir="$3"

  mkdir -p "${eval_dir}"
  echo "========== Eval: ${label} =========="
  echo "  checkpoint: ${ckpt}"
  echo "  output:     ${eval_dir}"

  python -m torch.distributed.run --nproc_per_node="${NUM_GPUS}" train_vqa.py \
    --output_dir="${eval_dir}" \
    --evaluate \
    --no-resume \
    --config configs/aokvqa.yaml \
    --overrides \
      "vqa_root='${COCO_DIR}'" \
      "ann_root='${AOKVQA_DIR}'" \
      "pretrained='${ckpt}'" \
      use_validation_set_as_test_set=true \
      wandb=false

  echo "  result: ${eval_dir}/result/vqa_result.json"
}

if [ "${SKIP_XCONS_EVAL:-0}" != "1" ]; then
  CKPT_XCONS="$(resolve_checkpoint "${CHECKPOINT_XCONS:-}" "${TRAIN_XCONS}" "xcons")"
  run_eval "xcons (gate 4)" "${CKPT_XCONS}" "${EVAL_XCONS}"
else
  echo "SKIP_XCONS_EVAL=1 — skipping xcons eval."
fi

if [ "${SKIP_G124_EVAL:-0}" != "1" ]; then
  CKPT_G124="$(resolve_checkpoint "${CHECKPOINT_G124:-}" "${TRAIN_G124}" "g124")"
  run_eval "gates 1+2+4" "${CKPT_G124}" "${EVAL_G124}"
else
  echo "SKIP_G124_EVAL=1 — skipping g124 eval."
fi

echo "========== Evaluation finished =========="
[ -f "${EVAL_XCONS}/result/vqa_result.json" ] && \
  echo "  xcons: ${EVAL_XCONS}/result/vqa_result.json"
[ -f "${EVAL_G124}/result/vqa_result.json" ] && \
  echo "  g124:  ${EVAL_G124}/result/vqa_result.json"
