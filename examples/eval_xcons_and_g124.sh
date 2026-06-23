#!/bin/bash
# A-OKVQA multiple-choice eval for both self-trained students (aokvqa_mc_eval.ipynb).
#
# Prerequisites:
#   conda activate blip
#   export PYTHONNOUSERSITE=1
#   datasets/aokvqa/aokvqa_v1p0_val.json  (from dataset.sh)
#   Training finished (bash examples/train_xcons_and_g124.sh)
#
# Optional env:
#   CHECKPOINT_XCONS=path/to/checkpoint_XX.pth
#   CHECKPOINT_G124=path/to/checkpoint_XX.pth
#   DEVICE=cuda
#   SKIP_XCONS_EVAL=1
#   SKIP_G124_EVAL=1
#
# Results:
#   cache/evals_xcons/mc_eval_result.json
#   cache/evals_g124/mc_eval_result.json

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

DEVICE="${DEVICE:-cuda}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[ -f "${AOKVQA_DIR}/aokvqa_v1p0_val.json" ] || die "Missing ${AOKVQA_DIR}/aokvqa_v1p0_val.json (run dataset.sh)"

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

COMMON_OVERRIDES=(
  "vqa_root='${COCO_DIR}'"
  "ann_root='${AOKVQA_DIR}'"
)

run_eval() {
  local label="$1"
  local ckpt="$2"
  local eval_dir="$3"

  mkdir -p "${eval_dir}"
  echo "========== MC eval: ${label} =========="
  echo "  checkpoint: ${ckpt}"
  echo "  output:     ${eval_dir}/mc_eval_result.json"

  python aokvqa_mc_eval.py \
    --checkpoint "${ckpt}" \
    --config configs/aokvqa.yaml \
    --device "${DEVICE}" \
    --output "${eval_dir}/mc_eval_result.json" \
    --overrides "${COMMON_OVERRIDES[@]}"
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

echo "========== MC evaluation finished =========="
[ -f "${EVAL_XCONS}/mc_eval_result.json" ] && \
  echo "  xcons: ${EVAL_XCONS}/mc_eval_result.json"
[ -f "${EVAL_G124}/mc_eval_result.json" ] && \
  echo "  g124:  ${EVAL_G124}/mc_eval_result.json"
