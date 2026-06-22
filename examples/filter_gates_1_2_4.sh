#!/bin/bash
# Filter pseudo-QA with gates 1+2+4 (conf → itm → xcons; vqascore / gate 3 off).
#
# xcons-only pool (gate 4) is NOT produced here — use the published file from
# dataset.sh: datasets/aokvqa/synthetic_data_xcons_full.json
#
# Gate map (filter_pseudo GATE_ORDER):
#   1 = conf   2 = itm   3 = vqascore (off)   4 = xcons
#
# Prerequisites:
#   conda activate blip
#   export PYTHONNOUSERSITE=1
#   datasets/aokvqa/synthetic_data_raw.json
#   datasets/coco2017 images
#   cache/student_weights/checkpoint_09.pth
#   datasets/aokvqa/score_cache/synthetic_data_raw__xcons.json (optional, from dataset.sh)
#
# Optional env:
#   KEEP_TOP=0.75    quantile per enabled gate
#   DEVICE=cuda
#
# Outputs:
#   datasets/aokvqa/synthetic_data_g124.json
#   datasets/aokvqa/filter_report_g124.json

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONNOUSERSITE=1

DATASETS_DIR="${PROJECT_ROOT}/datasets"
COCO_DIR="${DATASETS_DIR}/coco2017"
AOKVQA_DIR="${DATASETS_DIR}/aokvqa"
RAW="${AOKVQA_DIR}/synthetic_data_raw.json"
STUDENT_CKPT="${PROJECT_ROOT}/cache/student_weights/checkpoint_09.pth"

KEEP_TOP="${KEEP_TOP:-0.75}"
DEVICE="${DEVICE:-cuda}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[ -f "${RAW}" ] || die "Missing raw synthetic pool: ${RAW}"
[ -d "${COCO_DIR}" ] || die "Missing COCO root: ${COCO_DIR}"
[ -f "${STUDENT_CKPT}" ] || die "Missing student checkpoint for xcons gate: ${STUDENT_CKPT}"

COMMON_OVERRIDES=(
  "input='${RAW}'"
  "image_root='${COCO_DIR}'"
  "score_cache.dir='${AOKVQA_DIR}/score_cache'"
  "gates.xcons.student_ckpt='${STUDENT_CKPT}'"
  "device=${DEVICE}"
)

echo "========== Filter gates 1+2+4 (conf → itm → xcons), keep_top=${KEEP_TOP} =========="
python filter_pseudo.py \
  --config configs/filter_pseudo_gates_124.yaml \
  --overrides \
    "${COMMON_OVERRIDES[@]}" \
    "output='${AOKVQA_DIR}/synthetic_data_g124.json'" \
    "report='${AOKVQA_DIR}/filter_report_g124.json'" \
    "gates.conf.keep_top=${KEEP_TOP}" \
    "gates.itm.keep_top=${KEEP_TOP}" \
    "gates.xcons.keep_top=${KEEP_TOP}"

[ -f "${AOKVQA_DIR}/synthetic_data_g124.json" ] \
  || die "Filter did not produce ${AOKVQA_DIR}/synthetic_data_g124.json"

echo "========== Filter finished =========="
echo "  gates 1+2+4: ${AOKVQA_DIR}/synthetic_data_g124.json"
echo "  xcons pool:  ${AOKVQA_DIR}/synthetic_data_xcons_full.json (from dataset.sh, no filter needed)"
