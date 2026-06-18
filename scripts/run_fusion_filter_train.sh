#!/bin/bash
# Wait for filter fusion to finish, then self-train student on train + synthetic_data_fusion.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONNOUSERSITE=1

LOG_DIR="${PROJECT_ROOT}/cache/logs"
SYNTH_OUT="${PROJECT_ROOT}/datasets/aokvqa/synthetic_data_fusion.json"
REPORT="${PROJECT_ROOT}/datasets/aokvqa/filter_report_fusion.json"
OUT_TRAIN="${PROJECT_ROOT}/cache/self_trained_weights_fusion"
FILTER_PID="${1:-}"

mkdir -p "${LOG_DIR}" "${OUT_TRAIN}"

log() {
  echo "[$(date -Is)] $*"
}

wait_for_filter() {
  if [[ -n "${FILTER_PID}" ]]; then
    log "Waiting for filter PID ${FILTER_PID}..."
    while kill -0 "${FILTER_PID}" 2>/dev/null; do
      sleep 60
    done
    log "Filter process exited."
  else
    log "Waiting for ${SYNTH_OUT}..."
    while [[ ! -f "${SYNTH_OUT}" ]]; do
      sleep 60
    done
  fi
}

wait_for_filter

if [[ ! -f "${SYNTH_OUT}" || ! -f "${REPORT}" ]]; then
  log "ERROR: filter outputs missing — expected:"
  log "  ${SYNTH_OUT}"
  log "  ${REPORT}"
  exit 1
fi

log "Filter outputs ready. Starting student training..."
python -m torch.distributed.run --nproc_per_node=1 train_vqa.py \
  --output_dir="${OUT_TRAIN}" \
  --config configs/aokvqa.yaml \
  --overrides \
    "train_files=[train,synthetic_data_fusion]" \
    truncate_train_dataset_to=34000 \
    wandb=false \
    use_validation_set_as_test_set=true \
    "torch_home=${PROJECT_ROOT}/cache/torch_home" \
  2>&1 | tee "${LOG_DIR}/train_fusion.log"

log "Student training finished. Checkpoints: ${OUT_TRAIN}"
