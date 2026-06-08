#!/usr/bin/env bash
# Wait for GRPO teacher training to finish, then run generate -> train -> eval student.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

LOG="${ROOT}/logs/wait_teacher_then_student.log"
TEACHER_LOG="${ROOT}/logs/train_vqg_rl_vqascore.log"
STUDENT_LOG="${ROOT}/logs/rl_round1_pipeline_vqascore.log"

log() { echo "[$(date -Is)] $*" | tee -a "${LOG}"; }

log "Waiting for train_vqg_rl.py to finish..."
while pgrep -f "python.*train_vqg_rl.py" >/dev/null 2>&1; do
  if [[ -f "${TEACHER_LOG}" ]]; then
    tail -1 "${TEACHER_LOG}" 2>/dev/null | tee -a "${LOG}" || true
  fi
  sleep 60
done

log "Teacher training stopped."
if [[ ! -f orchestration/state/teacher_1.pth ]]; then
  log "ERROR: missing orchestration/state/teacher_1.pth"
  exit 1
fi

# Free disk: drop intermediate teacher epoch snapshots (keep best + final).
rm -f orchestration/state/teacher_1_epoch*.pth 2>/dev/null || true
log "Disk after cleanup: $(df -h / | tail -1)"

log "Starting student pipeline (generate -> train -> eval)..."
: >> "${STUDENT_LOG}"
export PYTHONNOUSERSITE=1
unset PYTORCH_CUDA_ALLOC_CONF SKIP_GENERATE SKIP_TRAIN SKIP_EVAL SKIP_FILTER
rm -f "${ROOT}/datasets/aokvqa/synthetic_data_rl_raw.json" \
  "${ROOT}/datasets/aokvqa/synthetic_data_rl.json" 2>/dev/null || true
env SKIP_FILTER=1 BATCH_SIZE_TRAIN=12 bash scripts/run_rl_round1_student.sh \
  >> "${STUDENT_LOG}" 2>&1

log "Student pipeline finished."
