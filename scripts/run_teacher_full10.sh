#!/usr/bin/env bash
# GRPO teacher: finish epochs 7-9 (no early stop), then keep best checkpoint only.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG="${ROOT}/logs/train_vqg_rl_vqascore.log"
STATE="${ROOT}/orchestration/state"
ROUND=1

log() { echo "[$(date -Is)] $*" | tee -a "${LOG}"; }

log "========== Teacher full-10 (no early stop) =========="

python train_vqg_rl.py \
  --config configs/rl_teacher_aokvqa.yaml \
  --output_dir "${STATE}" \
  --resume "${STATE}/teacher_${ROUND}_best.pth" \
  --overrides \
    grpo.epochs_per_round=10 \
    grpo.auto_tune=false \
    grpo.batch_size=1 \
    grpo.group_size=8 \
    audit.stop_on_judge_drop=false \
    audit.stop_on_reward_plateau=false \
    audit.rollback_to_best_judge=false \
  2>&1 | tee -a "${LOG}"

log "Training done — pruning to best checkpoint only"
rm -f "${STATE}"/teacher_${ROUND}_epoch*.pth
if [[ -f "${STATE}/teacher_${ROUND}_best.pth" ]]; then
  cp -f "${STATE}/teacher_${ROUND}_best.pth" "${STATE}/teacher_${ROUND}.pth"
  log "Kept teacher_${ROUND}_best.pth (+ teacher_${ROUND}.pth copy for scripts)"
else
  log "WARNING: teacher_${ROUND}_best.pth missing; left teacher_${ROUND}.pth as-is"
fi
log "========== Finished =========="
