#!/usr/bin/env bash
# GRPO teacher: VQAScore-only reward = P(yes)-P(no) + teacher gen_logprob (gate 1 conf).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG="${ROOT}/logs/train_vqg_rl_vqascore_conf.log"
STATE="${ROOT}/orchestration/state"
ROUND=2

log() { echo "[$(date -Is)] $*" | tee -a "${LOG}"; }

log "========== Teacher VQAScore-conf (round ${ROUND}) =========="

python train_vqg_rl.py \
  --config configs/rl_teacher_vqascore_conf.yaml \
  --output_dir "${STATE}" \
  --overrides \
    grpo.epochs_per_round=10 \
    grpo.auto_tune=false \
    grpo.batch_size=1 \
    grpo.group_size=8 \
    audit.stop_on_judge_drop=false \
    audit.stop_on_reward_plateau=false \
  2>&1 | tee -a "${LOG}"

log "Training done — pruning to best checkpoint only"
rm -f "${STATE}"/teacher_${ROUND}_epoch*.pth
if [[ -f "${STATE}/teacher_${ROUND}_best.pth" ]]; then
  cp -f "${STATE}/teacher_${ROUND}_best.pth" "${STATE}/teacher_${ROUND}.pth"
  log "Kept teacher_${ROUND}_best.pth (+ teacher_${ROUND}.pth copy)"
else
  log "WARNING: teacher_${ROUND}_best.pth missing"
fi
log "========== Finished =========="
