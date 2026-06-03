#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${ROOT}/logs/train_vqg_rl_full.log"
EPOCH_LOG="${ROOT}/orchestration/state/round_1_epochs.jsonl"
AUDIT_LOG="${ROOT}/orchestration/state/round_1_audit.jsonl"

echo "=== RL teacher GRPO $(date -Is) ==="
if pgrep -f "python train_vqg_rl.py" >/dev/null; then
  echo "PROCESS: running (pid $(pgrep -f 'python train_vqg_rl.py' | head -1))"
else
  echo "PROCESS: not running"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | \
    awk -F', ' '{printf "GPU: util=%s mem=%s/%s temp=%sC\n", $1, $2, $3, $4}'
fi

if [[ -f "${LOG}" ]]; then
  echo "LOG: ${LOG}"
  grep -E "Auto-tuned|Round.*epoch|Saved|Traceback|Error" "${LOG}" 2>/dev/null | tail -5 | sed 's/^/  /' || true
  echo "  (last lines)"
  tail -3 "${LOG}" | sed 's/^/  /'
else
  echo "LOG: missing"
fi

if [[ -f "${EPOCH_LOG}" ]]; then
  echo "EPOCHS:"
  tail -3 "${EPOCH_LOG}" | sed 's/^/  /'
  STEPS=$(python3 -c "import yaml;print(yaml.safe_load(open('${ROOT}/configs/rl_teacher_aokvqa.yaml'))['grpo']['steps_per_epoch'])" 2>/dev/null || echo 80)
  DONE=$(grep -c "\"steps\": ${STEPS}" "${EPOCH_LOG}" 2>/dev/null || echo 0)
  echo "  completed_epochs=${DONE}/3 target_steps_per_epoch=${STEPS}"
fi

if [[ -f "${AUDIT_LOG}" ]]; then
  echo "AUDIT:"
  tail -2 "${AUDIT_LOG}" | sed 's/^/  /'
elif grep -q "Audit epoch" "${LOG}" 2>/dev/null; then
  echo "AUDIT (log):"
  grep "Audit epoch" "${LOG}" | tail -2 | sed 's/^/  /'
fi

if [[ -f "${ROOT}/orchestration/state/teacher_1.pth" ]]; then
  ls -lh "${ROOT}/orchestration/state/teacher_1.pth" | awk '{print "CHECKPOINT:", $5, $6, $7, $8, $9}'
fi
