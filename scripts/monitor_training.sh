#!/usr/bin/env bash
# Emit training status snapshot for periodic monitoring.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT}/logs"
LATEST_LOG="$(ls -t "${LOG_DIR}"/train_vqa_*.log 2>/dev/null | head -1 || true)"

echo "=== SelTDA training status $(date -Is) ==="
if pgrep -f "python train_vqa.py" >/dev/null; then
  echo "PROCESS: running (pid $(pgrep -f 'python train_vqa.py' | head -1))"
else
  echo "PROCESS: not running"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | \
    awk -F', ' '{printf "GPU: util=%s mem=%s/%s temp=%sC\n", $1, $2, $3, $4}'
fi

if [[ -n "${LATEST_LOG}" ]]; then
  echo "LOG: ${LATEST_LOG}"
  tail -5 "${LATEST_LOG}" | sed 's/^/  /'
  if grep -q "Train Epoch:" "${LATEST_LOG}"; then
    grep "Train Epoch:" "${LATEST_LOG}" | tail -1 | sed 's/^/LAST_EPOCH: /'
  elif grep -q "%|" "${LATEST_LOG}"; then
    echo "LAST_LINE: $(tail -1 "${LATEST_LOG}" | tr -d '\r' | cut -c1-120)"
  fi
else
  echo "LOG: none found"
fi

CKPT_DIR="${ROOT}/cache/self_trained_weights"
if [[ -d "${CKPT_DIR}" ]]; then
  ls -lt "${CKPT_DIR}"/*.pth 2>/dev/null | head -3 | awk '{print "CHECKPOINT:", $9, $6, $7, $8}' || true
fi
