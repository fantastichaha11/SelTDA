#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="${ROOT}/logs/rl_round1_generate_17k.log"
echo "=== RL generate $(date -Is) ==="
pgrep -af generate_questions_rl_round1 | head -1 || echo "not running"
grep -oE "[0-9]+/[0-9]+" "$LOG" 2>/dev/null | tail -1
grep "Sucessfully parsed" "$LOG" 2>/dev/null | tail -1
tail -1 "$LOG" 2>/dev/null
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
