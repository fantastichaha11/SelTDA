#!/usr/bin/env bash
# Run hourly RL reports to logs/monitor_rl_hourly.log
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/logs/monitor_rl_hourly.log"
REPORT="${ROOT}/scripts/report_rl_hourly.sh"

chmod +x "${REPORT}"

while true; do
  {
    echo ""
    echo "========================================"
    bash "${REPORT}"
    echo ""
  } >> "${OUT}" 2>&1
  sleep 3600
done
