#!/bin/bash
# Start (or restart) VQAScore pipeline in tmux for the current SSH user.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"

export PATH="/opt/conda/bin:${PATH}"
# shellcheck source=/dev/null
source /opt/conda/etc/profile.d/conda.sh

tmux kill-session -t vqascore 2>/dev/null || true
tmux new-session -d -s vqascore \
  "export PATH=/opt/conda/bin:\$PATH && source /opt/conda/etc/profile.d/conda.sh && cd ${REPO} && bash scripts/gcp/run_vqascore_pipeline.sh; echo EXIT=\$? | tee -a cache/logs/vqascore_run/exit_code.txt; exec bash"

echo "Pipeline started in tmux session 'vqascore' (user: $(whoami))"
echo "Log: tail -f ${REPO}/cache/logs/vqascore_run/pipeline.log"
echo "Attach: tmux attach -t vqascore"
