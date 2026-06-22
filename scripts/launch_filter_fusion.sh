#!/usr/bin/env bash
# Launch the fusion filter fully detached so it survives the agent session.
set -euo pipefail

cd /workspace/SelTDA
export PYTHONNOUSERSITE=1

ROOT="$(pwd)"
mkdir -p cache/logs cache/filter_fusion_run

# Kill any stragglers first.
pkill -9 -f filter_pseudo.py 2>/dev/null || true
sleep 2

setsid nohup python filter_pseudo.py \
  --config configs/filter_pseudo_fusion.yaml \
  --output_dir cache/filter_fusion_run \
  --overrides \
    device=cuda \
    torch_home="${ROOT}/cache/torch_home" \
    input="${ROOT}/datasets/aokvqa/synthetic_data_raw.json" \
    image_root="${ROOT}/datasets/coco2017" \
    output="${ROOT}/datasets/aokvqa/synthetic_data_fusion.json" \
    report="${ROOT}/datasets/aokvqa/filter_report_fusion.json" \
  > cache/logs/filter_fusion.log 2>&1 < /dev/null &

PID=$!
echo "${PID}" > cache/logs/filter_fusion.pid
echo "LAUNCHED_PID:${PID}"
