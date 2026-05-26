#!/bin/bash
# SAT-1: sweep synthetic:real ratio × filter mode (orchestration only).
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

RATIOS=("1:1" "2:1" "4:1" "8:1")
MODES=(
  "unfiltered:datasets/aokvqa/synthetic_data_raw.json"
  "filtered_CIX:datasets/aokvqa/synthetic_data.json"
  "filtered_CIX_CSX:datasets/aokvqa/synthetic_data.json"
)

OUT_DIR="${PROJECT_ROOT}/research/experiments/saturation"
mkdir -p "${OUT_DIR}"
CSV="${OUT_DIR}/accuracy_vs_ratio.csv"
echo "ratio,filter_mode,synthetic_json,notes" > "${CSV}"

for ratio in "${RATIOS[@]}"; do
  for entry in "${MODES[@]}"; do
    mode="${entry%%:*}"
    json_path="${entry#*:}"
    echo "${ratio},${mode},${json_path},pending_train_eval" >> "${CSV}"
    echo "[SAT-1] ratio=${ratio} mode=${mode} → use ${json_path} with truncate_train_dataset_to override"
  done
done

echo "Wrote sweep manifest: ${CSV}"
echo "Run training via examples/run_experiment.sh with FILTER_CONFIG and ratio overrides."
