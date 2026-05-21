#!/bin/bash
# Filter raw synthetic data between generate and train.
# Usage: bash examples/filter_synthetic.sh <dataset_name>
set -euo pipefail
DS="${1:-aokvqa}"

python filter_pseudo.py \
    --config configs/filter_pseudo.yaml \
    --overrides \
        input=datasets/${DS}/synthetic_data_raw.json \
        output=datasets/${DS}/synthetic_data.json \
        report=datasets/${DS}/filter_report.json
