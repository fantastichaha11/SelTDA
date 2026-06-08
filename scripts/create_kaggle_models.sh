#!/usr/bin/env bash
# Register SelTDA teacher + student on Kaggle Models.
# Requires Legacy API key in ~/.kaggle/kaggle.json (models.create scope).
set -eu
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ]]; then
  if [[ ! -f "${HOME}/.kaggle/kaggle.json" ]]; then
    echo "Set KAGGLE_USERNAME + KAGGLE_KEY or create ~/.kaggle/kaggle.json" >&2
    echo "Use Legacy API Key: https://www.kaggle.com/settings" >&2
    exit 1
  fi
fi

echo "Creating teacher model..."
kaggle models create -p "${ROOT}/kaggle_models/teacher"

echo "Creating student model..."
kaggle models create -p "${ROOT}/kaggle_models/student"

echo "Done. Upload all versions: bash scripts/upload_kaggle_model_versions.sh"
