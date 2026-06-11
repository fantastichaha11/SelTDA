#!/bin/bash
# Minimal A-OKVQA + COCO train/val only (~25 GB) — fits 200 GB boot disk.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

# shellcheck source=conda_helpers.sh
source "${PROJECT_ROOT}/scripts/gcp/conda_helpers.sh"

ensure_gdown() {
  command -v gdown &>/dev/null || run_blip python -m pip install -q gdown
}

download_gdrive_file() {
  local file_id="$1" output_path="$2" label="$3"
  if [ -f "${output_path}" ]; then
    echo "[skip] ${label}: ${output_path}"
    return 0
  fi
  ensure_gdown
  mkdir -p "$(dirname "${output_path}")"
  gdown "https://drive.google.com/uc?id=${file_id}" -O "${output_path}"
}

if ! command -v aria2c &>/dev/null; then
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq aria2 unzip curl
fi

DATASETS_DIR="${PROJECT_ROOT}/datasets"
COCO_DIR="${DATASETS_DIR}/coco2017"
AOKVQA_DIR="${DATASETS_DIR}/aokvqa"
mkdir -p "${COCO_DIR}" "${AOKVQA_DIR}"

# A-OKVQA annotations
if [ ! -f "${AOKVQA_DIR}/aokvqa_v1p0_train.json" ]; then
  echo "Downloading A-OKVQA..."
  curl -fsSL https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz \
    | tar xvz -C "${AOKVQA_DIR}"
fi

download_coco_split() {
  local url="$1" zip="$2" marker="$3" min_jpgs="$4"
  cd "${COCO_DIR}"
  if [ -f "${marker}" ]; then
    echo "[skip] ${zip}"
    return 0
  fi
  if [ ! -f "${zip}" ]; then
    aria2c -x 16 -s 32 "${url}" -o "${zip}"
  fi
  unzip -q -j "${zip}"
  rm -f "${zip}"
  touch "${marker}"
  echo "COCO ${zip} done ($(find . -maxdepth 1 -name '*.jpg' | wc -l) jpgs)"
}

download_coco_split \
  "http://images.cocodataset.org/zips/train2017.zip" \
  "train2017.zip" ".train2017.complete" 100000

download_coco_split \
  "http://images.cocodataset.org/zips/val2017.zip" \
  "val2017.zip" ".val2017.complete" 40000

cd "${PROJECT_ROOT}"

for file in configs/aokvqa.yaml configs/aokvqg.yaml; do
  sed -i "s|^vqa_root:.*|vqa_root: '${COCO_DIR}'|" "$file"
  sed -i "s|^ann_root:.*|ann_root: '${AOKVQA_DIR}'|" "$file"
done

sed -i "s|IMAGES_ROOT = Path(\".*\")|IMAGES_ROOT = Path(\"${COCO_DIR}\")|" convert_aokvqa.py

download_gdrive_file "19Y9oQNlYBTkoT4sYuUQWrEV9iUatPkdI" \
  cache/teacher_weights/checkpoint_04.pth "teacher checkpoint"
download_gdrive_file "1mZbIX4lfKNgdPq6j41CWjMpPM-xl_ewj" \
  cache/student_weights/checkpoint_09.pth "student checkpoint"
download_gdrive_file "1WH8SG1FPtUqaNNDWtFrl7SZnC-4pfWlf" \
  "${AOKVQA_DIR}/synthetic_data_raw.json" "synthetic_data_raw.json"

run_blip python convert_aokvqa.py --config configs/aokvqa.yaml

echo "dataset_minimal.sh complete"
