#!/bin/bash

set -euo pipefail

# =========================
# Save project root
# =========================

PROJECT_ROOT=$(pwd)

# =========================
# Helpers: skip-if-present downloads
# =========================

ensure_gdown() {
    if ! command -v gdown &> /dev/null; then
        echo "Installing gdown..."
        pip install -q gdown
    fi
}

download_gdrive_file() {
    local file_id="$1"
    local output_path="$2"
    local label="$3"

    if [ -f "${output_path}" ]; then
        echo "[skip] ${label} already exists: ${output_path}"
        return 0
    fi

    echo "Downloading ${label} to ${output_path}..."
    ensure_gdown
    mkdir -p "$(dirname "${output_path}")"
    gdown "https://drive.google.com/uc?id=${file_id}" -O "${output_path}"
}

count_jpg_in_dir() {
    local dir="$1"
    find "${dir}" -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l | tr -d ' '
}

# Return 0 when a COCO split is already on disk (marker, split folder, or flat layout).
coco_split_already_present() {
    local marker="$1"
    local split_dir="$2"
    local flat_min_jpgs="${3:-0}"

    if [ -f "${marker}" ]; then
        echo "[skip] ${split_dir} already extracted (${marker})"
        return 0
    fi

    if [ -d "${split_dir}" ]; then
        local folder_count
        folder_count=$(count_jpg_in_dir "${split_dir}")
        if [ "${folder_count}" -ge 1 ]; then
            echo "[skip] ${split_dir}/ already present (${folder_count} images) — marking ${marker}"
            touch "${marker}"
            return 0
        fi
    fi

    if [ "${flat_min_jpgs}" -gt 0 ]; then
        local flat_count
        flat_count=$(count_jpg_in_dir ".")
        if [ "${flat_count}" -ge "${flat_min_jpgs}" ]; then
            echo "[skip] ${split_dir} images already extracted flat in $(pwd) (${flat_count} images) — marking ${marker}"
            touch "${marker}"
            return 0
        fi
    fi

    return 1
}

download_coco_zip() {
    local url="$1"
    local zip_name="$2"
    local marker="$3"
    local unzip_mode="${4:-flat}"
    local split_dir="${5:-}"
    local flat_min_jpgs="${6:-0}"

    if coco_split_already_present "${marker}" "${split_dir}" "${flat_min_jpgs}"; then
        return 0
    fi

    if [ ! -f "${zip_name}" ]; then
        echo "Downloading ${zip_name}..."
        aria2c -x 16 -s 32 "${url}" -o "${zip_name}"
    else
        echo "[skip] ${zip_name} download — archive already on disk, extracting..."
    fi

    echo "Extracting ${zip_name}..."
    if [ "${unzip_mode}" = "flat" ]; then
        unzip -q -j "${zip_name}"
    else
        unzip -q "${zip_name}"
    fi
    rm -f "${zip_name}"
    touch "${marker}"
}

# =========================
# Install aria2
# =========================

if ! command -v aria2c &> /dev/null
then
    echo "Installing aria2..."

    if command -v apt &> /dev/null
    then
        sudo apt update
        sudo apt install -y aria2 unzip

    elif command -v yum &> /dev/null
    then
        sudo yum install -y aria2 unzip

    elif command -v pacman &> /dev/null
    then
        sudo pacman -Sy --noconfirm aria2 unzip

    else
        echo "Unsupported package manager"
        exit 1
    fi
fi

# =========================
# Create dataset dirs
# =========================

export DATASETS_DIR="${PROJECT_ROOT}/datasets"
export COCO_DIR="${DATASETS_DIR}/coco2017"
export AOKVQA_DIR="${DATASETS_DIR}/aokvqa"

mkdir -p "${COCO_DIR}"
mkdir -p "${AOKVQA_DIR}"

# =========================
# Download A-OKVQA
# =========================

AOKVQA_MARKER="${AOKVQA_DIR}/aokvqa_v1p0_train.json"

if [ -f "${AOKVQA_MARKER}" ]; then
    echo "[skip] A-OKVQA already present: ${AOKVQA_MARKER}"
else
    echo "Downloading A-OKVQA..."

    curl -fsSL \
    https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz \
    | tar xvz -C "${AOKVQA_DIR}"
fi

# =========================
# Download COCO2017
# =========================

cd "${COCO_DIR}"

download_coco_zip \
    "http://images.cocodataset.org/zips/train2017.zip" \
    "train2017.zip" \
    ".train2017.complete" \
    "flat" \
    "train2017" \
    100000

download_coco_zip \
    "http://images.cocodataset.org/zips/val2017.zip" \
    "val2017.zip" \
    ".val2017.complete" \
    "flat" \
    "val2017" \
    122000

download_coco_zip \
    "http://images.cocodataset.org/zips/test2017.zip" \
    "test2017.zip" \
    ".test2017.complete" \
    "flat" \
    "test2017" \
    160000

UNLABELED_MARKER=".unlabeled2017.complete"
if coco_split_already_present "${UNLABELED_MARKER}" "unlabeled2017" 0; then
    :
else
    download_coco_zip \
        "http://images.cocodataset.org/zips/unlabeled2017.zip" \
        "unlabeled2017.zip" \
        "${UNLABELED_MARKER}" \
        "folder" \
        "unlabeled2017" \
        0
fi

echo "COCO extraction completed!"

# =========================
# Update configs
# =========================

COCO_PATH="${COCO_DIR}"
AOKVQA_PATH="${AOKVQA_DIR}"

for file in \
    "${PROJECT_ROOT}/configs/aokvqa.yaml" \
    "${PROJECT_ROOT}/configs/aokvqg.yaml"
do
    sed -i "s|^vqa_root:.*|vqa_root: '${COCO_PATH}'|" "$file"
    sed -i "s|^ann_root:.*|ann_root: '${AOKVQA_PATH}'|" "$file"

    echo "Updated $file"
done

# =========================
# Update convert_aokvqa.py
# =========================

PY_FILE="${PROJECT_ROOT}/convert_aokvqa.py"

sed -i \
"s|IMAGES_ROOT = Path(\".*\")|IMAGES_ROOT = Path(\"${COCO_DIR}\")|" \
"${PY_FILE}"

echo "Updated ${PY_FILE}"

# =========================
# Download teacher checkpoint (VQG, A-OKVQA)
# Google Drive: checkpoint_04.pth
# https://drive.google.com/file/d/19Y9oQNlYBTkoT4sYuUQWrEV9iUatPkdI/view
# =========================

TEACHER_DIR="${PROJECT_ROOT}/cache/teacher_weights"
TEACHER_CKPT="${TEACHER_DIR}/checkpoint_04.pth"

download_gdrive_file \
    "19Y9oQNlYBTkoT4sYuUQWrEV9iUatPkdI" \
    "${TEACHER_CKPT}" \
    "teacher checkpoint"

# =========================
# Download student checkpoint (VQA, A-OKVQA SelTDA)
# Google Drive: checkpoint_09.pth
# https://drive.google.com/file/d/1mZbIX4lfKNgdPq6j41CWjMpPM-xl_ewj/view?usp=drive_link
# =========================

STUDENT_DIR="${PROJECT_ROOT}/cache/student_weights"
STUDENT_CKPT="${STUDENT_DIR}/checkpoint_09.pth"

download_gdrive_file \
    "1mZbIX4lfKNgdPq6j41CWjMpPM-xl_ewj" \
    "${STUDENT_CKPT}" \
    "student checkpoint"

# =========================
# Download published synthetic data (A-OKVQA)
# synthetic_data_raw.json:
# https://drive.google.com/file/d/1WH8SG1FPtUqaNNDWtFrl7SZnC-4pfWlf/view?usp=drive_link
# synthetic_data_filter.json (saved as synthetic_data.json for train_vqa.py):
# https://drive.google.com/file/d/1X_ok9p-VDT4h_4zEtLXLi7VTC5xGtSEj/view?usp=drive_link
# =========================

SYNTHETIC_RAW="${AOKVQA_DIR}/synthetic_data_raw.json"
SYNTHETIC_FILTERED="${AOKVQA_DIR}/synthetic_data.json"

download_gdrive_file \
    "1WH8SG1FPtUqaNNDWtFrl7SZnC-4pfWlf" \
    "${SYNTHETIC_RAW}" \
    "synthetic_data_raw.json"

download_gdrive_file \
    "1X_ok9p-VDT4h_4zEtLXLi7VTC5xGtSEj" \
    "${SYNTHETIC_FILTERED}" \
    "synthetic_data_filter.json → synthetic_data.json"

echo "All done!"
