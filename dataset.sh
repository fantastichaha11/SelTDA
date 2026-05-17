#!/bin/bash

set -e

# =========================
# Save project root
# =========================

PROJECT_ROOT=$(pwd)

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

echo "Downloading A-OKVQA..."

curl -fsSL \
https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz \
| tar xvz -C "${AOKVQA_DIR}"

# =========================
# Download COCO2017
# =========================

cd "${COCO_DIR}"

echo "Downloading COCO2017 train images..."

aria2c -x 16 -s 32 \
http://images.cocodataset.org/zips/train2017.zip

echo "Downloading COCO2017 val images..."

aria2c -x 16 -s 32 \
http://images.cocodataset.org/zips/val2017.zip

echo "Downloading COCO2017 test images..."

aria2c -x 16 -s 32 \
http://images.cocodataset.org/zips/test2017.zip

# =========================
# Extract COCO
# =========================

echo "Extracting COCO2017..."

unzip -q -j train2017.zip
rm train2017.zip

unzip -q -j val2017.zip
rm val2017.zip

unzip -q -j test2017.zip
rm test2017.zip

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

echo "All done!"