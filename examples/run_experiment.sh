#!/bin/bash
# End-to-end A-OKVQA filtering experiment:
#   1) dataset.sh  — A-OKVQA + COCO (incl. unlabeled2017) + teacher checkpoint
#   2) convert_aokvqa.py
#   3) generate_questions.py — pseudo-QA from unlabeled COCO
#   4) filter_pseudo.py — 3-gate filtering
#   5) train_vqa.py — self-train student on train + filtered synthetic
#
# Prerequisites:
#   conda activate vqa
#   export PYTHONNOUSERSITE=1
#
# Optional env vars:
#   SKIP_DATASET=1          skip dataset.sh (data + checkpoint already present)
#   SKIP_GENERATE=1         skip generation (reuse synthetic_data_raw.json)
#   SKIP_FILTER=1           skip filtering (reuse synthetic_data.json)
#   SKIP_TRAIN=1            stop after filter
#   TRUNCATE_GENERATE=N     limit unlabeled images during generation (debug)
#   FILTER_KEEP_TOP=0.75    quantile keep ratio per gate (default 0.75)
#   NUM_GPUS=1              GPUs for train_vqa.py
#
# Teacher checkpoint (downloaded by dataset.sh if missing):
#   https://drive.google.com/file/d/19Y9oQNlYBTkoT4sYuUQWrEV9iUatPkdI/view
#
# COCO unlabeled (downloaded by dataset.sh if missing):
#   http://images.cocodataset.org/zips/unlabeled2017.zip

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

DATASETS_DIR="${PROJECT_ROOT}/datasets"
COCO_DIR="${DATASETS_DIR}/coco2017"
COCO_UNLABELED="${COCO_DIR}/unlabeled2017"
AOKVQA_DIR="${DATASETS_DIR}/aokvqa"
TEACHER_CKPT="${PROJECT_ROOT}/cache/teacher_weights/checkpoint_04.pth"
OUTPUT_DIR="${PROJECT_ROOT}/cache/self_trained_weights"
MED_CONFIG="${PROJECT_ROOT}/configs/med_config.json"

FILTER_KEEP_TOP="${FILTER_KEEP_TOP:-0.75}"
NUM_GPUS="${NUM_GPUS:-1}"

if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "vqa" ]; then
    echo "Warning: recommended to run inside conda env 'vqa' (conda activate vqa)"
fi

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

echo "========== Step 0: Download datasets + teacher checkpoint =========="
if [ "${SKIP_DATASET:-0}" != "1" ]; then
    bash "${PROJECT_ROOT}/dataset.sh"
else
    echo "SKIP_DATASET=1 — skipping dataset.sh"
fi

if [ ! -f "${TEACHER_CKPT}" ]; then
    echo "Missing teacher checkpoint: ${TEACHER_CKPT}"
    echo "Run: bash dataset.sh   or set SKIP_DATASET=0"
    exit 1
fi

if [ ! -d "${COCO_UNLABELED}" ]; then
    echo "Missing unlabeled COCO folder: ${COCO_UNLABELED}"
    echo "Run: bash dataset.sh"
    exit 1
fi

echo "========== Step 1: Convert A-OKVQA annotations =========="
if [ ! -f "${AOKVQA_DIR}/train.json" ]; then
    python convert_aokvqa.py \
        --config configs/aokvqa.yaml \
        --overrides \
            "vqa_root='${COCO_DIR}'" \
            "ann_root='${AOKVQA_DIR}'"
else
    echo "train.json exists — skipping convert_aokvqa.py"
fi

echo "========== Step 2: Generate pseudo-QA from unlabeled COCO =========="
if [ "${SKIP_GENERATE:-0}" != "1" ]; then
    GENERATE_OVERRIDES=(
        "image_folder='${COCO_UNLABELED}'"
        "output_folder='${AOKVQA_DIR}'"
        "pretrained='${TEACHER_CKPT}'"
        "output_annotations_name=synthetic_data_raw.json"
        "multimodal_encoder_decoder_config='${MED_CONFIG}'"
        "questions_per_image=2"
        "max_length=40"
        "batch_size=16"
        "num_workers=4"
        "shuffle=true"
    )
    if [ -n "${TRUNCATE_GENERATE:-}" ]; then
        GENERATE_OVERRIDES+=("truncate_to=${TRUNCATE_GENERATE}")
    fi

    python generate_questions.py \
        --config configs/generate_questions_coco.yaml \
        --overrides "${GENERATE_OVERRIDES[@]}"
else
    echo "SKIP_GENERATE=1 — reusing ${AOKVQA_DIR}/synthetic_data_raw.json"
fi

if [ ! -f "${AOKVQA_DIR}/synthetic_data_raw.json" ]; then
    echo "Expected raw synthetic file not found: ${AOKVQA_DIR}/synthetic_data_raw.json"
    exit 1
fi

echo "========== Step 3: Filter pseudo-labels (Conf + ITM + X-cons) =========="
if [ "${SKIP_FILTER:-0}" != "1" ]; then
    python filter_pseudo.py \
        --config configs/filter_pseudo.yaml \
        --overrides \
            "input='${AOKVQA_DIR}/synthetic_data_raw.json'" \
            "image_root='${COCO_DIR}'" \
            "output='${AOKVQA_DIR}/synthetic_data.json'" \
            "report='${AOKVQA_DIR}/filter_report.json'" \
            "gates.conf.enabled=true" \
            "gates.conf.keep_top=${FILTER_KEEP_TOP}" \
            "gates.itm.enabled=true" \
            "gates.itm.keep_top=${FILTER_KEEP_TOP}" \
            "gates.xcons.enabled=true" \
            "gates.xcons.keep_top=${FILTER_KEEP_TOP}" \
            "gates.xcons.student_ckpt='https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'"
else
    echo "SKIP_FILTER=1 — reusing ${AOKVQA_DIR}/synthetic_data.json"
fi

if [ ! -f "${AOKVQA_DIR}/synthetic_data.json" ]; then
    echo "Expected filtered synthetic file not found: ${AOKVQA_DIR}/synthetic_data.json"
    exit 1
fi

if [ "${SKIP_TRAIN:-0}" = "1" ]; then
    echo "SKIP_TRAIN=1 — done after filtering."
    exit 0
fi

echo "========== Step 4: Self-train student (train + filtered synthetic) =========="
python -m torch.distributed.run --nproc_per_node="${NUM_GPUS}" train_vqa.py \
    --output_dir="${OUTPUT_DIR}" \
    --config configs/aokvqa.yaml \
    --overrides \
        "vqa_root='${COCO_DIR}'" \
        "ann_root='${AOKVQA_DIR}'" \
        "train_files=[train,synthetic_data]" \
        "truncate_train_dataset_to=34000" \
        "wandb=false"

echo "========== Experiment finished =========="
echo "Filtered synthetic: ${AOKVQA_DIR}/synthetic_data.json"
echo "Filter report:      ${AOKVQA_DIR}/filter_report.json"
echo "Student checkpoints: ${OUTPUT_DIR}"
