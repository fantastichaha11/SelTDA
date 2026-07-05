#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

bool_value() {
    case "${1:-0}" in
        1|true|TRUE|yes|YES|on|ON) echo true ;;
        *) echo false ;;
    esac
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

checkpoint_name() {
    local epoch_count="$1"
    local idx=$((epoch_count - 1))
    if [ "${idx}" -lt 0 ]; then
        die "epoch count must be >= 1, got ${epoch_count}"
    fi
    printf 'checkpoint_%02d.pth' "${idx}"
}

DATASETS_DIR="${DATASETS_DIR:-${PROJECT_ROOT}/datasets}"
VIZWIZ_DIR="${VIZWIZ_DIR:-${DATASETS_DIR}/vizwiz}"
VIZWIZ_IMAGES="${VIZWIZ_IMAGES:-${VIZWIZ_DIR}/images}"
VIZWIZ_TRAIN_ANNOTATIONS="${VIZWIZ_TRAIN_ANNOTATIONS:-${VIZWIZ_DIR}/annotations/train.json}"
VIZWIZ_VAL_ANNOTATIONS="${VIZWIZ_VAL_ANNOTATIONS:-${VIZWIZ_DIR}/annotations/val.json}"
INCLUDE_UNANSWERABLE="$(bool_value "${INCLUDE_UNANSWERABLE:-1}")"

NUM_GPUS="${NUM_GPUS:-1}"
VQA_EPOCHS="${VQA_EPOCHS:-10}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-16}"
VQA_BATCH_SIZE_TRAIN="${VQA_BATCH_SIZE_TRAIN:-16}"
VQA_BATCH_SIZE_TEST="${VQA_BATCH_SIZE_TEST:-16}"
FILTER_KEEP_TOP="${FILTER_KEEP_TOP:-0.75}"
TRUNCATE_GENERATE="${TRUNCATE_GENERATE:-}"
TORCH_HOME_OVERRIDE="${TORCH_HOME_OVERRIDE:-null}"

BASELINE_OUTPUT_DIR="${BASELINE_OUTPUT_DIR:-${PROJECT_ROOT}/cache/vizwiz_baseline_weights}"
STUDENT_OUTPUT_DIR="${STUDENT_OUTPUT_DIR:-${PROJECT_ROOT}/cache/vizwiz_self_trained_weights}"
GEN_OUTPUT_DIR="${GEN_OUTPUT_DIR:-${PROJECT_ROOT}/cache/vizwiz_generation}"
FILTER_OUTPUT_DIR="${FILTER_OUTPUT_DIR:-${PROJECT_ROOT}/cache/vizwiz_filter}"
BASELINE_CKPT="${BASELINE_OUTPUT_DIR}/$(checkpoint_name "${VQA_EPOCHS}")"

SYNTH_RAW="${SYNTH_RAW:-${VIZWIZ_DIR}/synthetic_data_raw.json}"
SYNTH_FILTERED="${SYNTH_FILTERED:-${VIZWIZ_DIR}/synthetic_data.json}"
FILTER_REPORT="${FILTER_REPORT:-${VIZWIZ_DIR}/filter_report.json}"
SCORE_CACHE_DIR="${SCORE_CACHE_DIR:-${VIZWIZ_DIR}/score_cache}"
UNLABELED_ANNOTATIONS="${UNLABELED_ANNOTATIONS:-${VIZWIZ_DIR}/train.json}"

ENABLE_CONF="$(bool_value "${ENABLE_CONF:-1}")"
ENABLE_ITM="$(bool_value "${ENABLE_ITM:-1}")"
ENABLE_XCONS="$(bool_value "${ENABLE_XCONS:-1}")"
RUN_BASELINE="$(bool_value "${RUN_BASELINE:-1}")"

mkdir -p "${BASELINE_OUTPUT_DIR}" "${STUDENT_OUTPUT_DIR}" "${GEN_OUTPUT_DIR}" "${FILTER_OUTPUT_DIR}"

echo "========== Step 0: Download VizWiz =========="
if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
    python scripts/download_vizwiz.py --output-root "${VIZWIZ_DIR}"
else
    echo "SKIP_DOWNLOAD=1 - reusing ${VIZWIZ_DIR}"
fi

echo "========== Step 1: Convert VizWiz =========="
if [ "${SKIP_CONVERT:-0}" != "1" ]; then
    [ -f "${VIZWIZ_TRAIN_ANNOTATIONS}" ] || die "Missing ${VIZWIZ_TRAIN_ANNOTATIONS}"
    [ -f "${VIZWIZ_VAL_ANNOTATIONS}" ] || die "Missing ${VIZWIZ_VAL_ANNOTATIONS}"
    [ -d "${VIZWIZ_IMAGES}/train" ] || die "Missing ${VIZWIZ_IMAGES}/train"
    [ -d "${VIZWIZ_IMAGES}/val" ] || die "Missing ${VIZWIZ_IMAGES}/val"
    CONVERT_ARGS=()
    if [ "${INCLUDE_UNANSWERABLE}" != "true" ]; then
        CONVERT_ARGS+=(--exclude-unanswerable)
    fi
    python convert_vizwiz.py \
        --vizwiz-root "${VIZWIZ_DIR}" \
        --train-annotations "${VIZWIZ_TRAIN_ANNOTATIONS}" \
        --val-annotations "${VIZWIZ_VAL_ANNOTATIONS}" \
        --train-image-dir "${VIZWIZ_IMAGES}/train" \
        --val-image-dir "${VIZWIZ_IMAGES}/val" \
        --output-root "${VIZWIZ_DIR}" \
        "${CONVERT_ARGS[@]}"
else
    echo "SKIP_CONVERT=1 - reusing converted VizWiz JSON files"
fi

for file in train.json val.json answer_list.json vizwiz_val_metadata.json; do
    [ -f "${VIZWIZ_DIR}/${file}" ] || die "Missing converted file: ${VIZWIZ_DIR}/${file}"
done

echo "========== Step 2: Train VizWiz real-only baseline =========="
if [ "${RUN_BASELINE}" = "true" ]; then
    if [ ! -f "${BASELINE_CKPT}" ]; then
        python -m torch.distributed.run --nproc_per_node="${NUM_GPUS}" train_vqa.py \
            --output_dir="${BASELINE_OUTPUT_DIR}" \
            --config configs/vizwiz.yaml \
            --overrides \
                "ann_root='${VIZWIZ_DIR}'" \
                "vqa_root='${VIZWIZ_IMAGES}'" \
                "train_files=[train]" \
                "batch_size_train=${VQA_BATCH_SIZE_TRAIN}" \
                "batch_size_test=${VQA_BATCH_SIZE_TEST}" \
                "max_epoch=${VQA_EPOCHS}" \
                "torch_home=${TORCH_HOME_OVERRIDE}" \
                "wandb=false"
    else
        echo "Baseline checkpoint exists: ${BASELINE_CKPT}"
    fi
else
    echo "RUN_BASELINE=0 - skipping baseline train"
fi

echo "========== Step 3: Generate VizWiz pseudo-QA =========="
if [ "${SKIP_GENERATE:-0}" != "1" ]; then
    [ -f "${UNLABELED_ANNOTATIONS}" ] || die "Missing UNLABELED_ANNOTATIONS=${UNLABELED_ANNOTATIONS}"
    GEN_OVERRIDES=(
        "image_folder='${VIZWIZ_IMAGES}'"
        "output_folder='${VIZWIZ_DIR}'"
        "annotations='${UNLABELED_ANNOTATIONS}'"
        "output_annotations_name=$(basename "${SYNTH_RAW}")"
        "batch_size=${GEN_BATCH_SIZE}"
        "num_workers=4"
        "vqa_dataset_origin=vqa"
        "shuffle=true"
        "torch_home=${TORCH_HOME_OVERRIDE}"
    )
    if [ -n "${TRUNCATE_GENERATE}" ]; then
        GEN_OVERRIDES+=("truncate_to=${TRUNCATE_GENERATE}")
    fi
    python generate_questions.py \
        --output_dir="${GEN_OUTPUT_DIR}" \
        --config configs/generate_questions_pathvqa.yaml \
        --overrides "${GEN_OVERRIDES[@]}"
else
    echo "SKIP_GENERATE=1 - reusing ${SYNTH_RAW}"
fi
[ -f "${SYNTH_RAW}" ] || die "Missing raw synthetic file: ${SYNTH_RAW}"

echo "========== Step 4: Filter VizWiz pseudo-QA =========="
if [ "${SKIP_FILTER:-0}" != "1" ]; then
    if [ "${ENABLE_XCONS}" = "true" ] && [ ! -f "${BASELINE_CKPT}" ]; then
        die "X-consistency enabled but baseline checkpoint missing: ${BASELINE_CKPT}"
    fi
    python filter_pseudo.py \
        --output_dir="${FILTER_OUTPUT_DIR}" \
        --config configs/filter_pseudo.yaml \
        --overrides \
            "input='${SYNTH_RAW}'" \
            "image_root='${VIZWIZ_IMAGES}'" \
            "output='${SYNTH_FILTERED}'" \
            "report='${FILTER_REPORT}'" \
            "gates.conf.enabled=${ENABLE_CONF}" \
            "gates.conf.keep_top=${FILTER_KEEP_TOP}" \
            "gates.itm.enabled=${ENABLE_ITM}" \
            "gates.itm.keep_top=${FILTER_KEEP_TOP}" \
            "gates.xcons.enabled=${ENABLE_XCONS}" \
            "gates.xcons.keep_top=${FILTER_KEEP_TOP}" \
            "gates.xcons.student_ckpt='${BASELINE_CKPT}'" \
            "score_cache.dir='${SCORE_CACHE_DIR}'"
else
    echo "SKIP_FILTER=1 - reusing ${SYNTH_FILTERED}"
fi
[ -f "${SYNTH_FILTERED}" ] || die "Missing filtered synthetic file: ${SYNTH_FILTERED}"

echo "========== Step 5: Train VizWiz augmented student =========="
if [ "${SKIP_TRAIN:-0}" != "1" ]; then
    python -m torch.distributed.run --nproc_per_node="${NUM_GPUS}" train_vqa.py \
        --output_dir="${STUDENT_OUTPUT_DIR}" \
        --config configs/vizwiz.yaml \
        --overrides \
            "ann_root='${VIZWIZ_DIR}'" \
            "vqa_root='${VIZWIZ_IMAGES}'" \
            "train_files=[train,$(basename "${SYNTH_FILTERED}" .json)]" \
            "batch_size_train=${VQA_BATCH_SIZE_TRAIN}" \
            "batch_size_test=${VQA_BATCH_SIZE_TEST}" \
            "max_epoch=${VQA_EPOCHS}" \
            "torch_home=${TORCH_HOME_OVERRIDE}" \
            "wandb=false"
else
    echo "SKIP_TRAIN=1 - skipping student train"
fi

echo "========== Step 6: Evaluate VizWiz =========="
RESULT_FILE="${STUDENT_OUTPUT_DIR}/result/vqa_result.json"
if [ "${SKIP_EVAL:-0}" != "1" ] && [ -f "${RESULT_FILE}" ]; then
    python vizwiz_eval.py \
        --annotation-file "${VIZWIZ_DIR}/val.json" \
        --metadata-file "${VIZWIZ_DIR}/vizwiz_val_metadata.json" \
        "${RESULT_FILE}"
fi
