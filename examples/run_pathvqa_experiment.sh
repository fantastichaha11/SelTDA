#!/usr/bin/env bash
# End-to-end PathVQA direct SelTDA experiment.
#
# Default flow:
#   1) download PathVQA from HuggingFace into datasets/pathvqa
#   2) convert PathVQA to SelTDA JSON files
#   3) train a PathVQA VQG teacher
#   4) generate pseudo-QA from held-out validation images
#   5) train a real-only PathVQA baseline for x-consistency scoring
#   6) filter pseudo-QA
#   7) train a PathVQA student on train + filtered synthetic
#   8) evaluate with exact match on PathVQA test
#
# Prerequisites:
#   conda activate vqa
#   export PYTHONNOUSERSITE=1
#
# Useful env vars:
#   SKIP_DOWNLOAD=1          reuse datasets/pathvqa/all_data.json + images/
#   SKIP_CONVERT=1           reuse train/val/test/answer_list JSON files
#   SKIP_TEACHER=1           reuse PATHVQA_TEACHER_CKPT or cache/pathvqa_teacher_weights/checkpoint_04.pth
#   SKIP_GENERATE=1          reuse datasets/pathvqa/synthetic_data_raw.json
#   RUN_BASELINE=0           do not train real-only baseline; set ENABLE_XCONS=0 or XCONS_STUDENT_CKPT=...
#   SKIP_FILTER=1            reuse datasets/pathvqa/synthetic_data.json
#   SKIP_TRAIN=1             do not train final student; eval existing result if present
#   SKIP_EVAL=1              skip pathvqa_eval.py
#   UNLABELED_SPLIT=val      default. Use test_val_combined only for transductive experiments.
#   TRUNCATE_GENERATE=N      debug generation on first N unlabeled images
#   FILTER_KEEP_TOP=0.75     quantile keep ratio per enabled gate
#   NUM_GPUS=1               GPUs for torch.distributed.run
#   VQA_EPOCHS=10            max_epoch for baseline and final student
#   TEACHER_EPOCHS=5         max_epoch for VQG teacher

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
PATHVQA_DIR="${PATHVQA_DIR:-${DATASETS_DIR}/pathvqa}"
PATHVQA_IMAGES="${PATHVQA_IMAGES:-${PATHVQA_DIR}/images}"
MED_CONFIG="${MED_CONFIG:-${PROJECT_ROOT}/configs/med_config.json}"

NUM_GPUS="${NUM_GPUS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
TORCH_HOME_OVERRIDE="${TORCH_HOME_OVERRIDE:-null}"

TEACHER_EPOCHS="${TEACHER_EPOCHS:-5}"
VQA_EPOCHS="${VQA_EPOCHS:-10}"
TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-16}"
VQA_BATCH_SIZE_TRAIN="${VQA_BATCH_SIZE_TRAIN:-16}"
VQA_BATCH_SIZE_TEST="${VQA_BATCH_SIZE_TEST:-16}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-16}"
QUESTIONS_PER_IMAGE="${QUESTIONS_PER_IMAGE:-2}"
FILTER_KEEP_TOP="${FILTER_KEEP_TOP:-0.75}"
TRUNCATE_TRAIN_DATASET_TO="${TRUNCATE_TRAIN_DATASET_TO:-null}"

TEACHER_OUTPUT_DIR="${TEACHER_OUTPUT_DIR:-${PROJECT_ROOT}/cache/pathvqa_teacher_weights}"
BASELINE_OUTPUT_DIR="${BASELINE_OUTPUT_DIR:-${PROJECT_ROOT}/cache/pathvqa_baseline_weights}"
STUDENT_OUTPUT_DIR="${STUDENT_OUTPUT_DIR:-${PROJECT_ROOT}/cache/pathvqa_self_trained_weights}"
GEN_OUTPUT_DIR="${GEN_OUTPUT_DIR:-${PROJECT_ROOT}/cache/pathvqa_generation}"
FILTER_OUTPUT_DIR="${FILTER_OUTPUT_DIR:-${PROJECT_ROOT}/cache/pathvqa_filter}"

TEACHER_CKPT_DEFAULT="${TEACHER_OUTPUT_DIR}/$(checkpoint_name "${TEACHER_EPOCHS}")"
PATHVQA_TEACHER_CKPT="${PATHVQA_TEACHER_CKPT:-${TEACHER_CKPT_DEFAULT}}"
BASELINE_CKPT_DEFAULT="${BASELINE_OUTPUT_DIR}/$(checkpoint_name "${VQA_EPOCHS}")"
XCONS_STUDENT_CKPT="${XCONS_STUDENT_CKPT:-${BASELINE_CKPT_DEFAULT}}"

UNLABELED_SPLIT="${UNLABELED_SPLIT:-val}"
UNLABELED_ANNOTATIONS="${UNLABELED_ANNOTATIONS:-${PATHVQA_DIR}/${UNLABELED_SPLIT}.json}"
SYNTH_RAW="${SYNTH_RAW:-${PATHVQA_DIR}/synthetic_data_raw.json}"
SYNTH_FILTERED="${SYNTH_FILTERED:-${PATHVQA_DIR}/synthetic_data.json}"
FILTER_REPORT="${FILTER_REPORT:-${PATHVQA_DIR}/filter_report.json}"
SCORE_CACHE_DIR="${SCORE_CACHE_DIR:-${PATHVQA_DIR}/score_cache}"

ENABLE_CONF="$(bool_value "${ENABLE_CONF:-1}")"
ENABLE_ITM="$(bool_value "${ENABLE_ITM:-1}")"
ENABLE_XCONS="$(bool_value "${ENABLE_XCONS:-1}")"
RUN_BASELINE="$(bool_value "${RUN_BASELINE:-1}")"
WANDB_ENABLED="$(bool_value "${WANDB_ENABLED:-1}")"
WANDB_PROJECT="${WANDB_PROJECT:-seltda}"
WANDB_ENTITY_OVERRIDE="${WANDB_ENTITY_OVERRIDE:-null}"
WANDB_MODE_OVERRIDE="${WANDB_MODE_OVERRIDE:-online}"

if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "vqa" ]; then
    echo "Warning: recommended to run inside conda env 'vqa' (conda activate vqa)"
fi
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

mkdir -p "${PATHVQA_DIR}" "${TEACHER_OUTPUT_DIR}" "${BASELINE_OUTPUT_DIR}" \
    "${STUDENT_OUTPUT_DIR}" "${GEN_OUTPUT_DIR}" "${FILTER_OUTPUT_DIR}"

echo "========== Step 0: PathVQA download =========="
if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
    python scripts/download_pathvqa.py --output-root "${PATHVQA_DIR}"
else
    echo "SKIP_DOWNLOAD=1 - reusing ${PATHVQA_DIR}"
fi
[ -f "${PATHVQA_DIR}/all_data.json" ] || die "Missing ${PATHVQA_DIR}/all_data.json"
[ -d "${PATHVQA_IMAGES}" ] || die "Missing ${PATHVQA_IMAGES}"

echo "========== Step 1: Convert PathVQA annotations =========="
if [ "${SKIP_CONVERT:-0}" != "1" ]; then
    python convert_pathvqa.py \
        --pathvqa-root "${PATHVQA_DIR}" \
        --images-dir "${PATHVQA_IMAGES}"
else
    echo "SKIP_CONVERT=1 - reusing converted JSON files"
fi
for name in train val test answer_list; do
    [ -f "${PATHVQA_DIR}/${name}.json" ] || die "Missing ${PATHVQA_DIR}/${name}.json"
done
[ -f "${UNLABELED_ANNOTATIONS}" ] || die "Missing unlabeled annotations: ${UNLABELED_ANNOTATIONS}"

echo "========== Step 2: Train PathVQA VQG teacher =========="
if [ "${SKIP_TEACHER:-0}" != "1" ]; then
    if [ -f "${PATHVQA_TEACHER_CKPT}" ] && [ "${REBUILD_TEACHER:-0}" != "1" ]; then
        echo "Teacher checkpoint exists - skipping teacher train: ${PATHVQA_TEACHER_CKPT}"
    else
        python -m torch.distributed.run \
            --master_port="${TEACHER_PORT:-37771}" \
            --nproc_per_node="${NUM_GPUS}" \
            train_vqg.py \
            --output_dir="${TEACHER_OUTPUT_DIR}" \
            --config configs/pathvqg.yaml \
            --overrides \
                "vqa_root='${PATHVQA_IMAGES}'" \
                "ann_root='${PATHVQA_DIR}'" \
                "batch_size=${TEACHER_BATCH_SIZE}" \
                "max_epoch=${TEACHER_EPOCHS}" \
                "torch_home=${TORCH_HOME_OVERRIDE}" \
                "wandb=${WANDB_ENABLED}" \
                "wandb_project='${WANDB_PROJECT}'" \
                "wandb_entity=${WANDB_ENTITY_OVERRIDE}" \
                "wandb_name='pathvqa-teacher'" \
                "wandb_group='pathvqa'" \
                "wandb_mode='${WANDB_MODE_OVERRIDE}'" \
                "+save_last_only=false"
    fi
else
    echo "SKIP_TEACHER=1 - reusing ${PATHVQA_TEACHER_CKPT}"
fi
[ -f "${PATHVQA_TEACHER_CKPT}" ] || die "Missing teacher checkpoint: ${PATHVQA_TEACHER_CKPT}"

echo "========== Step 3: Generate pseudo-QA =========="
echo "Unlabeled annotation pool: ${UNLABELED_ANNOTATIONS}"
if [ "${SKIP_GENERATE:-0}" != "1" ]; then
    GENERATE_OVERRIDES=(
        "image_folder='${PATHVQA_IMAGES}'"
        "output_folder='${PATHVQA_DIR}'"
        "annotations='${UNLABELED_ANNOTATIONS}'"
        "pretrained='${PATHVQA_TEACHER_CKPT}'"
        "output_annotations_name=$(basename "${SYNTH_RAW}")"
        "multimodal_encoder_decoder_config='${MED_CONFIG}'"
        "questions_per_image=${QUESTIONS_PER_IMAGE}"
        "max_length=40"
        "batch_size=${GEN_BATCH_SIZE}"
        "num_workers=${NUM_WORKERS}"
        "vqa_dataset_origin=vqa"
        "shuffle=true"
        "torch_home=${TORCH_HOME_OVERRIDE}"
    )
    if [ -n "${TRUNCATE_GENERATE:-}" ]; then
        GENERATE_OVERRIDES+=("truncate_to=${TRUNCATE_GENERATE}")
    fi

    python generate_questions.py \
        --output_dir="${GEN_OUTPUT_DIR}" \
        --config configs/generate_questions_pathvqa.yaml \
        --overrides "${GENERATE_OVERRIDES[@]}"
else
    echo "SKIP_GENERATE=1 - reusing ${SYNTH_RAW}"
fi
[ -f "${SYNTH_RAW}" ] || die "Missing raw synthetic file: ${SYNTH_RAW}"

echo "========== Step 4: Train real-only baseline for x-consistency =========="
if [ "${RUN_BASELINE}" = "true" ]; then
    if [ -f "${BASELINE_CKPT_DEFAULT}" ] && [ "${REBUILD_BASELINE:-0}" != "1" ]; then
        echo "Baseline checkpoint exists - skipping baseline train: ${BASELINE_CKPT_DEFAULT}"
    else
        BASELINE_RESUME_ARGS=()
        if [ "${REBUILD_BASELINE:-0}" = "1" ]; then
            BASELINE_RESUME_ARGS+=(--no-resume)
        fi
        python -m torch.distributed.run \
            --master_port="${BASELINE_PORT:-37772}" \
            --nproc_per_node="${NUM_GPUS}" \
            train_vqa.py \
            --output_dir="${BASELINE_OUTPUT_DIR}" \
            --config configs/pathvqa.yaml \
            "${BASELINE_RESUME_ARGS[@]}" \
            --overrides \
                "vqa_root='${PATHVQA_IMAGES}'" \
                "ann_root='${PATHVQA_DIR}'" \
                "train_files=[train]" \
                "batch_size_train=${VQA_BATCH_SIZE_TRAIN}" \
                "batch_size_test=${VQA_BATCH_SIZE_TEST}" \
                "max_epoch=${VQA_EPOCHS}" \
                "truncate_train_dataset_to=null" \
                "torch_home=${TORCH_HOME_OVERRIDE}" \
                "wandb=${WANDB_ENABLED}" \
                "wandb_project='${WANDB_PROJECT}'" \
                "wandb_entity=${WANDB_ENTITY_OVERRIDE}" \
                "wandb_name='pathvqa-baseline'" \
                "wandb_group='pathvqa'" \
                "wandb_mode='${WANDB_MODE_OVERRIDE}'" \
                "+save_last_only=false" \
                "+max_checkpoints=3"
    fi
    if [ -f "${BASELINE_OUTPUT_DIR}/result/vqa_result.json" ] && [ "${SKIP_EVAL:-0}" != "1" ]; then
        BASELINE_EVAL_WANDB_ARGS=()
        if [ "${WANDB_ENABLED}" = "true" ]; then
            BASELINE_EVAL_WANDB_ARGS+=(
                --wandb
                --wandb-project "${WANDB_PROJECT}"
                --wandb-name "pathvqa-baseline-eval"
                --wandb-group "pathvqa"
                --wandb-mode "${WANDB_MODE_OVERRIDE}"
            )
            if [ -n "${WANDB_ENTITY_OVERRIDE}" ] && [ "${WANDB_ENTITY_OVERRIDE}" != "null" ]; then
                BASELINE_EVAL_WANDB_ARGS+=(--wandb-entity "${WANDB_ENTITY_OVERRIDE}")
            fi
        fi
        python pathvqa_eval.py \
            --annotation-file "${PATHVQA_DIR}/test.json" \
            "${BASELINE_EVAL_WANDB_ARGS[@]}" \
            "${BASELINE_OUTPUT_DIR}/result/vqa_result.json"
    fi
else
    echo "RUN_BASELINE=0 - skipping baseline train"
fi
if [ "${ENABLE_XCONS}" = "true" ] && [ ! -f "${XCONS_STUDENT_CKPT}" ]; then
    die "X-consistency is enabled but checkpoint is missing: ${XCONS_STUDENT_CKPT}. Run with RUN_BASELINE=1, set XCONS_STUDENT_CKPT, or set ENABLE_XCONS=0."
fi

echo "========== Step 5: Filter pseudo-labels =========="
if [ "${SKIP_FILTER:-0}" != "1" ]; then
    python filter_pseudo.py \
        --output_dir="${FILTER_OUTPUT_DIR}" \
        --config configs/filter_pseudo.yaml \
        --overrides \
            "input='${SYNTH_RAW}'" \
            "image_root='${PATHVQA_IMAGES}'" \
            "output='${SYNTH_FILTERED}'" \
            "report='${FILTER_REPORT}'" \
            "device=${DEVICE}" \
            "torch_home=${TORCH_HOME_OVERRIDE}" \
            "gates.conf.enabled=${ENABLE_CONF}" \
            "gates.conf.keep_top=${FILTER_KEEP_TOP}" \
            "gates.itm.enabled=${ENABLE_ITM}" \
            "gates.itm.keep_top=${FILTER_KEEP_TOP}" \
            "gates.xcons.enabled=${ENABLE_XCONS}" \
            "gates.xcons.keep_top=${FILTER_KEEP_TOP}" \
            "gates.xcons.student_ckpt='${XCONS_STUDENT_CKPT}'" \
            "gates.xcons.med_config='${MED_CONFIG}'" \
            "score_cache.dir='${SCORE_CACHE_DIR}'"
else
    echo "SKIP_FILTER=1 - reusing ${SYNTH_FILTERED}"
fi
[ -f "${SYNTH_FILTERED}" ] || die "Missing filtered synthetic file: ${SYNTH_FILTERED}"

echo "========== Step 6: Train PathVQA student on train + filtered synthetic =========="
if [ "${SKIP_TRAIN:-0}" != "1" ]; then
    STUDENT_RESUME_ARGS=()
    if [ "${REBUILD_STUDENT:-0}" = "1" ]; then
        STUDENT_RESUME_ARGS+=(--no-resume)
    fi
    python -m torch.distributed.run \
        --master_port="${STUDENT_PORT:-37773}" \
        --nproc_per_node="${NUM_GPUS}" \
        train_vqa.py \
        --output_dir="${STUDENT_OUTPUT_DIR}" \
        --config configs/pathvqa.yaml \
        "${STUDENT_RESUME_ARGS[@]}" \
        --overrides \
            "vqa_root='${PATHVQA_IMAGES}'" \
            "ann_root='${PATHVQA_DIR}'" \
            "train_files=[train,$(basename "${SYNTH_FILTERED}" .json)]" \
            "batch_size_train=${VQA_BATCH_SIZE_TRAIN}" \
            "batch_size_test=${VQA_BATCH_SIZE_TEST}" \
            "max_epoch=${VQA_EPOCHS}" \
            "truncate_train_dataset_to=${TRUNCATE_TRAIN_DATASET_TO}" \
            "torch_home=${TORCH_HOME_OVERRIDE}" \
            "wandb=${WANDB_ENABLED}" \
            "wandb_project='${WANDB_PROJECT}'" \
            "wandb_entity=${WANDB_ENTITY_OVERRIDE}" \
            "wandb_name='pathvqa-student'" \
            "wandb_group='pathvqa'" \
            "wandb_mode='${WANDB_MODE_OVERRIDE}'" \
            "+save_last_only=false" \
            "+max_checkpoints=3"
else
    echo "SKIP_TRAIN=1 - not training final student"
fi

echo "========== Step 7: Evaluate =========="
RESULT_FILE="${STUDENT_OUTPUT_DIR}/result/vqa_result.json"
if [ "${SKIP_EVAL:-0}" != "1" ]; then
    if [ -f "${RESULT_FILE}" ]; then
        FINAL_EVAL_WANDB_ARGS=()
        if [ "${WANDB_ENABLED}" = "true" ]; then
            FINAL_EVAL_WANDB_ARGS+=(
                --wandb
                --wandb-project "${WANDB_PROJECT}"
                --wandb-name "pathvqa-student-eval"
                --wandb-group "pathvqa"
                --wandb-mode "${WANDB_MODE_OVERRIDE}"
            )
            if [ -n "${WANDB_ENTITY_OVERRIDE}" ] && [ "${WANDB_ENTITY_OVERRIDE}" != "null" ]; then
                FINAL_EVAL_WANDB_ARGS+=(--wandb-entity "${WANDB_ENTITY_OVERRIDE}")
            fi
        fi
        python pathvqa_eval.py \
            --annotation-file "${PATHVQA_DIR}/test.json" \
            "${FINAL_EVAL_WANDB_ARGS[@]}" \
            "${RESULT_FILE}"
    else
        echo "No final result file found at ${RESULT_FILE}; skipping eval"
    fi
else
    echo "SKIP_EVAL=1 - skipping final eval"
fi

echo "========== PathVQA experiment finished =========="
echo "PathVQA dir:        ${PATHVQA_DIR}"
echo "Teacher checkpoint: ${PATHVQA_TEACHER_CKPT}"
echo "Raw synthetic:      ${SYNTH_RAW}"
echo "Filtered synthetic: ${SYNTH_FILTERED}"
echo "Filter report:      ${FILTER_REPORT}"
echo "Student output:     ${STUDENT_OUTPUT_DIR}"
