#!/usr/bin/env bash
set -euo pipefail

TS="${TS:-$(date -u +%Y%m%d_%H%M%S)}"
LOG_PATH="${1:-outputs/logs/pathvqa_original_teacher_regen_train_40k_${TS}.log}"
GEN_DIR="${GEN_DIR:-outputs/generated_qa/pathvqa_original_teacher_regen_${TS}}"
SYNTH_NAME="${SYNTH_NAME:-synthetic_original_teacher_regen_${TS}}"
SYNTH_PATH="datasets/pathvqa/${SYNTH_NAME}.json"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/student/pathvqa_original_teacher_regen_40k_${TS}}"

mkdir -p "$(dirname "${LOG_PATH}")" "${GEN_DIR}" "${OUTPUT_DIR}"

{
  echo "[pipeline] timestamp=${TS}"
  echo "[pipeline] generate_dir=${GEN_DIR}"
  echo "[pipeline] synthetic=${SYNTH_PATH}"
  echo "[pipeline] output_dir=${OUTPUT_DIR}"

  /usr/bin/python3 generate_questions.py \
    --config configs/generate_questions_pathvqa.yaml \
    --output_dir="${GEN_DIR}" \
    --overrides \
      "image_folder=datasets/pathvqa/images" \
      "output_folder=${GEN_DIR}" \
      "annotations=outputs/generated_qa/pathvqa_grpo_teacher_200x8/pathvqa_train_unique_images.json" \
      "pretrained=cache/pathvqa_teacher_weights/checkpoint_04.pth" \
      "output_annotations_name=generated_qa.json" \
      "multimodal_encoder_decoder_config=configs/med_config.json" \
      "torch_home=null" \
      "batch_size=32" \
      "questions_per_image=2"

  /usr/bin/python3 scripts/prepare_pathvqa_synthetic.py \
    "${GEN_DIR}/generated_qa.json" \
    "${SYNTH_PATH}"

  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  /usr/bin/python3 -m torch.distributed.run \
    --master_port="${MASTER_PORT:-37803}" \
    --nproc_per_node="${NUM_GPUS:-1}" \
    train_vqa.py \
    --output_dir="${OUTPUT_DIR}" \
    --config configs/pathvqa.yaml \
    --no-resume \
    --overrides \
      "vqa_root='datasets/pathvqa/images'" \
      "ann_root='datasets/pathvqa'" \
      "train_files=[train,${SYNTH_NAME}]" \
      "truncate_train_dataset_to=40000" \
      "batch_size_train=16" \
      "batch_size_test=1" \
      "max_epoch=10" \
      "torch_home=null" \
      "wandb=false" \
      "wandb_mode=disabled" \
      "wandb_name='pathvqa-student-original-teacher-regen-40k'" \
      "+save_last_only=false" \
      "+max_checkpoints=3"

  /usr/bin/python3 pathvqa_eval.py \
    "${OUTPUT_DIR}/result/vqa_result.json" \
    --annotation-file datasets/pathvqa/test.json \
    --wandb-mode disabled
} 2>&1 | tee -a "${LOG_PATH}"
