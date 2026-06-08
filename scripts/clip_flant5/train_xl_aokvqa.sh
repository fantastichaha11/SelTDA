#!/usr/bin/env bash
# Stage-2 fine-tune CLIP-FlanT5-XL on A-OKVQA (DeepSpeed ZeRO-3).
# Prerequisites: bash scripts/clip_flant5/setup_vast_aokvqa.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SELTDA="${SELTDA:-${ROOT}}"
CLIP_ROOT="${CLIP_ROOT:-${SELTDA}/external/CLIP-FlanT5}"
PLAYGROUND="${PLAYGROUND:-${CLIP_ROOT}/playground/data}"
DATA_JSON="${DATA_JSON:-${SELTDA}/datasets/clip_flant5/aokvqa_vqa.json}"
STAGE1_CKPT="${STAGE1_CKPT:-${CLIP_ROOT}/checkpoints/clip-flant5-xl-stage-1/mm_projector.bin}"
OUTPUT_DIR="${OUTPUT_DIR:-${CLIP_ROOT}/checkpoints/clip-flant5-xl-aokvqa}"

NUM_GPUS="${NUM_GPUS:-8}"
PER_DEVICE_BS="${PER_DEVICE_BS:-6}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-5}"

log() { echo "[$(date -Is)] $*"; }

if [[ ! -f "${DATA_JSON}" ]]; then
  echo "Missing ${DATA_JSON}; run setup_vast_aokvqa.sh first" >&2
  exit 1
fi
if [[ ! -f "${STAGE1_CKPT}" ]]; then
  echo "Missing stage-1 projector ${STAGE1_CKPT}" >&2
  exit 1
fi

cd "${CLIP_ROOT}"
mkdir -p "${OUTPUT_DIR}"

log "========== CLIP-FlanT5-XL A-OKVQA train =========="
log "GPUs=${NUM_GPUS} data=${DATA_JSON} out=${OUTPUT_DIR}"

deepspeed --num_gpus="${NUM_GPUS}" llava/train/t5_train_mem.py \
  --deepspeed ./scripts/zero3.json \
  --model_name_or_path google/flan-t5-xl \
  --version t5_v1 \
  --data_path "${DATA_JSON}" \
  --image_folder "${PLAYGROUND}" \
  --vision_tower openai/clip-vit-large-patch14-336 \
  --pretrain_mm_mlp_adapter "${STAGE1_CKPT}" \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --bf16 True \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs "${EPOCHS}" \
  --per_device_train_batch_size "${PER_DEVICE_BS}" \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --evaluation_strategy no \
  --save_strategy steps \
  --save_steps 5000 \
  --save_total_limit 2 \
  --learning_rate "${LR}" \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --report_to none

log "========== Train finished =========="
log "Export HF checkpoint for SelTDA reward:"
log "  reward.vqascore_model=clip-flant5-xl"
log "  reward.vqascore_checkpoint=${OUTPUT_DIR}"
