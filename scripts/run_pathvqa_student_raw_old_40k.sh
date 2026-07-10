#!/usr/bin/env bash
set -euo pipefail

LOG_PATH="${1:-outputs/logs/pathvqa_student_raw_old_40k.log}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/student/pathvqa_raw_old_40k}"
mkdir -p "$(dirname "${LOG_PATH}")" "${OUTPUT_DIR}"

PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
/usr/bin/python3 -m torch.distributed.run \
  --master_port="${MASTER_PORT:-37793}" \
  --nproc_per_node="${NUM_GPUS:-1}" \
  train_vqa.py \
  --output_dir="${OUTPUT_DIR}" \
  --config configs/pathvqa.yaml \
  --no-resume \
  --overrides \
    "vqa_root='datasets/pathvqa/images'" \
    "ann_root='datasets/pathvqa'" \
    "train_files=[train,synthetic_data_raw]" \
    "truncate_train_dataset_to=40000" \
    "batch_size_train=16" \
    "batch_size_test=1" \
    "max_epoch=10" \
    "torch_home=null" \
    "wandb=false" \
    "wandb_mode=disabled" \
    "wandb_name='pathvqa-student-raw-old-40k'" \
    "+save_last_only=false" \
    "+max_checkpoints=3" \
  2>&1 | tee -a "${LOG_PATH}"
