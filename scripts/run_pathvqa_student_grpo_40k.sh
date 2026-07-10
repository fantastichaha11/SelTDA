#!/usr/bin/env bash
set -euo pipefail

LOG_PATH="${1:-outputs/logs/pathvqa_student_grpo_40k.log}"
mkdir -p "$(dirname "${LOG_PATH}")" outputs/student/pathvqa_grpo_40k

python3 -m torch.distributed.run \
  --master_port="${MASTER_PORT:-37783}" \
  --nproc_per_node="${NUM_GPUS:-1}" \
  train_vqa.py \
  --output_dir=outputs/student/pathvqa_grpo_40k \
  --config configs/pathvqa.yaml \
  --no-resume \
  --overrides \
    "vqa_root='datasets/pathvqa/images'" \
    "ann_root='datasets/pathvqa'" \
    "train_files=[train,synthetic_grpo_teacher_200x8]" \
    "truncate_train_dataset_to=40000" \
    "batch_size_train=16" \
    "batch_size_test=16" \
    "max_epoch=10" \
    "torch_home=null" \
    "wandb=false" \
    "wandb_mode=disabled" \
    "wandb_name='pathvqa-student-grpo-40k'" \
    "+save_last_only=false" \
    "+max_checkpoints=3" \
  2>&1 | tee -a "${LOG_PATH}"
