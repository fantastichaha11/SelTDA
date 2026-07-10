#!/bin/bash
# Just provide the file name without the extension.
# The reader code will look at the config file (in this case, configs/aokvqa.yaml)
# and read both the synthetic JSON and real JSON from the root folder of the dataset.
OUTPUT_DIR=cache/self_trained_weights
TRAIN_FILES="[train,synthetic_data]"
python -m torch.distributed.run --nproc_per_node=1 train_vqa.py \
    --output_dir=$OUTPUT_DIR \
    --config configs/aokvqa.yaml \
    --overrides train_files=$TRAIN_FILES \
    truncate_train_dataset_to=34000