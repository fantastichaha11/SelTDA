#!/bin/bash
# Filter (VQAScore gate only) → train student → eval on A-OKVQA val.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

LOG_DIR="${PROJECT_ROOT}/cache/logs/vqascore_run"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/pipeline.log") 2>&1

echo "=== SelTDA VQAScore-only pipeline $(date -Is) ==="

export PYTHONNOUSERSITE=1
export PATH="/opt/conda/bin:${PATH}"
# shellcheck source=/dev/null
source /opt/conda/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# --- blip env (train / eval / convert) ---
if ! conda env list | grep -q '^blip '; then
  conda env create -f environment.yaml
fi
conda activate blip

bash scripts/gcp/dataset_minimal.sh

# --- vqascore env (filter only; t2v-metrics needs Py3.10+) ---
if ! conda env list | grep -q '^vqascore '; then
  conda create -n vqascore python=3.10 -y
fi
conda activate vqascore
pip install -q --upgrade pip
pip install -q torch torchvision t2v-metrics omegaconf hydra-core pillow tqdm numpy
conda deactivate
conda activate blip

SYNTH_IN="datasets/aokvqa/synthetic_data_raw.json"
SYNTH_OUT="datasets/aokvqa/synthetic_data_vqascore.json"
REPORT="datasets/aokvqa/filter_report_vqascore.json"
OUT_TRAIN="cache/self_trained_weights_vqascore"

echo "=== Step 1: Filter (vqascore gate only) ==="
conda run -n vqascore python filter_pseudo.py \
  --config configs/filter_pseudo_vqascore_only.yaml \
  --overrides \
    input="${SYNTH_IN}" \
    output="${SYNTH_OUT}" \
    report="${REPORT}" \
    image_root=datasets/coco2017 \
    device=cuda

echo "=== Step 2: Train student ==="
python -m torch.distributed.run --nproc_per_node=1 train_vqa.py \
  --output_dir="${OUT_TRAIN}" \
  --config configs/aokvqa.yaml \
  --overrides \
    "train_files=[train,synthetic_data_vqascore]" \
    truncate_train_dataset_to=34000 \
    wandb=false \
    use_validation_set_as_test_set=true

echo "=== Step 3: Evaluate ==="
CKPT=$(ls -1 "${OUT_TRAIN}"/checkpoint_*.pth 2>/dev/null | sort -V | tail -1)
python -m torch.distributed.run --nproc_per_node=1 train_vqa.py \
  --output_dir=cache/evals_vqascore \
  --evaluate \
  --config configs/aokvqa.yaml \
  --overrides \
    pretrained="${CKPT}" \
    use_validation_set_as_test_set=true \
    wandb=false

echo "=== Done $(date -Is) ==="
echo "Filtered synthetic: ${SYNTH_OUT}"
echo "Checkpoint: ${CKPT}"
echo "Eval result: cache/evals_vqascore/result/vqa_result.json"
