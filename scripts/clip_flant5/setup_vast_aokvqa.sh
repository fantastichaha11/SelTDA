#!/usr/bin/env bash
# Setup CLIP-FlanT5-XL stage-2 fine-tune on A-OKVQA (Vast.ai / multi-GPU).
# Run from SelTDA repo root on a machine with 4-8× A100 40G (recommended).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

SELTDA="${SELTDA:-${ROOT}}"
CLIP_ROOT="${CLIP_ROOT:-${SELTDA}/external/CLIP-FlanT5}"
PLAYGROUND="${PLAYGROUND:-${CLIP_ROOT}/playground/data}"
STAGE1_CKPT="${STAGE1_CKPT:-${CLIP_ROOT}/checkpoints/clip-flant5-xl-stage-1/mm_projector.bin}"
DATA_JSON="${DATA_JSON:-${SELTDA}/datasets/clip_flant5/aokvqa_vqa.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${CLIP_ROOT}/checkpoints/clip-flant5-xl-aokvqa}"

log() { echo "[$(date -Is)] $*"; }

log "========== CLIP-FlanT5 XL A-OKVQA setup =========="
log "SelTDA=${SELTDA}"
log "CLIP_ROOT=${CLIP_ROOT}"

mkdir -p "${CLIP_ROOT}" "${PLAYGROUND}/coco"

if [[ ! -d "${CLIP_ROOT}/.git" ]]; then
  log "Cloning CLIP-FlanT5..."
  git clone https://github.com/linzhiqiu/CLIP-FlanT5.git "${CLIP_ROOT}"
fi

# Symlink COCO images → playground/data/coco/train2017/
if [[ ! -e "${PLAYGROUND}/coco/train2017" ]]; then
  ln -sfn "${SELTDA}/datasets/coco2017" "${PLAYGROUND}/coco/train2017"
  log "Linked ${PLAYGROUND}/coco/train2017 -> ${SELTDA}/datasets/coco2017"
fi

log "Converting A-OKVQA train+val to LLaVA JSON..."
export PYTHONNOUSERSITE=1
python "${SELTDA}/scripts/convert_aokvqa_to_clip_flant5.py" \
  --config "${SELTDA}/configs/convert_clip_flant5.yaml" \
  --overrides \
    mode=vqa \
    output="${DATA_JSON}"

if [[ ! -f "${STAGE1_CKPT}" ]]; then
  log "Downloading stage-1 projector (clip-flant5-xl-stage-1)..."
  mkdir -p "$(dirname "${STAGE1_CKPT}")"
  python - <<PY
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="zhiqiulin/clip-flant5-xl-stage-1",
    filename="mm_projector.bin",
    local_dir="$(dirname "${STAGE1_CKPT}")",
)
print("Downloaded:", path)
PY
fi

log "Install CLIP-FlanT5 deps (once per machine)..."
if [[ ! -f "${CLIP_ROOT}/.deps_installed" ]]; then
  pip install -e "${CLIP_ROOT}" 2>/dev/null || pip install "${CLIP_ROOT}/." || true
  pip install deepspeed wandb accelerate 2>/dev/null || true
  touch "${CLIP_ROOT}/.deps_installed"
fi

log "========== Setup done =========="
log "Data: ${DATA_JSON}"
log "Images: ${PLAYGROUND}/coco/train2017"
log "Train with: bash ${SELTDA}/scripts/clip_flant5/train_xl_aokvqa.sh"
