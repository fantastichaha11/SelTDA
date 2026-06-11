#!/bin/bash
# Minimal blip conda env for GCP (avoids clip==1.0 PyPI failure in environment.yaml).
set -euo pipefail

export PATH="/opt/conda/bin:${PATH}"
source /opt/conda/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

if conda env list | grep -q '^blip '; then
  echo "[skip] blip env exists"
  exit 0
fi

conda create -n blip python=3.8 -y
conda activate blip
pip install --upgrade pip wheel setuptools

# PyTorch cu118 (L4 / CUDA 12 driver compatible)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

pip install \
  omegaconf==2.2.2 hydra-core==1.2.0 attrs==22.1.0 cattrs==22.1.0 \
  numpy==1.23.1 opencv-python==4.6.0.66 timm==0.4.12 fairscale==0.4.4 \
  transformers==4.21.3 huggingface_hub==0.13.4 open_clip_torch==2.20.0 \
  sentence-transformers==2.2.2 tqdm pillow gdown einops

pip install git+https://github.com/openai/CLIP.git

echo "blip env ready"
