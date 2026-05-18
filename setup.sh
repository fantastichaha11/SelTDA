#!/bin/bash

set -e

ENV_NAME="vqa"
PYTHON_VERSION="3.10"

echo "Creating conda environment: $ENV_NAME"

conda create -y -n $ENV_NAME python=$PYTHON_VERSION

echo "Activating environment"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "Installing PyTorch + torchvision (CUDA 12.6)"

pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu126

echo "Installing requirements"

pip install -r requirements.txt

echo "Done!"
echo "Activate with:"
echo "conda activate $ENV_NAME"