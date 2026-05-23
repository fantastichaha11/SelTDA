#!/bin/bash

set -e

ENV_NAME="vqa"
PYTHON_VERSION="3.10"

init_conda() {
    if command -v conda &>/dev/null; then
        # Shell function wrapper — need real base path for conda.sh
        eval "$(conda shell.bash hook 2>/dev/null)" && return 0
    fi

    local candidates=(
        "${CONDA_PREFIX}/etc/profile.d/conda.sh"
        "${HOME}/miniconda3/etc/profile.d/conda.sh"
        "${HOME}/anaconda3/etc/profile.d/conda.sh"
        "/opt/conda/etc/profile.d/conda.sh"
        "/usr/local/miniconda3/etc/profile.d/conda.sh"
    )

    for candidate in "${candidates[@]}"; do
        if [ -f "${candidate}" ]; then
            # shellcheck source=/dev/null
            source "${candidate}"
            return 0
        fi
    done

    echo "ERROR: conda not found."
    echo ""
    echo "Install Miniconda, then re-run this script:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh"
    echo "  bash /tmp/miniconda.sh -b -p \$HOME/miniconda3"
    echo "  \$HOME/miniconda3/bin/conda init bash"
    echo "  source ~/.bashrc"
    echo "  bash setup.sh"
    exit 1
}

init_conda

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Conda environment '${ENV_NAME}' already exists — reusing it."
else
    echo "Creating conda environment: ${ENV_NAME}"
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

echo "Activating environment ${ENV_NAME}"
conda activate "${ENV_NAME}"

# Tránh pip lấy package từ ~/.local (gây xung đột torch/transformers)
export PYTHONNOUSERSITE=1

echo "Installing PyTorch + torchvision (CUDA 12.6)"

pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu126

echo "Installing requirements"

pip install -r requirements.txt

echo "Done!"
echo "Activate with:"
echo "  source \"\$(conda info --base)/etc/profile.d/conda.sh\""
echo "  conda activate ${ENV_NAME}"
echo "Recommended (tránh leak ~/.local):"
echo "  export PYTHONNOUSERSITE=1"
