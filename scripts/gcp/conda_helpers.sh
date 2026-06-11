#!/bin/bash
# Shared conda helpers — use `conda run` (activate is unreliable in bash scripts).
init_conda() {
  export PATH="/opt/conda/bin:${PATH}"
  # shellcheck source=/dev/null
  source /opt/conda/etc/profile.d/conda.sh
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true
}

run_blip() {
  init_conda
  conda run --no-capture-output -n blip "$@"
}

run_vqascore() {
  init_conda
  conda run --no-capture-output -n vqascore "$@"
}

blip_ready() {
  init_conda
  conda env list | grep -q '^blip ' \
    && conda run -n blip python -c "import omegaconf" 2>/dev/null
}
