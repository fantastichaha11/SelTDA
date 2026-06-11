#!/bin/bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git tmux aria2 unzip curl wget

USER_HOME="/home/$(logname 2>/dev/null || echo ubuntu)"
export PATH="/opt/conda/bin:${PATH}"

if [ ! -d /opt/conda ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p /opt/conda
fi
/opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
/opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

REPO_DIR="${USER_HOME}/SelTDA"
BRANCH="feat/pseudo-label-filter"
REPO_URL="https://github.com/fantastichaha11/SelTDA.git"

if [ ! -d "${REPO_DIR}/.git" ]; then
  sudo -u "$(logname 2>/dev/null || echo ubuntu)" git clone -b "${BRANCH}" "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"
sudo -u "$(logname 2>/dev/null || echo ubuntu)" git fetch origin "${BRANCH}"
sudo -u "$(logname 2>/dev/null || echo ubuntu)" git checkout "${BRANCH}"
sudo -u "$(logname 2>/dev/null || echo ubuntu)" git pull --ff-only origin "${BRANCH}" || true
chmod +x scripts/gcp/*.sh

RUN_USER="$(logname 2>/dev/null || echo ubuntu)"
sudo -u "${RUN_USER}" tmux new-session -d -s vqascore \
  "export PATH=/opt/conda/bin:\$PATH && source /opt/conda/etc/profile.d/conda.sh && cd ${REPO_DIR} && bash scripts/gcp/run_vqascore_pipeline.sh; echo EXIT=\$? | tee -a cache/logs/vqascore_run/exit_code.txt; exec bash"

echo "VQAScore pipeline started in tmux session 'vqascore'" | tee /var/log/seltda-vqascore-startup.log
