#!/bin/bash
# Create GPU VM and launch VQAScore-only filter + train + eval pipeline.
set -euo pipefail

export PATH="${HOME}/google-cloud-sdk/bin:${PATH}"

PROJECT="${GCP_PROJECT:-thesis-497813}"
ZONE="${GCP_ZONE:-us-central1-b}"
INSTANCE="${GCP_INSTANCE:-seltda-vqascore}"
BRANCH="${GCP_BRANCH:-feat/pseudo-label-filter}"
REPO="${GCP_REPO:-https://github.com/fantastichaha11/SelTDA.git}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Creating ${INSTANCE} in ${ZONE} (project ${PROJECT})..."
gcloud compute instances create "${INSTANCE}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-balanced \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --maintenance-policy=TERMINATE \
  --scopes=cloud-platform \
  --metadata-from-file=startup-script="${SCRIPT_DIR}/vm_startup_vqascore.sh"

echo ""
echo "VM created. SSH:"
echo "  gcloud compute ssh ${INSTANCE} --zone=${ZONE} --project=${PROJECT}"
echo "Monitor pipeline:"
echo "  gcloud compute ssh ${INSTANCE} --zone=${ZONE} --project=${PROJECT} --command='tmux attach -t vqascore'"
echo "Tail log:"
echo "  gcloud compute ssh ${INSTANCE} --zone=${ZONE} --project=${PROJECT} --command='tail -f ~/SelTDA/cache/logs/vqascore_run/pipeline.log'"
