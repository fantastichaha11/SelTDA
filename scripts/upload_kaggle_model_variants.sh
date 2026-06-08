#!/usr/bin/env bash
# Upload each SelTDA checkpoint as its own Kaggle Model *instance* (variant).
# URL pattern: phong2004/<model>/PyTorch/<variant-id>  (not versions under default)
#
# Prereqs: ~/.kaggle/kaggle.json; models seltda-teacher-vqg / seltda-student-vqa exist
#
# Usage:
#   bash scripts/upload_kaggle_model_variants.sh
#   bash scripts/upload_kaggle_model_variants.sh --dry-run
#   MODEL=teacher bash scripts/upload_kaggle_model_variants.sh
#   ONLY=published-epoch04 bash scripts/upload_kaggle_model_variants.sh
#   SKIP_EXISTING=1 bash scripts/upload_kaggle_model_variants.sh
#
set -eu

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${ROOT}/scripts/kaggle_model_versions.manifest"
TEMPLATE="${ROOT}/kaggle_models/instance-variant-template/model-instance-metadata.json"
STAGING_ROOT="${ROOT}/kaggle_models/upload-staging-variants"
OWNER="${KAGGLE_OWNER:-phong2004}"
FRAMEWORK="${KAGGLE_FRAMEWORK:-PyTorch}"

DRY_RUN=0
SKIP_EXISTING=0
SKIP_IDS="${SKIP_IDS:-published-epoch04}"
ONLY="${ONLY:-}"
MODEL_FILTER="${MODEL:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --skip-existing) SKIP_EXISTING=1 ;;
    --only) ONLY="$2"; shift ;;
    --model) MODEL_FILTER="$2"; shift ;;
    -h|--help) sed -n '1,16p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

if [[ ! -f "${HOME}/.kaggle/kaggle.json" ]]; then
  echo "Missing ~/.kaggle/kaggle.json" >&2
  exit 1
fi

model_slug_for() {
  case "$1" in
    teacher) echo "seltda-teacher-vqg" ;;
    student) echo "seltda-student-vqa" ;;
    *) echo "unknown model kind: $1" >&2; exit 2 ;;
  esac
}

variant_handle() {
  local kind="$1" variant_id="$2"
  echo "${OWNER}/$(model_slug_for "${kind}")/${FRAMEWORK}/${variant_id}"
}

overview_for() {
  local kind="$1" variant_id="$2" notes="$3"
  if [[ "${kind}" == "teacher" ]]; then
    echo "SelTDA teacher (VQG) variant ${variant_id}. ${notes}"
  else
    echo "SelTDA student (VQA) variant ${variant_id}. ${notes}"
  fi
}

log() { echo "[$(date -Is)] $*"; }

upload_variant() {
  local kind="$1" variant_id="$2" relpath="$3" upload_name="$4" notes="$5"
  local model_slug src stage handle
  model_slug="$(model_slug_for "${kind}")"
  src="${ROOT}/${relpath}"
  stage="${STAGING_ROOT}/${kind}-${variant_id}"
  handle="$(variant_handle "${kind}" "${variant_id}")"

  if [[ -n "${MODEL_FILTER}" && "${MODEL_FILTER}" != "${kind}" ]]; then
    return 0
  fi
  if [[ -n "${ONLY}" && "${ONLY}" != "${variant_id}" ]]; then
    return 0
  fi
  if echo ",${SKIP_IDS}," | grep -qF ",${variant_id},"; then
    log "SKIP id in SKIP_IDS: ${variant_id}"
    return 0
  fi
  if [[ ! -f "${src}" ]]; then
    log "SKIP missing: ${src}"
    return 0
  fi

  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    if kaggle models instances get "${handle}" >/dev/null 2>&1; then
      log "SKIP exists: ${handle}"
      return 0
    fi
  fi

  rm -rf "${stage}"
  mkdir -p "${stage}"
  ln -sf "${src}" "${stage}/${upload_name}"

  local overview
  overview="$(overview_for "${kind}" "${variant_id}" "${notes}")"
  TEMPLATE="${TEMPLATE}" STAGE="${stage}" MODEL_SLUG="${model_slug}" \
    VARIANT_ID="${variant_id}" OVERVIEW="${overview}" python3 <<'PY'
import json
import os
from pathlib import Path

tpl = json.loads(Path(os.environ["TEMPLATE"]).read_text())
tpl["modelSlug"] = os.environ["MODEL_SLUG"]
tpl["instanceSlug"] = os.environ["VARIANT_ID"]
tpl["overview"] = os.environ["OVERVIEW"]
Path(os.environ["STAGE"], "model-instance-metadata.json").write_text(
    json.dumps(tpl, indent=2)
)
PY

  log "VARIANT ${handle} <- ${relpath}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY-RUN: kaggle models instances create -p ${stage}"
    return 0
  fi

  kaggle models instances create -p "${stage}"
}

log "SelTDA Kaggle model variants (one instance per checkpoint)"

while IFS='|' read -r kind variant_id relpath upload_name notes; do
  [[ "${kind}" =~ ^#.*$ ]] && continue
  [[ -z "${kind}" ]] && continue
  upload_variant "${kind}" "${variant_id}" "${relpath}" "${upload_name}" "${notes}"
done < "${MANIFEST}"

log "Finished."
