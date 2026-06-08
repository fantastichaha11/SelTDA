#!/usr/bin/env bash
# DEPRECATED: uploads as versions under instance "default".
# Prefer: scripts/upload_kaggle_model_variants.sh (one Kaggle instance per checkpoint).
#
# Upload SelTDA checkpoints as Kaggle Model instance versions.
#
# Prereqs: ~/.kaggle/kaggle.json (Legacy API, username must match owner phong2004)
# Models: phong2004/seltda-teacher-vqg, phong2004/seltda-student-vqa
# Instance handle: <owner>/<model>/pytorch/default
#
# Usage:
#   bash scripts/upload_kaggle_model_versions.sh              # all versions
#   bash scripts/upload_kaggle_model_versions.sh --dry-run
#   MODEL=teacher bash scripts/upload_kaggle_model_versions.sh
#   ONLY=published-epoch04 bash scripts/upload_kaggle_model_versions.sh
#   SKIP_EXISTING=1 bash scripts/upload_kaggle_model_versions.sh
#
set -eu

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${ROOT}/scripts/kaggle_model_versions.manifest"
STAGING_ROOT="${ROOT}/kaggle_models/upload-staging"
OWNER="${KAGGLE_OWNER:-phong2004}"
# Kaggle URL uses framework slug from API, e.g. PyTorch (not pytorch)
FRAMEWORK="${KAGGLE_FRAMEWORK:-PyTorch}"
INSTANCE_SLUG="${KAGGLE_INSTANCE:-default}"

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
    -h|--help)
      sed -n '1,18p' "$0"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
  shift
done

if [[ ! -f "${HOME}/.kaggle/kaggle.json" ]]; then
  echo "Missing ~/.kaggle/kaggle.json (Legacy API key)" >&2
  exit 1
fi

model_slug_for() {
  case "$1" in
    teacher) echo "seltda-teacher-vqg" ;;
    student) echo "seltda-student-vqa" ;;
    *) echo "unknown model kind: $1" >&2; exit 2 ;;
  esac
}

instance_handle() {
  local kind="$1"
  echo "${OWNER}/$(model_slug_for "${kind}")/${FRAMEWORK}/${INSTANCE_SLUG}"
}

log() { echo "[$(date -Is)] $*"; }

upload_version() {
  local kind="$1" version_id="$2" relpath="$3" upload_name="$4" notes="$5"
  local src="${ROOT}/${relpath}"
  local stage="${STAGING_ROOT}/${kind}-${version_id}"
  local handle
  handle="$(instance_handle "${kind}")"

  if [[ -n "${MODEL_FILTER}" && "${MODEL_FILTER}" != "${kind}" ]]; then
    return 0
  fi
  if [[ -n "${ONLY}" && "${ONLY}" != "${version_id}" ]]; then
    return 0
  fi
  if echo ",${SKIP_IDS}," | grep -qF ",${version_id},"; then
    log "SKIP id in SKIP_IDS: ${version_id}"
    return 0
  fi
  if [[ ! -f "${src}" ]]; then
    log "SKIP missing file: ${src}"
    return 0
  fi

  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    if kaggle models instances files "${handle}" 2>/dev/null | grep -qF "${upload_name}"; then
      log "SKIP existing on Kaggle: ${handle} (${upload_name})"
      return 0
    fi
  fi

  rm -rf "${stage}"
  mkdir -p "${stage}"
  ln -sf "${src}" "${stage}/${upload_name}"

  local full_notes="${version_id}: ${notes}"
  log "UPLOAD ${handle} <- ${relpath} (${upload_name})"

  if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY-RUN: kaggle models instances versions create ${handle} -p ${stage} -n ${full_notes}"
    return 0
  fi

  kaggle models instances versions create "${handle}" \
    -p "${stage}" \
    -n "${full_notes}"
}

log "SelTDA Kaggle model version upload (owner=${OWNER})"

while IFS='|' read -r kind version_id relpath upload_name notes; do
  [[ "${kind}" =~ ^#.*$ ]] && continue
  [[ -z "${kind}" ]] && continue
  upload_version "${kind}" "${version_id}" "${relpath}" "${upload_name}" "${notes}"
done < "${MANIFEST}"

log "Finished."
