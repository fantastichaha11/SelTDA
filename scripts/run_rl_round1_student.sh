#!/usr/bin/env bash
# RL round 1: generate (RL teacher) -> train student (train + synthetic raw) -> eval
# No filter_pseudo: GRPO reward already curates teacher outputs.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export PYTHONNOUSERSITE=1
# expandable_segments breaks generate_questions (cublasLtCreate); enable only for train if needed
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COCO_DIR="${ROOT}/datasets/coco2017"
COCO_UNLABELED="${COCO_DIR}/unlabeled2017"
AOKVQA_DIR="${ROOT}/datasets/aokvqa"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/orchestration/state/teacher_1.pth}"
# BLIP base init (not published SelTDA student / not --resume from output_dir)
STUDENT_INIT="${STUDENT_INIT:-https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth}"
OUTPUT_DIR="${ROOT}/cache/rl_round1_student"
FRESH_TRAIN="${FRESH_TRAIN:-0}"
EVAL_DIR="${ROOT}/cache/rl_round1_evals"
MED_CONFIG="${ROOT}/configs/med_config.json"
LOG="${ROOT}/logs/rl_round1_pipeline.log"
RAW="${AOKVQA_DIR}/synthetic_data_rl_raw.json"
FILTERED="${AOKVQA_DIR}/synthetic_data_rl.json"
TRUNCATE_GENERATE="${TRUNCATE_GENERATE:-17000}"
NUM_GPUS="${NUM_GPUS:-1}"
SKIP_GENERATE="${SKIP_GENERATE:-0}"
SKIP_FILTER="${SKIP_FILTER:-1}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

mkdir -p logs "${OUTPUT_DIR}" "${EVAL_DIR}"

log() { echo "[$(date -Is)] $*" | tee -a "${LOG}"; }

log "========== RL Round 1: student pipeline =========="

if [[ ! -f "${TEACHER_CKPT}" ]]; then
  echo "Missing RL teacher: ${TEACHER_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${AOKVQA_DIR}/train.json" ]]; then
  log "Converting A-OKVQA annotations..."
  python convert_aokvqa.py --config configs/aokvqa.yaml \
    --overrides "vqa_root='${COCO_DIR}'" "ann_root='${AOKVQA_DIR}'"
fi

if [[ "${SKIP_GENERATE}" != "1" ]]; then
  log "Generate pseudo-QA from unlabeled (teacher=${TEACHER_CKPT})"
  python generate_questions.py \
    --config configs/generate_questions_rl_round1.yaml \
    --output_dir "${ROOT}/cache/generate_rl_round1" \
    --overrides \
      "pretrained='${TEACHER_CKPT}'" \
      "truncate_to=${TRUNCATE_GENERATE}" \
    2>&1 | tee -a "${LOG}"
else
  log "SKIP_GENERATE=1"
fi

if [[ ! -f "${RAW}" ]]; then
  echo "Missing ${RAW}" >&2
  exit 1
fi

if [[ "${SKIP_FILTER}" != "1" ]]; then
  log "Filter pseudo-labels -> ${FILTERED}"
  python filter_pseudo.py \
    --config configs/filter_pseudo.yaml \
    --output_dir "${ROOT}/cache/filter_rl_round1" \
    --overrides \
      "input='${RAW}'" \
      "image_root='${COCO_DIR}'" \
      "output='${FILTERED}'" \
      "report='${AOKVQA_DIR}/filter_report_rl.json'" \
      "gates.conf.enabled=true" \
      "gates.conf.keep_top=0.75" \
      "gates.itm.enabled=true" \
      "gates.itm.keep_top=0.75" \
      "gates.xcons.enabled=true" \
      "gates.xcons.keep_top=0.75" \
      "gates.xcons.student_ckpt='${ROOT}/cache/student_weights/student_base_rl.pth'" \
      "torch_home=${ROOT}/cache/torch_home" \
    2>&1 | tee -a "${LOG}"
else
  log "SKIP_FILTER=1 — use raw synthetic as train file"
  if [[ ! -f "${FILTERED}" ]]; then
    cp -f "${RAW}" "${FILTERED}"
    log "Copied ${RAW} -> ${FILTERED}"
  fi
fi

if [[ ! -f "${FILTERED}" ]]; then
  echo "Missing ${FILTERED}" >&2
  exit 1
fi

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  if [[ "${FRESH_TRAIN}" == "1" ]]; then
    log "FRESH_TRAIN=1 — removing old checkpoints in ${OUTPUT_DIR}"
    rm -f "${OUTPUT_DIR}"/checkpoint_*.pth
  fi
  log "Train student: train + synthetic_data_rl -> ${OUTPUT_DIR} (init=${STUDENT_INIT})"
  TRAIN_OVERRIDES=(
    "vqa_root='${COCO_DIR}'"
    "ann_root='${AOKVQA_DIR}'"
    "train_files=[train,synthetic_data_rl]"
    "truncate_train_dataset_to=34000"
    "batch_size_train=${BATCH_SIZE_TRAIN:-12}"
    "pretrained='${STUDENT_INIT}'"
    "wandb=false"
    "torch_home=${ROOT}/cache/torch_home"
  )
  if [[ "${NUM_GPUS}" -le 1 ]]; then
    python train_vqa.py \
      --output_dir="${OUTPUT_DIR}" \
      --config configs/aokvqa.yaml \
      --overrides "${TRAIN_OVERRIDES[@]}" \
      2>&1 | tee -a "${LOG}"
  else
    python -m torch.distributed.run --nproc_per_node="${NUM_GPUS}" train_vqa.py \
      --output_dir="${OUTPUT_DIR}" \
      --config configs/aokvqa.yaml \
      --overrides "${TRAIN_OVERRIDES[@]}" \
      2>&1 | tee -a "${LOG}"
  fi
else
  log "SKIP_TRAIN=1"
fi

if [[ "${SKIP_EVAL}" != "1" ]]; then
  CKPT="${OUTPUT_DIR}/checkpoint_09.pth"
  if [[ ! -f "${CKPT}" ]]; then
    CKPT="$(ls -1 "${OUTPUT_DIR}"/checkpoint_*.pth 2>/dev/null | sort -V | tail -1)"
  fi
  if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
    echo "No student checkpoint in ${OUTPUT_DIR}" >&2
    exit 1
  fi
  log "Eval A-OKVQA val MC: ${CKPT}"
  python scripts/aokvqa_mc_eval.py "${CKPT}" --config configs/aokvqa.yaml 2>&1 | tee -a "${LOG}"
  python - <<PY
import json
import re
from pathlib import Path
log = Path("${LOG}").read_text()
m = re.findall(r"A-OKVQA val MC accuracy: ([\d.]+)%", log)
acc = float(m[-1]) if m else 0.0
Path("orchestration/state/round_1.json").write_text(
    json.dumps(
        {
            "round": 1,
            "teacher_ckpt": "orchestration/state/teacher_1.pth",
            "student_ckpt": "${CKPT}",
            "synthetic": "synthetic_data_rl.json",
            "val_mc_accuracy_pct": acc,
        },
        indent=2,
    )
)
print(f"Wrote round_1.json acc={acc}%")
PY
  log "Round summary: orchestration/state/round_1.json"
else
  log "SKIP_EVAL=1"
fi

log "========== Pipeline finished =========="
