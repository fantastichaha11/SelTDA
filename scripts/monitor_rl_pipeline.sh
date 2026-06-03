#!/usr/bin/env bash
# Status: generate (RL teacher) -> train student (raw synthetic) -> eval
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEN_LOG="${ROOT}/logs/rl_round1_generate_17k_bs64.log"
if [[ ! -f "${GEN_LOG}" ]]; then
  GEN_LOG="${ROOT}/logs/rl_round1_generate_17k.log"
fi
PIPE_LOG="${ROOT}/logs/rl_round1_pipeline.log"
RAW="${ROOT}/datasets/aokvqa/synthetic_data_rl_raw.json"
FILT="${ROOT}/datasets/aokvqa/synthetic_data_rl.json"
STUDENT_DIR="${ROOT}/cache/rl_round1_student"
EVAL_JSON="${ROOT}/orchestration/state/round_1.json"
TEACHER="${ROOT}/orchestration/state/teacher_1.pth"

echo "=== Pipeline status $(date -Is) ==="
echo "Teacher: ${TEACHER}"
echo ""

# --- 1. Generate ---
echo "## 1. Generate (17k unlabeled)"
if pgrep -f "generate_questions_rl_round1" >/dev/null; then
  echo "- Status: RUNNING"
else
  echo "- Status: stopped"
fi
if [[ -f "${GEN_LOG}" ]]; then
  prog=$(grep -oE '[0-9]+/1062' "${GEN_LOG}" 2>/dev/null | tail -1 || true)
  parsed=$(grep 'Sucessfully parsed' "${GEN_LOG}" 2>/dev/null | tail -1 || true)
  [[ -n "${prog}" ]] && echo "- Progress: batch ${prog}"
  [[ -n "${parsed}" ]] && echo "- ${parsed}"
fi
if [[ -f "${RAW}" ]]; then
  sz=$(du -h "${RAW}" | cut -f1)
  n=$(python3 -c "import json; print(len(json.load(open('${RAW}'))))" 2>/dev/null || echo "?")
  echo "- Output: ${RAW} (${sz}, ${n} records)"
else
  echo "- Output: not written yet"
fi
echo ""

# --- 2. Synthetic for train (raw, no filter) ---
echo "## 2. Synthetic train file (unfiltered raw)"
if [[ -f "${FILT}" ]]; then
  n=$(python3 -c "import json; print(len(json.load(open('${FILT}'))))" 2>/dev/null || echo "?")
  echo "- ${FILT} (${n} records)"
elif [[ -f "${RAW}" ]]; then
  echo "- ${FILT} missing; raw ready (${RAW})"
else
  echo "- not ready"
fi
echo ""

# --- 3. Train student ---
echo "## 3. Train student (train + synthetic_data_rl)"
if pgrep -f "train_vqa.py.*rl_round1_student" >/dev/null || pgrep -f "train_vqa.py" >/dev/null 2>&1; then
  if pgrep -f "train_vqa.py" >/dev/null; then
    echo "- Status: RUNNING (train_vqa)"
  fi
else
  echo "- Status: not running"
fi
if [[ -d "${STUDENT_DIR}" ]]; then
  ls -1 "${STUDENT_DIR}"/checkpoint_*.pth 2>/dev/null | tail -3 | sed 's/^/- ckpt: /' || echo "- ckpt: none yet"
  grep -E "epoch|loss" "${STUDENT_DIR}/log.txt" 2>/dev/null | tail -2 | sed 's/^/  /' || true
else
  echo "- Output dir: not created"
fi
echo ""

# --- 4. Eval ---
echo "## 4. Eval (A-OKVQA val MC)"
if [[ -f "${EVAL_JSON}" ]]; then
  cat "${EVAL_JSON}"
else
  echo "- round_1.json: not yet"
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  echo ""
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader | \
    awk -F', ' '{print "GPU: " $1 " util, " $2 "/" $3}'
fi
