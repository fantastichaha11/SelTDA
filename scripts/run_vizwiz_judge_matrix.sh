#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_ROOT}"

DATASET="vizwiz"
CONDITION="${1:?usage: $0 rubric_probe|filter4|pt4_j|pt8_jp|pt4_jp_60k}"
EXP_ROOT="${EXP_ROOT:-outputs/experiments/${DATASET}/${CONDITION}}"
RAW_DIR="${EXP_ROOT}/raw"
DATA_DIR="${EXP_ROOT}/data"
STUDENT_DIR="${EXP_ROOT}/student"
IMAGE_ROOT="datasets/vizwiz/images"
TRAIN_JSON="datasets/vizwiz/train.json"
VAL_JSON="datasets/vizwiz/val.json"
TEST_JSON="datasets/vizwiz/test.json"
TEST_METADATA="datasets/vizwiz/vizwiz_test_metadata.json"
SEED=42
TARGET_TOTAL=40000
if [[ "${CONDITION}" == "pt4_jp_60k" ]]; then TARGET_TOTAL=60000; fi

timestamp_utc() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

ACTION_START_EPOCH="$(date +%s)"
ACTION_START_TIME="$(timestamp_utc)"

record_command() {
  local status="$1"
  local started="$2"
  local ended="$3"
  shift 3
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return
  fi
  mkdir -p "${EXP_ROOT}"
  python - "${EXP_ROOT}/commands.jsonl" "${status}" "${started}" "${ended}" "$@" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
entry = {
    "argv": sys.argv[5:],
    "start_time": sys.argv[3],
    "end_time": sys.argv[4],
    "return_code": int(sys.argv[2]),
    "resume_decision": "run",
}
with path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
PY
}

write_compute() {
  local status="$1"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return
  fi
  local end_epoch
  local end_time
  local elapsed
  local gpu_name
  local gpu_memory
  end_epoch="$(date +%s)"
  end_time="$(timestamp_utc)"
  elapsed=$((end_epoch - ACTION_START_EPOCH))
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 || true)"
  gpu_memory="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -nr | head -n 1 || true)"
  mkdir -p "${EXP_ROOT}"
  python - \
    "${EXP_ROOT}/compute.json" \
    "${DATASET}" "${CONDITION}" "${TARGET_TOTAL}" "${status}" \
    "${ACTION_START_TIME}" "${end_time}" "${elapsed}" \
    "${gpu_name}" "${gpu_memory}" \
    "${TRAIN_JSON}" "${TEST_JSON}" "${VAL_JSON}" "${TEST_METADATA}" \
    "${PT4_JP_TEACHER_CKPT:-}" "${PT4_JP_SYNTHETIC_40K:-}" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

(
    output,
    dataset,
    condition,
    target_total,
    status,
    start_time,
    end_time,
    elapsed,
    gpu_name,
    gpu_memory,
    train_json,
    test_json,
    val_json,
    test_metadata,
    pt4_jp_teacher,
    pt4_jp_synthetic,
) = sys.argv[1:17]

def file_sha256(path_text: str):
    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

target_total_int = int(target_total)
student_epochs = 0 if condition == "rubric_probe" else 10
student_batch_size = 16
optimizer_steps = 0 if student_epochs == 0 else math.ceil(target_total_int / student_batch_size) * student_epochs
memory_value = int(gpu_memory) if gpu_memory.strip().isdigit() else None
payload = {
    "dataset": dataset,
    "condition": condition,
    "seed": 42,
    "target_total_examples": 0 if condition == "rubric_probe" else target_total_int,
    "student_epochs": student_epochs,
    "student_batch_size": student_batch_size,
    "optimizer_steps_estimate": optimizer_steps,
    "status": "success" if int(status) == 0 else "failed",
    "return_code": int(status),
    "start_time": start_time,
    "end_time": end_time,
    "elapsed_seconds": int(elapsed),
    "gpu_name": gpu_name or None,
    "peak_gpu_memory_mb": None,
    "gpu_memory_mb_at_finish": memory_value,
    "input_hashes": {
        "train_json": file_sha256(train_json),
        "test_json": file_sha256(test_json),
        "val_json": file_sha256(val_json),
        "test_metadata": file_sha256(test_metadata),
        "pt4_jp_teacher_ckpt": file_sha256(pt4_jp_teacher),
        "pt4_jp_synthetic_40k": file_sha256(pt4_jp_synthetic),
    },
}
Path(output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

on_exit() {
  local status=$?
  write_compute "${status}"
}
trap on_exit EXIT

run() {
  printf '+ '; printf '%q ' "$@"; printf '\n'
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    local started
    local ended
    local status
    started="$(timestamp_utc)"
    set +e
    "$@"
    status=$?
    set -e
    ended="$(timestamp_utc)"
    record_command "${status}" "${started}" "${ended}" "$@"
    if [[ "${status}" -ne 0 ]]; then
      return "${status}"
    fi
  fi
}

require_file() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '[dry-run] require file %s\n' "$1"
    return
  fi
  test -f "$1"
}

require_dir() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '[dry-run] require dir %s\n' "$1"
    return
  fi
  test -d "$1"
}

prepare_dirs() {
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    mkdir -p "${RAW_DIR}" "${DATA_DIR}" "${STUDENT_DIR}"
  fi
}

write_unique_images() {
  local annotations="$1"
  local output="$2"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '[dry-run] write unique image manifest %s from %s\n' "${output}" "${annotations}"
    return
  fi
  python - "${annotations}" "${output}" <<'PY'
import json
import sys
from pathlib import Path
source, output = map(Path, sys.argv[1:3])
seen = set()
rows = []
for row in json.loads(source.read_text(encoding="utf-8")):
    image = row["image"]
    if image in seen:
        continue
    seen.add(image)
    rows.append({"image": image})
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
PY
}

write_probe_images_from_manifest() {
  local manifest="$1"
  local output="$2"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '[dry-run] write probe generation images %s from %s\n' "${output}" "${manifest}"
    return
  fi
  python - "${manifest}" "${output}" <<'PY'
import json
import sys
from pathlib import Path
manifest, output = map(Path, sys.argv[1:3])
payload = json.loads(manifest.read_text(encoding="utf-8"))
rows = [{"image": row["image"]} for row in payload["selected_images"]]
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
PY
}

annotate_probe_groups() {
  local pairs_json="$1"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '[dry-run] annotate probe group_id/candidate_index in %s\n' "${pairs_json}"
    return
  fi
  python - "${pairs_json}" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path
path = Path(sys.argv[1])
rows = json.loads(path.read_text(encoding="utf-8"))
counters = defaultdict(int)
for row in rows:
    image = row["image"]
    row["group_id"] = image
    row["candidate_index"] = counters[image]
    counters[image] += 1
path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
PY
}

filter_questions_per_image() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '2'
    return
  fi
  python - "${TRAIN_JSON}" "${TARGET_TOTAL}" <<'PY'
import json
import math
import sys
from pathlib import Path
records = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
target = int(sys.argv[2])
real = len(records)
unique_images = len({row["image"] for row in records})
quota = target - real
print(math.ceil((2 * quota) / unique_images))
PY
}

generate_pool() {
  local teacher_ckpt="$1"
  local questions_per_image="$2"
  local annotations="$3"
  local output_name="${4:-generated_qa.json}"
  run python generate_questions.py \
    --config configs/generate_questions_pathvqa.yaml \
    --output_dir="${RAW_DIR}" \
    --overrides \
      "image_folder=${IMAGE_ROOT}" \
      "output_folder=${RAW_DIR}" \
      "annotations=${annotations}" \
      "pretrained=${teacher_ckpt}" \
      "output_annotations_name=${output_name}" \
      "multimodal_encoder_decoder_config=configs/med_config.json" \
      "torch_home=null" \
      "batch_size=32" \
      "questions_per_image=${questions_per_image}" \
      "max_length=40" \
      "min_length=5" \
      "top_p=0.9" \
      "shuffle=false"
}

write_build_config() {
  local output_json="$1"
  local source_json="$2"
  local base_json="${3:-}"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '[dry-run] write build config %s/build_config.yaml\n' "${DATA_DIR}"
    return
  fi
  mkdir -p "${DATA_DIR}"
  {
    printf 'real_annotations: %s\n' "${TRAIN_JSON}"
    printf 'source_pools:\n'
    printf '  - %s\n' "${source_json}"
    if [[ -n "${base_json}" ]]; then
      printf 'base_synthetic: %s\n' "${base_json}"
    fi
    printf 'image_root: %s\n' "${IMAGE_ROOT}"
    printf 'dataset: %s\n' "${DATASET}"
    printf 'target_total: %s\n' "${TARGET_TOTAL}"
    printf 'output: %s\n' "${output_json}"
    printf 'manifest: %s\n' "${DATA_DIR}/manifest.json"
    printf 'question_id_start: 10000000\n'
  } > "${DATA_DIR}/build_config.yaml"
}

train_student() {
  local synthetic_name="$1"
  local wandb_name="$2"
  run python -m torch.distributed.run \
    --master_port="${MASTER_PORT:-37813}" \
    --nproc_per_node="${NUM_GPUS:-1}" \
    train_vqa.py \
    --output_dir="${STUDENT_DIR}" \
    --config configs/vizwiz.yaml \
    --no-resume \
    --overrides \
      "vqa_root='${IMAGE_ROOT}'" \
      "ann_root='datasets/vizwiz'" \
      "train_files=[train,${synthetic_name}]" \
      "truncate_train_dataset_to=${TARGET_TOTAL}" \
      "batch_size_train=16" \
      "batch_size_test=1" \
      "max_epoch=10" \
      "torch_home=null" \
      "wandb=false" \
      "wandb_mode=disabled" \
      "wandb_name='${wandb_name}'"
}

evaluate_student() {
  run python vizwiz_eval.py \
    "${STUDENT_DIR}/result/vqa_result.json" \
    --annotation-file "${TEST_JSON}" \
    --metadata-file "${TEST_METADATA}" \
    --wandb-mode disabled
}

preflight_common() {
  require_file "${TRAIN_JSON}"
  require_file "${TEST_JSON}"
  require_file "${TEST_METADATA}"
  require_dir "${IMAGE_ROOT}"
}

run_rubric_probe() {
  prepare_dirs
  require_file "${VAL_JSON}"
  require_file cache/vizwiz_teacher_weights/vizwiz_teacher_checkpoint_04.pth
  run python scripts/diagnose_rubrics.py prepare \
    --validation-annotations "${VAL_JSON}" \
    --image-root "${IMAGE_ROOT}" \
    --teacher-checkpoint cache/vizwiz_teacher_weights/vizwiz_teacher_checkpoint_04.pth \
    --output "${EXP_ROOT}/probe_images.json" \
    --image-count 25 --seed "${SEED}"
  write_probe_images_from_manifest "${EXP_ROOT}/probe_images.json" "${RAW_DIR}/probe_generation_images.json"
  generate_pool cache/vizwiz_teacher_weights/vizwiz_teacher_checkpoint_04.pth 8 "${RAW_DIR}/probe_generation_images.json"
  annotate_probe_groups "${RAW_DIR}/generated_qa.json"
  run python scripts/diagnose_rubrics.py score \
    --pairs-json "${RAW_DIR}/generated_qa.json" \
    --image-root "${IMAGE_ROOT}" \
    --rubric4-config configs/prometheus_judge_vizwiz.yaml \
    --rubric8-config configs/prometheus_judge_vizwiz_rubric8.yaml \
    --selected-model-path cache/prometheus-vision-7b-v1.0 \
    --output-dir "${EXP_ROOT}" \
    --expected-groups 25 --expected-group-size 8 --seed "${SEED}"
}

run_filter4() {
  prepare_dirs
  preflight_common
  require_file cache/vizwiz_teacher_weights/vizwiz_teacher_checkpoint_04.pth
  write_unique_images "${TRAIN_JSON}" "${RAW_DIR}/train_unique_images.json"
  local qpi
  qpi="$(filter_questions_per_image)"
  generate_pool cache/vizwiz_teacher_weights/vizwiz_teacher_checkpoint_04.pth "${qpi}" "${RAW_DIR}/train_unique_images.json"
  run python scripts/score_rank_prometheus_pool.py --config configs/global_filter_vizwiz_rubric4.yaml
  train_student synthetic_filter4_40k "vizwiz-filter4-40k"
  evaluate_student
}

run_posttrain_40k() {
  local config="$1"
  local teacher_ckpt="$2"
  local synthetic_name="$3"
  prepare_dirs
  preflight_common
  run python scripts/train_teacher_grpo.py --config "${config}"
  write_unique_images "${TRAIN_JSON}" "${RAW_DIR}/train_unique_images.json"
  generate_pool "${teacher_ckpt}" 2 "${RAW_DIR}/train_unique_images.json"
  write_build_config "datasets/vizwiz/${synthetic_name}.json" "${RAW_DIR}/generated_qa.json"
  run python scripts/build_student_synthetic.py --config "${DATA_DIR}/build_config.yaml"
  run python scripts/build_student_synthetic.py --verify-manifest "${DATA_DIR}/manifest.json"
  train_student "${synthetic_name}" "vizwiz-${CONDITION}-40k"
  evaluate_student
}

run_60k() {
  prepare_dirs
  preflight_common
  PT4_JP_TEACHER_CKPT="${PT4_JP_TEACHER_CKPT:-outputs/grpo_teacher/vizwiz_teacher_200x3epochs/checkpoint_step_000600.pth}"
  PT4_JP_SYNTHETIC_40K="${PT4_JP_SYNTHETIC_40K:?set PT4_JP_SYNTHETIC_40K to the exact existing VizWiz Full-40K synthetic JSON}"
  require_file "${PT4_JP_TEACHER_CKPT}"
  require_file "${PT4_JP_SYNTHETIC_40K}"
  write_unique_images "${TRAIN_JSON}" "${RAW_DIR}/train_unique_images.json"
  generate_pool "${PT4_JP_TEACHER_CKPT}" 2 "${RAW_DIR}/train_unique_images.json" extra_generated_qa.json
  write_build_config "datasets/vizwiz/synthetic_pt4_jp_60k.json" "${RAW_DIR}/extra_generated_qa.json" "${PT4_JP_SYNTHETIC_40K}"
  run python scripts/build_student_synthetic.py --config "${DATA_DIR}/build_config.yaml"
  run python scripts/build_student_synthetic.py --verify-manifest "${DATA_DIR}/manifest.json"
  train_student synthetic_pt4_jp_60k "vizwiz-pt4-jp-60k"
  evaluate_student
}

case "${CONDITION}" in
  rubric_probe) run_rubric_probe ;;
  filter4) run_filter4 ;;
  pt4_j) run_posttrain_40k configs/grpo_teacher_vizwiz_rubric4_judge_only.yaml "${EXP_ROOT}/teacher/checkpoint_step_000600.pth" synthetic_pt4_j_40k ;;
  pt8_jp) run_posttrain_40k configs/grpo_teacher_vizwiz_rubric8_penalties.yaml "${EXP_ROOT}/teacher/checkpoint_step_000600.pth" synthetic_pt8_jp_40k ;;
  pt4_jp_60k) run_60k ;;
  *) printf 'unknown condition: %s\n' "${CONDITION}" >&2; exit 2 ;;
esac
