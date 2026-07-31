#!/usr/bin/env bash
# Enforce the fixed judge-experiment implementation surface.
set -euo pipefail
BASE="${1:-55565c0}"

APPROVED_REGEX='^(configs/(global_filter_(pathvqa|vizwiz)_rubric4|grpo_teacher_(pathvqa|vizwiz)_rubric(4_judge_only|8_penalties)|prometheus_judge_(pathvqa|vizwiz)(_rubric8)?)\.yaml|experiments/(__init__|analysis|synthetic_data)\.py|judge/(__init__|factory)\.py|(pathvqa_eval|vizwiz_eval)\.py|scripts/(analyze_experiment_results|build_student_synthetic|diagnose_rubrics|eval_prometheus_judge|prepare_pathvqa_synthetic|score_rank_prometheus_pool|train_teacher_grpo)\.py|scripts/(check_invariants|run_pathvqa_judge_matrix|run_vizwiz_judge_matrix)\.sh|tests/test_(build_student_synthetic|diagnose_rubrics|experiment_analysis|experiment_configs|experiment_scripts|experiment_synthetic_data|judge_factory|prometheus_judge_scripts|score_rank_prometheus_pool|teacher_grpo)\.py)$'

FORBIDDEN_CHANGED_REGEX='^(train_vqa\.py|train_vqg\.py|models/|data/|dataset_adapters/|vqa_eval_tools/|examples/(self_train_synthetic|evaluate|train_teacher)\.sh|configs/(aokvqa|pathvqa|vizwiz|okvqa|advqa|artvqa|rsvqa|vqa)\.yaml|outputs/|datasets/.*synthetic_.*\.json$|checkpoints/|.*\.ipynb$|.*\.pdf$)'
FORBIDDEN_STAGED_REGEX='^(outputs/|datasets/.*synthetic_.*\.json$|checkpoints/|.*\.ipynb$|.*\.pdf$)'

print_paths() {
    while IFS= read -r path; do
        [ -n "${path}" ] && printf "  %s\n" "${path}"
    done
}

CHANGED=$(git diff --name-only "${BASE}"...HEAD 2>/dev/null || git diff --name-only "${BASE}" HEAD)
echo "Changed vs ${BASE}:"
echo "${CHANGED}"
echo "---"

FORBIDDEN_CHANGED=""
OUT_OF_SCOPE=""
if [ -n "${CHANGED}" ]; then
    FORBIDDEN_CHANGED=$(printf "%s\n" "${CHANGED}" | grep -E "${FORBIDDEN_CHANGED_REGEX}" || true)
    OUT_OF_SCOPE=$(printf "%s\n" "${CHANGED}" | grep -Ev "${APPROVED_REGEX}" || true)
fi

STAGED_RUNTIME=$(git diff --cached --name-only | grep -E "${FORBIDDEN_STAGED_REGEX}" || true)

if [ -n "${FORBIDDEN_CHANGED}" ] || [ -n "${OUT_OF_SCOPE}" ] || [ -n "${STAGED_RUNTIME}" ]; then
    echo "INVARIANT VIOLATION"
    if [ -n "${FORBIDDEN_CHANGED}" ]; then
        echo "Forbidden changed files:"
        printf "%s\n" "${FORBIDDEN_CHANGED}" | print_paths
    fi
    if [ -n "${OUT_OF_SCOPE}" ]; then
        echo "Changed files outside the approved experiment surface:"
        printf "%s\n" "${OUT_OF_SCOPE}" | print_paths
    fi
    if [ -n "${STAGED_RUNTIME}" ]; then
        echo "Staged runtime/generated artifacts:"
        printf "%s\n" "${STAGED_RUNTIME}" | print_paths
    fi
    exit 1
fi

echo "OK: changed files are inside the approved judge-experiment surface and no runtime artifacts are staged."
