#!/bin/bash
# Enforce spec §1.4: filtering work must not modify training/eval code.
# models/blip.py is allowed (return_logprob optimization for Gate 1 only).
set -euo pipefail
BASE="${1:-origin/master}"

FORBIDDEN_REGEX='^(train_vqa\.py|train_vqg\.py|data/|vqa_eval_tools/|.*_eval\.py|examples/self_train_synthetic\.sh|examples/evaluate\.sh|examples/train_teacher\.sh|configs/(aokvqa|pathvqa|okvqa|advqa|artvqa|rsvqa|vqa)\.yaml)$'

CHANGED=$(git diff --name-only "${BASE}"...HEAD 2>/dev/null || git diff --name-only "${BASE}" HEAD)
echo "Changed vs ${BASE}:"
echo "${CHANGED}"
echo "---"

VIOLATIONS=$(printf "%s\n" "${CHANGED}" | grep -E "${FORBIDDEN_REGEX}" || true)

# models/ is forbidden except models/blip.py (Gate 1 logprob cache)
MODEL_VIOLATIONS=$(printf "%s\n" "${CHANGED}" | grep -E '^models/' | grep -v '^models/blip\.py$' || true)

if [ -n "${VIOLATIONS}" ] || [ -n "${MODEL_VIOLATIONS}" ]; then
    echo "INVARIANT VIOLATION — these files must not be modified:"
    [ -n "${VIOLATIONS}" ] && printf "  %s\n" ${VIOLATIONS}
    [ -n "${MODEL_VIOLATIONS}" ] && printf "  %s\n" ${MODEL_VIOLATIONS}
    exit 1
fi
echo "OK: no forbidden files modified."
