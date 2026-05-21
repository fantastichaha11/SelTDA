#!/bin/bash
# Enforce spec §1.4: filtering work must not modify training/eval code.
set -euo pipefail
BASE="${1:-origin/master}"

FORBIDDEN_REGEX='^(train_vqa\.py|train_vqg\.py|data/|models/|vqa_eval_tools/|.*_eval\.py|examples/self_train_synthetic\.sh|examples/evaluate\.sh|examples/train_teacher\.sh|configs/(aokvqa|pathvqa|okvqa|advqa|artvqa|rsvqa|vqa)\.yaml)$'

CHANGED=$(git diff --name-only "${BASE}"...HEAD 2>/dev/null || git diff --name-only "${BASE}" HEAD)
echo "Changed vs ${BASE}:"
echo "${CHANGED}"
echo "---"

VIOLATIONS=$(printf "%s\n" "${CHANGED}" | grep -E "${FORBIDDEN_REGEX}" || true)

if [ -n "${VIOLATIONS}" ]; then
    echo "INVARIANT VIOLATION — these files must not be modified:"
    printf "  %s\n" ${VIOLATIONS}
    exit 1
fi
echo "OK: no forbidden files modified."
