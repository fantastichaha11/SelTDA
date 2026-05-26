#!/bin/bash
# C2: staged curriculum — filter easy (keep_top=0.9) + hard (C+I+X @ 0.75), then merge.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

python filter_pseudo.py --config configs/filter_pseudo.yaml \
  --overrides gates.conf.keep_top=0.9,gates.itm.keep_top=0.9,gates.xcons.keep_top=0.9,\
output=datasets/aokvqa/synthetic_easy.json

python filter_pseudo.py --config configs/filter_pseudo_stratified.yaml \
  --overrides output=datasets/aokvqa/synthetic_hard.json

python -c "
from pathlib import Path
import json
from orchestration.merge_pools import merge_with_weights
easy = json.loads(Path('datasets/aokvqa/synthetic_easy.json').read_text())
hard = json.loads(Path('datasets/aokvqa/synthetic_hard.json').read_text())
merged = merge_with_weights({'easy': (easy, 1), 'hard': (hard, 3)})
Path('datasets/aokvqa/synthetic_staged.json').write_text(json.dumps(merged, indent=2))
print('wrote synthetic_staged.json', len(merged), 'records')
"
