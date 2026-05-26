# C1 Structured Curation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `feat/pseudo-label-filter` with type-stratified thresholds (TS-1), coreset selection (CS-X), random-subsample control, saturation sweep orchestration, and extensible gate registry — on top of existing CF-1 cascade.

**Architecture:** Refactor cascade to `gate_registry.py`; add `strata.py` for per-type quantiles and `coreset.py` for submodular dedup; wire hooks in `filter_pseudo.py`. Random control and ratio sweep as standalone scripts. All §1.4 invariants preserved.

**Tech Stack:** Python 3.x, numpy, torch, open_clip (existing), pytest. Optional: port ZCore greedy function (~80 LOC).

**Workspace:** `/home/phongcoder/Workspace/thesis/SelTDA` on branch `feat/pseudo-label-filter`.

**Spec:** `docs/superpowers/specs/2026-05-27-c1-structured-curation-design.md`

---

## Hard Constraints

Same as `2026-05-21-pseudo-label-filtering` plan — do NOT touch `train_vqa.py`, `train_vqg.py`, `data/`, eval scripts. Run `bash scripts/check_invariants.sh` before every commit.

---

## File Structure

```
filtering/gate_registry.py          # NEW
filtering/strata.py                 # NEW
filtering/coreset.py                # NEW
filtering/gates.py                  # MODIFY — thin wrapper
filter_pseudo.py                    # MODIFY — registry + stratify + coreset
scripts/random_subsample_control.py # NEW
scripts/sweep_synth_ratio.sh        # NEW
configs/filter_pseudo_stratified.yaml
configs/filter_pseudo_coreset.yaml
research/experiments/H1-cascade-filter/aokvqa_val_type_dist.json
tests/test_gate_registry.py
tests/test_strata.py
tests/test_coreset.py
tests/test_random_subsample.py
```

---

### Task 1: Gate registry

**Files:**
- Create: `filtering/gate_registry.py`
- Create: `tests/test_gate_registry.py`
- Modify: `filtering/gates.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_gate_registry.py
from filtering.gate_registry import GATE_SCORERS, apply_cascade, enabled_gate_names


def test_enabled_gate_names_respects_config():
    class G:
        enabled = True
        keep_top = 0.75

    class Gates:
        conf = G()
        itm = G()
        xcons = type("X", (), {"enabled": False, "keep_top": 0.75})()

    class Config:
        gates = Gates()

    assert enabled_gate_names(Config()) == ["conf", "itm"]


def test_apply_cascade_fails_at_first_gate():
    scores = {"conf": 0.2, "itm": 0.9, "xcons": 0.9}
    thresholds = {"conf": 0.5, "itm": 0.5, "xcons": 0.5}
    keep, reason = apply_cascade(scores, thresholds, ["conf", "itm", "xcons"])
    assert keep is False
    assert reason == "conf"


def test_apply_cascade_keeps_when_all_pass():
    scores = {"conf": 0.9, "itm": 0.9, "xcons": 0.9}
    thresholds = {"conf": 0.5, "itm": 0.5, "xcons": 0.5}
    keep, reason = apply_cascade(scores, thresholds, ["conf", "itm", "xcons"])
    assert keep is True
    assert reason == "kept"
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `cd /home/phongcoder/Workspace/thesis/SelTDA && pytest tests/test_gate_registry.py -v`

- [ ] **Step 3: Implement**

```python
# filtering/gate_registry.py
from __future__ import annotations

from filtering.scorers import score_clip_itm, score_confidence, score_xcons

GATE_SCORERS = {
    "conf": score_confidence,
    "itm": score_clip_itm,
    "xcons": score_xcons,
}


def enabled_gate_names(config) -> list[str]:
    return [name for name in GATE_SCORERS if getattr(config.gates, name).enabled]


def apply_cascade(scores: dict, thresholds: dict, gate_order: list[str]) -> tuple[bool, str]:
    for gate in gate_order:
        if scores[gate] < thresholds[gate]:
            return False, gate
    return True, "kept"
```

Update `filtering/gates.py`:

```python
from filtering.gate_registry import apply_cascade as apply_gates  # re-export
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add filtering/gate_registry.py filtering/gates.py tests/test_gate_registry.py
git commit -m "feat(filter): add extensible gate registry for C1/C3 cascade"
```

---

### Task 2: Question type classification (TS-1)

**Files:**
- Create: `filtering/strata.py`
- Create: `tests/test_strata.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_strata.py
from filtering.strata import classify_question_type, thresholds_per_stratum


def test_classify_yes_no():
    assert classify_question_type("Is the dog brown?", "yes") == "yes_no"


def test_classify_how_many():
    assert classify_question_type("How many people are there?", "3") == "how_many"


def test_classify_ek():
    q = "What kind of sport is this?"
    assert classify_question_type(q, "soccer") == "external_knowledge"


def test_thresholds_per_stratum_separate_quantiles():
    records = [
        {"question_type": "yes_no", "scores": {"conf": 0.9}},
        {"question_type": "yes_no", "scores": {"conf": 0.1}},
        {"question_type": "visual_reasoning", "scores": {"conf": 0.5}},
        {"question_type": "visual_reasoning", "scores": {"conf": 0.6}},
    ]
    tau = thresholds_per_stratum(records, gate="conf", keep_top=0.5)
    assert "yes_no" in tau
    assert "visual_reasoning" in tau
    assert tau["yes_no"] != tau["visual_reasoning"]
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `filtering/strata.py`**

```python
from __future__ import annotations

import re
from typing import Callable

import numpy as np

from filtering.gates import thresholds_from_quantile

EK_PATTERNS = [
    r"\bwhat kind of\b",
    r"\bwhat type of\b",
    r"\bwhat brand\b",
    r"\bwhat sport\b",
    r"\bwhat country\b",
]


def classify_question_type(question: str, answer: str) -> str:
    q = question.lower().strip()
    a = answer.lower().strip()
    if a in {"yes", "no"}:
        return "yes_no"
    if re.search(r"\bhow many\b", q):
        return "how_many"
    if re.search(r"\bwhat color\b", q):
        return "color"
    for pat in EK_PATTERNS:
        if re.search(pat, q):
            return "external_knowledge"
    return "visual_reasoning"


def assign_types(records: list[dict]) -> None:
    for r in records:
        ans = r["answer"][0] if isinstance(r["answer"], list) else r["answer"]
        r["question_type"] = classify_question_type(r["question"], ans)


def thresholds_per_stratum(
    records: list[dict],
    gate: str,
    keep_top: float,
    min_stratum_size: int = 50,
    global_fallback: float | None = None,
) -> dict[str, float]:
    assign_types(records)
    global_vals = [r["scores"][gate] for r in records if gate in r.get("scores", {})]
    global_tau = thresholds_from_quantile(global_vals, keep_top)
    by_type: dict[str, list[float]] = {}
    for r in records:
        by_type.setdefault(r["question_type"], []).append(r["scores"][gate])
    out = {}
    for t, vals in by_type.items():
        if len(vals) < min_stratum_size and global_fallback is not None:
            out[t] = global_fallback
        else:
            out[t] = thresholds_from_quantile(vals, keep_top)
    return out
```

- [ ] **Step 4: Run tests — expect PASS**

- [ ] **Step 5: Commit**

---

### Task 3: Wire TS-1 into filter_pseudo.py

**Files:**
- Modify: `filter_pseudo.py`
- Create: `configs/filter_pseudo_stratified.yaml`
- Create: `research/experiments/H1-cascade-filter/aokvqa_val_type_dist.json`

- [ ] **Step 1: Add config section**

```yaml
# configs/filter_pseudo_stratified.yaml
defaults:
  - filter_pseudo

stratify:
  enabled: true
  min_stratum_size: 50
  global_keep_top_fallback: 0.75
```

- [ ] **Step 2: After all scores computed, if `config.stratify.enabled`:**
  - Call `assign_types(records)`
  - For final cascade decision, use `thresholds_per_stratum` per gate instead of global `thresholds_from_quantile`
  - Add integration test in `tests/test_filter_pseudo_smoke.py` with `stratify.enabled=true` on tiny fixture

- [ ] **Step 3: Run** `pytest tests/test_filter_pseudo_smoke.py tests/test_strata.py -v`

- [ ] **Step 4: Commit**

---

### Task 4: Coreset selection (CS-X)

**Files:**
- Create: `filtering/coreset.py`
- Create: `tests/test_coreset.py`

- [ ] **Step 1: Write failing test**

```python
import numpy as np
from filtering.coreset import select_coreset


def test_select_coreset_respects_budget():
    emb = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=np.float32)
    idx = select_coreset(emb, budget=2, diversity_weight=0.5)
    assert len(idx) == 2
    assert len(set(idx)) == 2
```

- [ ] **Step 2: Implement greedy submodular**

```python
# filtering/coreset.py
import numpy as np


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def select_coreset(
    embeddings: np.ndarray,
    budget: int,
    diversity_weight: float = 0.5,
) -> list[int]:
    n = len(embeddings)
    if budget >= n:
        return list(range(n))
    selected: list[int] = [0]
    while len(selected) < budget:
        best_j, best_gain = -1, -1.0
        for j in range(n):
            if j in selected:
                continue
            relevance = float(np.linalg.norm(embeddings[j]))
            diversity = min(_cosine_sim(embeddings[j], embeddings[s]) for s in selected)
            gain = (1 - diversity_weight) * relevance + diversity_weight * (1 - diversity)
            if gain > best_gain:
                best_gain, best_j = gain, j
        selected.append(best_j)
    return selected
```

- [ ] **Step 3: Wire in filter_pseudo.py** when `config.coreset.enabled`: embed kept records via CLIP adapter, run per-stratum `select_coreset`, filter indices.

- [ ] **Step 4: Create `configs/filter_pseudo_coreset.yaml`**

- [ ] **Step 5: Commit**

---

### Task 5: Random-subsample control

**Files:**
- Create: `scripts/random_subsample_control.py`
- Create: `tests/test_random_subsample.py`

- [ ] **Step 1: Implement CLI**

```python
#!/usr/bin/env python3
"""Random subsample N records from raw pool matching |reference|."""
import argparse, json, random
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seeds", default="42,43,44")
    args = p.parse_args()
    raw = json.loads(Path(args.input).read_text())
    n = len(json.loads(Path(args.reference).read_text()))
    for seed in map(int, args.seeds.split(",")):
        rng = random.Random(seed)
        sample = rng.sample(raw, min(n, len(raw)))
        out = Path(args.output.replace(".json", f"_seed{seed}.json"))
        out.write_text(json.dumps(sample, indent=2))
        print(f"wrote {out} ({len(sample)} records)")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test with tiny fixture**

- [ ] **Step 3: Commit**

---

### Task 6: Saturation sweep script

**Files:**
- Create: `scripts/sweep_synth_ratio.sh`

- [ ] **Step 1: Shell script** looping ratios `{1:1,2:1,4:1,8:1}` × filter modes, calling existing `examples/run_experiment.sh` or documented train commands with `--overrides truncate_train_dataset_to=...`

- [ ] **Step 2: Document in `research/experiments/H1-cascade-filter/protocol.md`** appendix pointer

- [ ] **Step 3: Commit**

---

### Task 7: Regression & invariant check

- [ ] **Step 1:** `pytest tests/ -m "not slow" -v`
- [ ] **Step 2:** `bash scripts/check_invariants.sh`
- [ ] **Step 3:** Final commit on `feat/pseudo-label-filter`

---

## Spec Coverage Checklist

| Spec § | Task |
|--------|------|
| REG-1 gate registry | Task 1 |
| TS-1 stratified | Task 2, 3 |
| CS-X coreset | Task 4 |
| IDEA-04 random | Task 5 |
| SAT-1 sweep | Task 6 |
| §1.4 invariants | Task 7 |

## Branch Setup (C2/C3 forks — run after Task 1 merged)

```bash
git checkout feat/pseudo-label-filter
git checkout -b feat/iterative-seltda    # for C2 plan
git checkout feat/pseudo-label-filter
git checkout -b feat/grounding-gates     # for C3 plan
```

Do NOT implement C2/C3 on `feat/pseudo-label-filter`.
