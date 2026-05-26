# C2 Closed-Loop Self-Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On branch `feat/iterative-seltda`, implement IT-1 (2-round iterative SelTDA with early stop), J-1 (staged easy/hard curriculum via pool merge), and TC-1 (type-conditioned generation) — without modifying train internals.

**Architecture:** Python orchestrator spawns subprocesses for `train_vqg.py`, `generate_questions.py`, `filter_pseudo.py`, `train_vqa.py`, `evaluate`. Skill-gap parsed from eval output drives weak-type targeting. PV-1 deferred unless C1 gain < +1.0.

**Tech Stack:** Python 3.x, hydra/omegaconf (existing), subprocess, pytest with mocks.

**Workspace:** Fork from `feat/pseudo-label-filter` → `feat/iterative-seltda`

**Spec:** `docs/superpowers/specs/2026-05-27-c2-closed-loop-design.md`

---

## Prerequisites

- C1 gate registry merged on `feat/pseudo-label-filter`
- `bash scripts/check_invariants.sh` passes on parent branch

```bash
cd /home/phongcoder/Workspace/thesis/SelTDA
git checkout feat/pseudo-label-filter
git pull
git checkout -b feat/iterative-seltda
```

---

## File Structure

```
orchestration/
├── __init__.py
├── iterative_seltda.py
├── skill_gap.py
├── merge_pools.py
├── iterative_aokvqa.yaml
└── staged_curriculum.yaml
generate_questions.py          # MODIFY — type schedule only
configs/generate_questions_aokvqa.yaml  # MODIFY
scripts/run_iterative_round.sh
tests/test_skill_gap.py
tests/test_merge_pools.py
tests/test_type_schedule.py
tests/test_iterative_smoke.py
```

---

### Task 0: Branch + smoke import

- [ ] **Step 1:** Create branch from `feat/pseudo-label-filter`
- [ ] **Step 2:** `pytest tests/ -m "not slow" -q` — parent tests still pass

---

### Task 1: Skill gap parser

**Files:**
- Create: `orchestration/skill_gap.py`
- Create: `tests/test_skill_gap.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_skill_gap.py
import json
from pathlib import Path
from orchestration.skill_gap import compute_skill_gap_from_results


def test_compute_skill_gap_top3_weak(tmp_path):
    results = [
        {"question": "Is it red?", "answer": "yes", "prediction": "no", "correct": False},
        {"question": "Is it blue?", "answer": "no", "prediction": "yes", "correct": False},
        {"question": "How many?", "answer": "2", "prediction": "2", "correct": True},
        {"question": "What kind of sport?", "answer": "soccer", "prediction": "tennis", "correct": False},
    ]
    p = tmp_path / "results.json"
    p.write_text(json.dumps(results))
    report = compute_skill_gap_from_results(p)
    assert len(report.weak_types) == 3
    assert "yes_no" in report.weak_types or "external_knowledge" in report.weak_types
```

- [ ] **Step 2: Implement**

```python
# orchestration/skill_gap.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from filtering.strata import classify_question_type


@dataclass
class SkillGapReport:
    per_type_accuracy: dict[str, float]
    per_type_error_rate: dict[str, float]
    weak_types: list[str]


def compute_skill_gap_from_results(results_path: Path) -> SkillGapReport:
    rows = json.loads(results_path.read_text())
    stats: dict[str, list[bool]] = {}
    for r in rows:
        t = classify_question_type(r["question"], r.get("answer", ""))
        stats.setdefault(t, []).append(bool(r.get("correct", False)))
    acc = {t: sum(v) / len(v) for t, v in stats.items() if v}
    err = {t: 1.0 - acc[t] for t in acc}
    weak = sorted(err, key=err.get, reverse=True)[:3]
    return SkillGapReport(acc, err, weak)
```

- [ ] **Step 3: Run tests — PASS**

- [ ] **Step 4: Commit**

---

### Task 2: J-1 merge pools

**Files:**
- Create: `orchestration/merge_pools.py`
- Create: `tests/test_merge_pools.py`

- [ ] **Step 1: Test duplicate expansion**

```python
from orchestration.merge_pools import merge_with_weights

def test_merge_duplicates_by_weight(tmp_path):
    easy = [{"question_id": 1, "question": "Q?", "answer": ["a"], "image": "x.jpg", "dataset": "aokvqa"}]
    hard = [{"question_id": 2, "question": "Q2?", "answer": ["b"], "image": "y.jpg", "dataset": "aokvqa"}]
    merged = merge_with_weights({"easy": (easy, 1), "hard": (hard, 3)})
    assert len(merged) == 4
    assert sum(1 for r in merged if r["question_id"] == 2) == 3
```

- [ ] **Step 2: Implement `merge_with_weights` — deep-copy records per weight**

- [ ] **Step 3: Commit**

---

### Task 3: TC-1 type schedule in generate_questions.py

**Files:**
- Modify: `generate_questions.py`
- Modify: `configs/generate_questions_aokvqa.yaml`
- Create: `tests/test_type_schedule.py`

- [ ] **Step 1: Add constants and helper**

```python
QUESTION_TYPES = ["yes_no", "how_many", "color", "external_knowledge", "visual_reasoning"]

def next_question_type(index: int, schedule: str, weak_types: list[str] | None = None) -> str:
    if schedule == "round_robin":
        return QUESTION_TYPES[index % len(QUESTION_TYPES)]
    if schedule == "target_weak" and weak_types:
        pool = weak_types * 3 + QUESTION_TYPES
        return pool[index % len(pool)]
    return QUESTION_TYPES[index % len(QUESTION_TYPES)]

def type_prompt_prefix(qtype: str) -> str:
    return f"[TYPE={qtype}] "
```

- [ ] **Step 2: In generation loop**, prepend `type_prompt_prefix(qtype)` to decoder prompt when `config.question_type_schedule != "none"`

- [ ] **Step 3: Config yaml**

```yaml
question_type_schedule: round_robin  # none | round_robin | target_weak
weak_types_file: null  # path to skill_gap JSON when target_weak
```

- [ ] **Step 4: Tests for `next_question_type`**

- [ ] **Step 5: Commit**

---

### Task 4: Iterative orchestrator

**Files:**
- Create: `orchestration/iterative_seltda.py`
- Create: `orchestration/iterative_aokvqa.yaml`
- Create: `tests/test_iterative_smoke.py`

- [ ] **Step 1: State dataclass + save/load**

```python
@dataclass
class RoundState:
    round_id: int
    val_accuracy: float
    skill_gap_path: Path
    synthetic_path: Path
    student_ckpt: Path
```

- [ ] **Step 2: `run_round(config, state)` subprocess sequence:**
  1. Optional `train_vqg.py` if round > 0 (with weak-type subset config override)
  2. `generate_questions.py` with type schedule
  3. `filter_pseudo.py` with C1 stratified config
  4. `merge_pools.py` if J-1 enabled
  5. `train_vqa.py` from pretrained (not resume prior round)
  6. evaluate → write `results_round_{n}.json`

- [ ] **Step 3: Main loop with early stop**

```python
def main():
    # load iterative_aokvqa.yaml
    acc_prev = 0.0
    for r in range(max_rounds):
        acc = run_round(r, ...)
        if r > 0 and acc - acc_prev < early_stop_delta:
            break
        acc_prev = acc
```

- [ ] **Step 4: Smoke test** — mock `subprocess.run`, assert call order

- [ ] **Step 5: Commit**

---

### Task 5: Staged curriculum configs

**Files:**
- Create: `orchestration/staged_curriculum.yaml`
- Create: `scripts/run_iterative_round.sh`

- [ ] **Step 1:** Two filter invocations documented in shell script:
  - easy: all gates keep_top=0.9
  - hard: C+I+X @ 0.75
- [ ] **Step 2:** Merge via `merge_pools.py` → `synthetic_staged.json`
- [ ] **Step 3: Commit**

---

### Task 6: PV-1 gate (conditional — skip unless triggered)

- [ ] **Step 1:** Document trigger in plan README: only if C1 val < 61.0%
- [ ] **Step 2:** If triggered, add `--two_stage` flag to `generate_questions.py` (separate PR/task)

---

### Task 7: Invariant check

- [ ] `bash scripts/check_invariants.sh`
- [ ] `pytest tests/ -m "not slow" -v`
- [ ] Commit on `feat/iterative-seltda`

---

## Spec Coverage

| Spec | Task |
|------|------|
| IT-1 loop | Task 4 |
| J-1 curriculum | Task 2, 5 |
| TC-1 generation | Task 3 |
| Skill gap | Task 1 |
| PV-1 conditional | Task 6 (deferred) |

## Merge note

Do not merge `feat/iterative-seltda` into `feat/pseudo-label-filter` until C1 experiments complete. Resolve `generate_questions.py` conflicts carefully — C2 owns type schedule block.
