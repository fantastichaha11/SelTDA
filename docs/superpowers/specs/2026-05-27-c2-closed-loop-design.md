# Thiết kế C2 — Closed-Loop Self-Training

- **Ngày**: 2026-05-27
- **Contribution**: C2 — Closed-loop self-training (thesis Ch. 4.2)
- **Branch**: `feat/iterative-seltda` (fork từ `feat/pseudo-label-filter`)
- **Trạng thái**: Approved — chờ implement
- **Dependency**: C1 gate registry + filter pipeline stable trên parent branch

---

## 0. Tóm tắt

SelTDA dừng sau **một vòng** teacher→student dù val errors chỉ rõ skill yếu. C2 đóng vòng lặp: đo skill-gap per question type → re-target generation & training → compound gain. Kết hợp staged curriculum (J-1) và type-conditioned generation (TC-1). PV-1 (two-stage VQG) chỉ chạy **conditional** nếu C1 best gain < +1.0 val.

**§1.4**: Chỉ orchestration + sửa `generate_questions.py` (type schedule). Không sửa `train_vqa.py` / `train_vqg.py` internals.

---

## 1. Yêu cầu & ràng buộc

### 1.1 In scope

| ID | Thành phần | Mô tả |
|----|------------|--------|
| IT-1 | Iterative SelTDA | 2 vòng max; early stop nếu round-2 gain < 0.5 val |
| J-1 | Staged pool curriculum | `synthetic_easy.json` + `synthetic_hard.json`; duplicate-sampling |
| TC-1 | Type-conditioned generation | `--question_type` schedule; oversample weak types |
| PV-1 | Two-stage VQG (BARE) | **Conditional** — chỉ nếu C1 < +1.0 |

### 1.2 Out of scope

- Filter gates mới (C3)
- TS-1/CS-X implementation (inherit từ C1 parent)
- Sửa training loss / eval code

### 1.3 Success criteria

| Metric | Target | Falsifier |
|--------|--------|-----------|
| Round-2 vs Round-1 val | ≥ +1.0 absolute | Round-2 < +0.5 → stop, report saturation |
| EK / VR per-type (round-2) | ≥ +3 each vs round-1 | Regression > 0.5 overall |
| J-1 vs single pool | ≥ +0.5 final acc | No gain |
| TC-1 type distribution | KL div to val < 0.1 | Still dominated by how_many |
| PathVQA (1 round sanity) | ≥ +2 vs zero-shot SelTDA | No gain |

---

## 2. Kiến trúc

### 2.1 Round loop (IT-1)

```
Round 0 (baseline — from C1):
  filter(synthetic_raw) → train student_0 → eval val → skill_gap_0.json

Round 1:
  identify top-3 weak types from skill_gap_0
  re-fine-tune VQG on train subset matching weak types (subprocess train_vqg.py)
  generate with TC-1 schedule targeting weak types
  filter (C1 pipeline) → synthetic_1.json
  train student_1 FROM PRETRAINED (restart weights — Cascante-Bonilla)
  eval → skill_gap_1.json
  if acc_1 - acc_0 < 0.5: STOP

Round 2 (optional — only if round-1 gain ≥ 0.5):
  repeat with skill_gap_1
  early stop same rule
```

**Restart policy**: Mỗi round student train từ `cache/student_weights/checkpoint_09.pth` (pretrained BLIP), **không** fine-tune tiếp từ student round trước — tránh confirmation bias.

### 2.2 Skill gap module

`filtering/skill_gap.py` (despite name, lives under orchestration concern — place in `orchestration/skill_gap.py`):

```python
@dataclass
class SkillGapReport:
    per_type_accuracy: dict[str, float]
    per_type_error_rate: dict[str, float]
    weak_types: list[str]  # top-3 by error rate

def compute_skill_gap(
    eval_results_path: Path,
    type_classifier: Callable[[str, str], str],  # reuse strata.classify_question_type
) -> SkillGapReport: ...
```

Input: output JSON từ eval script (parse `vqa_result.json` hoặc log file từ `--evaluate`).

### 2.3 J-1 — Staged curriculum

Hai lần chạy `filter_pseudo.py`:

| Pool | Config | keep_top |
|------|--------|----------|
| `synthetic_easy.json` | lenient | conf=0.9, itm=0.9, xcons=0.9 |
| `synthetic_hard.json` | strict C+I+X | 0.75 each |

Training orchestration (không sửa train_vqa.py):

```yaml
# orchestration/staged_curriculum.yaml
train_files:
  - train
  - synthetic_easy    # weight 1
  - synthetic_hard    # weight 3 (duplicate-sampling)
duplicate_weights:
  synthetic_easy: 1
  synthetic_hard: 3
```

Implementation: `orchestration/merge_pools.py` — merge hai JSON thành một file với duplicate entries theo weight, **hoặc** pre-expand JSON (preferred — train reader unchanged).

### 2.4 TC-1 — Type-conditioned generation

Sửa `generate_questions.py`:

```python
QUESTION_TYPES = [
    "yes_no", "how_many", "color", "external_knowledge", "visual_reasoning"
]

# New CLI override: question_type_schedule = round_robin | target_weak | uniform
# New prompt prefix: "[TYPE=external_knowledge] Question:"
```

Config `configs/generate_questions_aokvqa.yaml`:

```yaml
question_type_schedule: round_robin  # or target_weak (reads skill_gap JSON)
type_prompt_template: "[TYPE={type}] Question:"
questions_per_image: 2
```

**Oversample weak types**: Khi `target_weak`, 70% generation budget cho top-3 weak types từ skill gap.

### 2.5 PV-1 — Conditional two-stage VQG

**Trigger**: Chỉ implement nếu C1 H1 best < 61.0% val (gain < +1.0).

Stage 1: decode Q only từ image prompt.
Stage 2: conditional `"Question: {Q} Answer:"` → A.
Filter Q và A độc lập (reuse C1 gates per field).

Module: `generate_questions_twostage.py` hoặc flag `--two_stage` trong `generate_questions.py`.

---

## 3. Orchestration entrypoint

`orchestration/iterative_seltda.py`:

```bash
python orchestration/iterative_seltda.py \
  --config orchestration/iterative_aokvqa.yaml \
  --max-rounds 2 \
  --early_stop_delta 0.5
```

```yaml
# orchestration/iterative_aokvqa.yaml
dataset: aokvqa
rounds:
  max: 2
  early_stop_delta: 0.5
  restart_student: true
filter_config: configs/filter_pseudo_stratified.yaml
generate_config: configs/generate_questions_aokvqa.yaml
train_vqa_config: configs/aokvqa.yaml
train_vqg_config: configs/aokvqg.yaml
staged_curriculum: orchestration/staged_curriculum.yaml
output_dir: cache/iterative_seltda/
```

Subprocess calls only — không import train internals.

### State machine

```
INIT → ROUND_N_GENERATE → ROUND_N_FILTER → ROUND_N_TRAIN → ROUND_N_EVAL
  → (gain >= delta ? ROUND_N+1 : DONE)
  → (N >= max_rounds ? DONE)
```

Persist `orchestration/state/round_{n}.json` cho resume.

---

## 4. File structure

```
orchestration/
├── __init__.py
├── iterative_seltda.py       # NEW — main loop
├── skill_gap.py              # NEW — parse eval → weak types
├── merge_pools.py            # NEW — J-1 duplicate-sampling merge
├── staged_curriculum.yaml    # NEW
├── iterative_aokvqa.yaml     # NEW
└── state/                    # runtime (gitignored)

generate_questions.py         # MODIFY — type schedule + optional two_stage
configs/generate_questions_aokvqa.yaml  # MODIFY — type fields

scripts/
└── run_iterative_round.sh    # NEW — single round debug

tests/
├── test_skill_gap.py         # NEW
├── test_merge_pools.py       # NEW
├── test_type_schedule.py     # NEW
└── test_iterative_smoke.py     # NEW — mock subprocess
```

---

## 5. Experiment protocol

### H6 — Iterative SelTDA

| Run | Description | Seeds |
|-----|-------------|-------|
| R0 | C1 best filter, single round (baseline) | 3 |
| R1 | IT-1 round 1 only | 3 |
| R2 | IT-1 full (2 rounds, early stop) | 3 |
| R1+J | IT-1 + staged curriculum | 3 |
| R1+TC | IT-1 + TC-1 only | 3 |
| R2+J+TC | Full C2 stack | 3 |

Report: val accuracy per round, per-type breakdown, KL(type distribution synthetic vs val).

### PathVQA sanity

1 round IT-1 on PathVQA direct train — verify mechanism transfers, not primary claim.

---

## 6. Compute budget

| Block | GPU-h |
|-------|-------|
| IT-1 × 2 rounds (full train each) | ~320 |
| J-1 / TC-1 ablations | ~80 |
| PathVQA 1 round | ~40 |
| PV-1 (conditional) | ~60 |
| **C2 total** | **~440–500** |

---

## 7. Branch workflow

```bash
git checkout feat/pseudo-label-filter
git pull
git checkout -b feat/iterative-seltda
# implement C2 only
# merge back to feat/pseudo-label-filter after C1 frozen, or keep parallel until thesis merge
```

**Merge conflict zones**: `generate_questions.py` — C2 owns type schedule; C3 should not touch this file.

---

## 8. Risks

| Risk | Mitigation |
|------|------------|
| Confirmation bias despite restart | Hold-out human judge 100 samples per round |
| VQG re-fine-tune overfit 17k | Limit to 2 epochs; subset ≤ 5k weak-type samples |
| Orchestration failure mid-round | `state/round_n.json` resume |
| TC-1 prompt hack ignored by BLIP | Log % outputs matching type heuristic post-hoc |

---

## 9. References

- DataEnvGym 2410.06215; Cascante-Bonilla 2001.06001
- `research/to_human/2026-05-27-ideas-selection-report.md` §9 C2
- BARE 2502.01697 (PV-1)
