# Thiết kế C1 — Structured Pseudo-Label Curation

- **Ngày**: 2026-05-27
- **Contribution**: C1 — Structured pseudo-label curation (thesis Ch. 4.1)
- **Branch**: `feat/pseudo-label-filter` (giữ nguyên, không đổi tên)
- **Trạng thái**: Approved — chờ implement
- **Spec predecessor**: `2026-05-21-pseudo-label-filtering-design.md` (CF-1 baseline — giữ làm reference, không sửa)

---

## 0. Tóm tắt

SelTDA sinh ~30% pseudo-QA noisy (Khan Tab.3). Contribution C1 trả lời: **lọc và chọn subset pseudo-label có cấu trúc** — cascade gates, ngưỡng theo question type, coreset chống duplicate, và đặc tả đường cong bão hòa synthetic:real — vượt unfiltered SelTDA và random-subsample control.

**Phạm vi branch này**: mọi thay đổi filtering + extension interface; **không** iteration (C2), **không** grounding gates (C3).

---

## 1. Yêu cầu & ràng buộc

### 1.1 In scope

| ID | Thành phần | Mô tả |
|----|------------|--------|
| CF-1 | Cascade filter | `conf → ITM → xcons` — **đã implement** |
| IDEA-04 | Random-subsample control | Script chọn ngẫu nhiên N = \|filtered\|, 3 seeds |
| TS-1 | Type-stratified thresholds | Quantile **per question-type stratum** |
| CS-X | Coreset-stratified filter | ZCore-style selection trong từng stratum sau quality gate |
| SAT-1 | Saturation curve | Grid synth:real × {unfiltered, filtered, CS-X} |
| REG-1 | Gate registry | Interface mở rộng gate cho C3 fork sau này |

### 1.2 Out of scope (thuộc branch khác)

- IT-1, J-1, TC-1 → `feat/iterative-seltda` (C2)
- LP-1, KC-1 → `feat/grounding-gates` (C3)
- Sửa `train_vqa.py`, `train_vqg.py`, eval scripts

### 1.3 §1.4 bất biến

Giữ nguyên toàn bộ §1.4 từ spec 2026-05-21. Kiểm tra bằng `scripts/check_invariants.sh` trước mỗi merge.

### 1.4 Success criteria

| Metric | Target | Falsifier |
|--------|--------|-----------|
| A-OKVQA val (best C+I+X filtered) | ≥ 61.5% (+1.5 vs 60.01%) | Không variant nào ≥ +0.5 vs unfiltered |
| vs random-subsample @ matched N | ≥ +0.5 absolute | Filter ≤ random |
| CS-X @ matched N vs filter-only | ≥ +1.0 | CS-X ≤ filter-only |
| SAT-1 | Filtered optimal ratio ≥ 4:1 mà không regression vs 2:1 unfiltered | Curve không dịch |
| PathVQA test (pipeline replicate) | ≥ 30% (direct train + filter) | ≤ 26.76% published |

---

## 2. Kiến trúc

### 2.1 Pipeline

```
synthetic_data_raw.json
    → filter_pseudo.py (score all gates)
    → [optional] stratify thresholds (TS-1)
    → [optional] coreset select (CS-X)
    → synthetic_data.json
    → train_vqa.py (unchanged)
```

### 2.2 Gate registry (REG-1)

Thay `apply_gates()` cứng 3 gate bằng ordered list từ config:

```yaml
gates:
  conf:  { enabled: true,  keep_top: 0.75 }
  itm:   { enabled: true,  keep_top: 0.75 }
  xcons: { enabled: true,  keep_top: 0.75 }
  # C3 sẽ thêm lp, kcons — C1 chỉ define interface, không implement
```

```python
# filtering/gate_registry.py
GATE_SCORERS = {
    "conf": score_confidence,
    "itm": score_clip_itm,
    "xcons": score_xcons,
}

def enabled_gate_names(config) -> list[str]:
    return [name for name in GATE_SCORERS if config.gates[name].enabled]

def apply_cascade(scores: dict, thresholds: dict, gate_order: list[str]) -> tuple[bool, str]:
    for gate in gate_order:
        if scores[gate] < thresholds[gate]:
            return False, gate
    return True, "kept"
```

`filter_pseudo.py` refactor nhẹ: loop theo `enabled_gate_names(config)` thay vì hardcode 3 block.

### 2.3 TS-1 — Type-stratified thresholds

**Question type taxonomy** (rule-based trên `question` string):

| Type | Heuristic |
|------|-----------|
| `yes_no` | Answer ∈ {yes, no} (case-insensitive) |
| `how_many` | Question matches `\bhow many\b` |
| `color` | Question matches `\bwhat color\b` |
| `external_knowledge` | Question matches `\bwhat kind of\b`, `\bwhat type of\b`, `\bwhat brand\b`, `\bwhat sport\b`, … |
| `visual_reasoning` | Default bucket (còn lại) |

Implementation: `filtering/strata.py`

```python
def classify_question_type(question: str, answer: str) -> str: ...

def thresholds_per_stratum(
    records: list[dict],
    gate: str,
    keep_top: float,
    target_counts: dict[str, int] | None = None,
) -> dict[str, float]:
    """Quantile threshold riêng từng stratum; cap tổng N theo val distribution."""
```

**Config flag**:

```yaml
stratify:
  enabled: true
  # Tỷ lệ mục tiêu lấy từ A-OKVQA val set (precomputed JSON)
  target_distribution: research/experiments/H1-cascade-filter/aokvqa_val_type_dist.json
  global_keep_top_fallback: 0.75  # khi stratum quá nhỏ (<50 samples)
```

**Giữ tổng N**: Sau stratified filter, nếu |kept| < budget → backfill từ global pool theo score tổng hợp (mean of normalized gate scores).

### 2.4 CS-X — Coreset selection

Sau quality gates + TS-1, trong mỗi stratum:

1. Embed (I, "Q? A.") bằng CLIP ViT-B/32 (reuse `OpenClipAdapter`).
2. Chạy greedy submodular selection (ZCore algorithm — port từ [voxel51/zcore](https://github.com/voxel51/zcore) hoặc reimplement ~80 lines).
3. Budget = `floor(N_stratum * coreset.keep_ratio)` với `keep_ratio` default 1.0 (no reduction) hoặc 0.9 for ablation.

```yaml
coreset:
  enabled: false  # bật cho CS-X ablation
  keep_ratio: 0.9
  embedding: clip  # reuse ITM model
  diversity_weight: 0.5
```

Module: `filtering/coreset.py`

```python
def select_coreset(
    records: list[dict],
    embeddings: np.ndarray,
    budget: int,
    diversity_weight: float = 0.5,
) -> list[int]:  # indices into records
```

### 2.5 IDEA-04 — Random-subsample control

Script: `scripts/random_subsample_control.py`

```bash
python scripts/random_subsample_control.py \
  --input datasets/aokvqa/synthetic_data_raw.json \
  --reference datasets/aokvqa/synthetic_data.json \
  --output datasets/aokvqa/synthetic_data_random.json \
  --seeds 42,43,44
```

- `reference` = filtered set → lấy N = len(reference).
- Random sample N từ raw pool (không score).
- Output schema giống hệt `synthetic_data.json`.
- Báo cáo trong H1 table: filtered vs random @ matched N.

### 2.6 SAT-1 — Saturation sweep

Orchestration: `scripts/sweep_synth_ratio.sh`

Grid:
- `synth:real` ∈ {1:1, 2:1, 4:1, 8:1}
- Filter mode ∈ {unfiltered, filtered_CIX, filtered_CIX_CSX}

Dùng `--overrides truncate_train_dataset_to=...` trên train config + varying synthetic JSON size (truncate synthetic file hoặc duplicate-sampling inverse).

Output: `research/experiments/saturation/accuracy_vs_ratio.csv` + plot script.

---

## 3. File structure

```
filtering/
├── gate_registry.py      # NEW — ordered cascade
├── strata.py             # NEW — TS-1 type classify + per-stratum thresholds
├── coreset.py            # NEW — CS-X submodular selection
├── gates.py              # MODIFY — delegate to gate_registry
├── scorers.py            # unchanged (C1)
└── ...

filter_pseudo.py          # MODIFY — stratify + coreset hooks
configs/
├── filter_pseudo.yaml                    # unchanged default
├── filter_pseudo_stratified.yaml         # NEW — TS-1
└── filter_pseudo_coreset.yaml            # NEW — TS-1 + CS-X

scripts/
├── random_subsample_control.py           # NEW
└── sweep_synth_ratio.sh                  # NEW

tests/
├── test_strata.py                        # NEW
├── test_coreset.py                       # NEW
├── test_gate_registry.py                 # NEW
└── test_random_subsample.py              # NEW

research/experiments/H1-cascade-filter/
└── aokvqa_val_type_dist.json             # NEW — precomputed type ratios
```

---

## 4. Experiment protocol (H1 extended)

### Phase 1a — Cascade ablation (existing H1 protocol)
8 variants × 3 seeds @ synth:real 2:1.

### Phase 1b — TS-1
Best variant from 1a + `stratify.enabled=true` × 3 seeds.

### Phase 1c — CS-X factorial
2×2: {filter-only, filter+TS-1} × {no coreset, coreset} × 3 seeds.

### Phase 1d — Random control
Variant 7 (C+I+X) vs random-subsample @ matched N × 3 seeds.

### Phase 2 — SAT-1
Winner from 1c × ratio grid (1 seed screen, 3 seeds on peak).

### PathVQA replicate
C+I+X @ 0.75 on PathVQA direct train — 1 config, 3 seeds.

---

## 5. Interface cho C2/C3 fork

C2/C3 fork từ branch này khi **REG-1 merged** và H1 Phase 1a pass smoke test.

**Frozen interface contract:**
- `filter_pseudo.yaml` schema: `gates.<name>.{enabled, keep_top}`
- Output JSON schema: list of `{question_id, question, answer[], image, dataset}` + optional `scores`, `question_type`
- `scripts/check_invariants.sh` pass

C3 sẽ register `lp`, `kcons` vào `GATE_SCORERS` without modifying cascade logic.

---

## 6. Compute budget

| Block | GPU-h (A5000) |
|-------|---------------|
| H1 extended (1a–1d) | ~280 |
| CS-X embedding one-time | ~5 |
| SAT-1 sweep | ~40 |
| PathVQA replicate | ~40 |
| **C1 total** | **~365** |

---

## 7. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| TS-1 stratum quá nhỏ | `global_keep_top_fallback`; merge rare types vào `visual_reasoning` |
| CS-X embedding không capture semantic dup | Ablation: dedupe by SBERT Q-A cosine > 0.9 as naive baseline |
| Filter ≤ random | Falsify H1; document negative result; không chạy SAT-1 full grid |
| Gate registry refactor breaks tests | TDD: `test_gate_registry.py` trước khi refactor `filter_pseudo.py` |

---

## 8. References

- `research/to_human/2026-05-27-ideas-selection-report.md` §9 C1
- `research/experiments/H1-cascade-filter/protocol.md`
- Khan et al. CVPR 2023 Tab.3, Tab.4
- JointMatch 2310.14583; ZCore 2411.15349
