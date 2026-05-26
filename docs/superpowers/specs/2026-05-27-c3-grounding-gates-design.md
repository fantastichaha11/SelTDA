# Thiết kế C3 — Grounding-Aware Quality Control

- **Ngày**: 2026-05-27
- **Contribution**: C3 — Grounding-aware quality control (thesis Ch. 4.3 / robustness)
- **Branch**: `feat/grounding-gates` (fork từ `feat/pseudo-label-filter`)
- **Trạng thái**: Approved — chờ implement
- **Dependency**: C1 gate registry (REG-1) trên parent branch

---

## 0. Tóm tắt

Cascade C+I+X loại noise semantic nhưng **không** bắt language shortcuts (ViLP) hay factual hallucination trên EK questions. C3 thêm hai gate:

- **LP-1**: Drop pseudo-QA mà student vẫn trả đúng A khi ảnh bị corrupt mạnh.
- **KC-1**: Drop pseudo-QA mà answer không được supported bởi retrieved Wikipedia passage.

Eval trên A-OKVQA (primary), AdVQA/VQA-CE (robustness), PathVQA replicate (secondary).

---

## 1. Yêu cầu & ràng buộc

### 1.1 In scope

| ID | Gate | Signal |
|----|------|--------|
| LP-1 | `lp` | Answer change under image corruption |
| KC-1 | `kcons` | NLI entailment given retrieved Wikipedia context |

### 1.2 Out of scope

- CT-1 counterfactual (defer thesis appendix)
- SS-1 SemIAug (orthogonal — separate script)
- Sửa train/eval code

### 1.3 Success criteria

| Metric | Target | Falsifier |
|--------|--------|-----------|
| C+I+X+LP vs C+I+X (A-OKVQA) | ≥ +0.5 val | LP ≤ baseline |
| C+I+X+KC vs C+I+X (EK stratum) | ≥ +4.0 | KC ≤ baseline on EK |
| C+I+X+LP+KC vs C+I+X | ≥ +0.5 overall | Combined ≤ single gate |
| AdVQA (LP alone) | ≥ +2.0 | No robustness gain |
| VQA-CE (LP alone) | ≥ +2.0 | No robustness gain |
| Filter precision (human judge 100) | +10% vs C+I+X | No P/R improvement |

---

## 2. Kiến trúc

### 2.1 Extended cascade order

```
conf → itm → xcons → lp → kcons
```

Gate order configurable via `gates._order` in yaml; default above. LP và KC chạy **sau** xcons để giảm compute (xcons đã loại ~40% pool).

Register trong `filtering/gate_registry.py` (from C1):

```python
GATE_SCORERS.update({
    "lp": score_language_prior,
    "kcons": score_knowledge_consistency,
})
```

### 2.2 LP-1 — Language-prior gate

**Corruption strategies** (config chọn một hoặc max score):

| Method | Implementation |
|--------|----------------|
| `gaussian_noise` | PIL Image + numpy noise σ=0.5 |
| `gray_box` | Replace image with mean-color tensor |
| `shuffle_patches` | 4×4 patch shuffle |

Default: `gaussian_noise` (cheapest, ViLP paper comparable).

**Score**:

```python
def score_language_prior(
    record: dict,
    image_path: Path,
    student: BlipStudentAdapter,
    corruption: str = "gaussian_noise",
) -> float:
    a_clean = student.answer(image_path, record["question"])
    a_corrupt = student.answer(corrupt(image_path, corruption), record["question"])
    match_clean = match_answer(a_clean, record["answer"][0])
    match_corrupt = match_answer(a_corrupt, record["answer"][0])
    # Keep high score = visually grounded = answer CHANGES under corruption
    if match_clean and not match_corrupt:
        return 1.0
    if match_clean and match_corrupt:
        return 0.0  # language shortcut
    return 0.5  # ambiguous — neither clean nor corrupt matched pseudo-A
```

**Threshold**: quantile keep_top default 0.75 (same as other gates).

Module: `filtering/scorers_language_prior.py`

**Compute**: 2× student forward per sample (clean already from xcons — **cache xcons forward answer** to avoid triple forward).

### 2.3 KC-1 — Knowledge-consistency gate

**Retrieval**:

```python
# filtering/retrieval/wikipedia.py
def retrieve_passages(question: str, answer: str, k: int = 3) -> list[str]:
    """Wikipedia API search on question keywords + answer entity; return top-k snippets."""
```

Fallback offline: pre-downloaded ConceptNet/Wikipedia subset JSON if API unavailable.

**Entailment**:

```python
# filtering/scorers_knowledge.py
def score_knowledge_consistency(
    question: str,
    answer: str,
    passages: list[str],
    nli_model,  # cross-encoder/nli-deberta-v3-small
) -> float:
    """Max entailment P over passages for hypothesis: 'Q: ... A: ...'"""
```

Use `cross-encoder/nli-deberta-v3-small` via `sentence-transformers` CrossEncoder.

**Threshold**: quantile keep_top 0.75; **ablation**: kcons alone on EK subset.

Module layout:

```
filtering/
├── scorers_language_prior.py   # NEW
├── scorers_knowledge.py        # NEW
└── retrieval/
    ├── __init__.py
    └── wikipedia.py            # NEW
```

### 2.4 Config

`configs/filter_pseudo_grounding.yaml`:

```yaml
defaults:
  - filter_pseudo_stratified  # inherit C1 TS-1

gates:
  lp:
    enabled: true
    keep_top: 0.75
    corruption: gaussian_noise
  kcons:
    enabled: true
    keep_top: 0.75
    retrieval:
      backend: wikipedia_api
      k: 3
      cache_dir: cache/retrieval/
    nli_model: cross-encoder/nli-deberta-v3-small

gates_order: [conf, itm, xcons, lp, kcons]
```

### 2.5 Caching strategy

- **LP cache key**: `sha256(image_id + question + corruption)`
- **KC cache key**: `sha256(question + answer + k)`
- Store in `cache/gate_scores/{lp,kcons}/` JSONL — rerun filter without re-scoring.

---

## 3. Ablation matrix

| # | conf | itm | xcons | lp | kcons | Role |
|---|------|-----|-------|----|----|------|
| 0 | ✓ | ✓ | ✓ | | | C1 baseline |
| 1 | ✓ | ✓ | ✓ | ✓ | | LP ablation |
| 2 | ✓ | ✓ | ✓ | | ✓ | KC ablation (report EK slice) |
| 3 | ✓ | ✓ | ✓ | ✓ | ✓ | Full C3 |

3 seeds each. Primary: A-OKVQA val. Secondary: AdVQA, VQA-CE via existing eval scripts.

---

## 4. File structure

```
filtering/
├── gate_registry.py              # MODIFY (on C1) — C3 adds scorer registration
├── scorers_language_prior.py     # NEW
├── scorers_knowledge.py          # NEW
└── retrieval/
    └── wikipedia.py              # NEW

filter_pseudo.py                  # MODIFY — LP/KC scoring loops + cache
configs/filter_pseudo_grounding.yaml  # NEW

tests/
├── test_scorers_language_prior.py    # NEW — mock student
├── test_scorers_knowledge.py         # NEW — mock NLI
├── test_wikipedia_retrieval.py       # NEW — mock HTTP
└── test_grounding_cascade.py         # NEW — integration tiny fixture
```

---

## 5. Eval protocol

### Primary — A-OKVQA
Ablation matrix §3. Report overall + per-type (especially EK).

### Robustness — LP-1
Run eval scripts unchanged:
```bash
python okvqa_eval.py ...  # if applicable
python vqa_ce_eval.py ...
# AdVQA via configs/advqa.yaml evaluate mode
```

Compare variant 0 vs 1 (LP added).

### PathVQA replicate
Single config: C+I+X+LP+KC @ 0.75, 3 seeds — verify gates portable (Wikipedia → generic retrieval still helps closed-ended).

### Human judge (100 kept + 100 dropped)
Replicate Khan Tab.3 protocol for variant 3 vs variant 0.

---

## 6. Compute budget

| Block | GPU-h |
|-------|-------|
| LP scoring (2× forward, ~50k pool) | ~15 |
| KC scoring (NLI CPU + API) | ~5 (+ network) |
| Ablation matrix 4 variants × 3 seeds | ~120 |
| Robustness eval | ~20 |
| **C3 total** | **~160** |

---

## 7. Branch workflow

```bash
git checkout feat/pseudo-label-filter
git checkout -b feat/grounding-gates
# implement LP-1 + KC-1 only
# do NOT modify generate_questions.py or orchestration/
```

**Parallel with C2**: C2 touches `generate_questions.py`; C3 touches `filtering/` only — minimal merge conflict if developed concurrently.

---

## 8. Risks

| Risk | Mitigation |
|------|------------|
| LP drops counting questions (corrupt breaks count) | Log per-type drop rate; relax keep_top to 0.9 for `how_many` stratum |
| Wikipedia API rate limit | Disk cache + offline fallback JSON |
| KC latency | Score only EK stratum (`strata.classify` pre-filter) — config flag `kcons.strata: [external_knowledge]` |
| Triple student forward cost | Reuse xcons answer; corrupt-only second forward |

---

## 9. References

- ViLP 2501.00569; Prophet (OK-VQA); QACap A-OKVQA
- `research/to_human/2026-05-27-ideas-selection-report.md` §9 C3
