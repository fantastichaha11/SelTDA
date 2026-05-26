# C3 Grounding Gates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On branch `feat/grounding-gates`, add LP-1 (language-prior) and KC-1 (knowledge-consistency) gates to the filter cascade via gate registry — without touching generation or orchestration code.

**Architecture:** New scorers in `filtering/scorers_language_prior.py` and `filtering/scorers_knowledge.py`; Wikipedia retrieval module; register in `gate_registry.py`; extend `filter_pseudo.py` scoring loop with disk cache. Cascade order: conf → itm → xcons → lp → kcons.

**Tech Stack:** Python 3.x, torch, PIL, sentence-transformers CrossEncoder, requests (Wikipedia API), pytest.

**Workspace:** Fork from `feat/pseudo-label-filter` → `feat/grounding-gates`

**Spec:** `docs/superpowers/specs/2026-05-27-c3-grounding-gates-design.md`

---

## Prerequisites

- C1 Task 1 (gate registry) merged on `feat/pseudo-label-filter`

```bash
cd /home/phongcoder/Workspace/thesis/SelTDA
git checkout feat/pseudo-label-filter
git pull
git checkout -b feat/grounding-gates
```

---

## File Structure

```
filtering/scorers_language_prior.py
filtering/scorers_knowledge.py
filtering/retrieval/wikipedia.py
filtering/gate_registry.py       # MODIFY — register lp, kcons
filter_pseudo.py               # MODIFY — score + cache new gates
configs/filter_pseudo_grounding.yaml
tests/test_scorers_language_prior.py
tests/test_scorers_knowledge.py
tests/test_wikipedia_retrieval.py
tests/test_grounding_cascade.py
```

---

### Task 0: Branch setup

- [ ] Create `feat/grounding-gates` from updated `feat/pseudo-label-filter`
- [ ] Confirm parent tests pass

---

### Task 1: Image corruption utility

**Files:**
- Create: `filtering/scorers_language_prior.py`
- Create: `tests/test_scorers_language_prior.py`

- [ ] **Step 1: Test corruption**

```python
from PIL import Image
import numpy as np
from filtering.scorers_language_prior import corrupt_image

def test_corrupt_gaussian_changes_pixels():
    img = Image.new("RGB", (32, 32), color=(128, 128, 128))
    out = corrupt_image(img, "gaussian_noise")
    assert not np.array_equal(np.array(img), np.array(out))
```

- [ ] **Step 2: Implement corruption functions**

```python
def corrupt_image(img: Image.Image, method: str) -> Image.Image:
    import numpy as np
    arr = np.array(img).astype(np.float32)
    if method == "gaussian_noise":
        arr = np.clip(arr + np.random.randn(*arr.shape) * 64, 0, 255)
    elif method == "gray_box":
        arr[:] = arr.mean()
    elif method == "shuffle_patches":
        # 4x4 patch shuffle implementation
        ...
    return Image.fromarray(arr.astype(np.uint8))
```

- [ ] **Step 3: Implement `score_language_prior` with mocked student in tests**

```python
def score_language_prior(record, image_path, student, corruption="gaussian_noise") -> float:
    from filtering.matchers import max_match
    target = record["answer"][0]
    q = record["question"]
    a_clean = student.answer(image_path, q)
    a_corrupt = student.answer(corrupt_image(Image.open(image_path), corruption), q)
    clean_ok = max_match(a_clean, target) >= 0.5
    corrupt_ok = max_match(a_corrupt, target) >= 0.5
    if clean_ok and not corrupt_ok:
        return 1.0
    if clean_ok and corrupt_ok:
        return 0.0
    return 0.5
```

- [ ] **Step 4: Commit**

---

### Task 2: Wikipedia retrieval

**Files:**
- Create: `filtering/retrieval/wikipedia.py`
- Create: `tests/test_wikipedia_retrieval.py`

- [ ] **Step 1: Mock HTTP test**

```python
from unittest.mock import patch
from filtering.retrieval.wikipedia import retrieve_passages

@patch("filtering.retrieval.wikipedia._api_search")
def test_retrieve_passages_returns_snippets(mock_search):
    mock_search.return_value = ["Soccer is a sport.", "Football history."]
    passages = retrieve_passages("What sport is this?", "soccer", k=2)
    assert len(passages) == 2
```

- [ ] **Step 2: Implement with cache dir**

```python
def retrieve_passages(question: str, answer: str, k: int = 3, cache_dir: Path | None = None) -> list[str]:
    query = f"{question} {answer}"
    cache_key = hashlib.sha256(query.encode()).hexdigest()
    if cache_dir:
        cache_file = cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())
    snippets = _api_search(query, k)
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(snippets))
    return snippets
```

- [ ] **Step 3: Commit**

---

### Task 3: KC-1 NLI scorer

**Files:**
- Create: `filtering/scorers_knowledge.py`
- Create: `tests/test_scorers_knowledge.py`

- [ ] **Step 1: Test with mocked CrossEncoder**

```python
from unittest.mock import MagicMock
from filtering.scorers_knowledge import score_knowledge_consistency

def test_score_knowledge_max_entailment():
    nli = MagicMock()
    nli.predict.return_value = [0.1, 0.9, 0.2]
    s = score_knowledge_consistency("Q?", "A", ["p1", "p2", "p3"], nli)
    assert s == 0.9
```

- [ ] **Step 2: Implement**

```python
def score_knowledge_consistency(question, answer, passages, nli_model) -> float:
    hypothesis = f"Question: {question} Answer: {answer}"
    scores = []
    for p in passages:
        prob = nli_model.predict([(p, hypothesis)])[0]
        # map label index to entailment prob per model API
        scores.append(float(prob))
    return max(scores) if scores else 0.0
```

- [ ] **Step 3: Commit**

---

### Task 4: Register gates in gate_registry

**Files:**
- Modify: `filtering/gate_registry.py`

- [ ] **Step 1: Extend GATE_SCORERS**

```python
from filtering.scorers_language_prior import score_language_prior
from filtering.scorers_knowledge import score_knowledge_consistency

GATE_SCORERS["lp"] = score_language_prior  # wrapped in filter_pseudo for deps
GATE_SCORERS["kcons"] = score_knowledge_consistency
```

Note: `filter_pseudo.py` passes adapters/models into scorers — registry may use `GATE_RUNNERS` dict of callables `(record, ctx) -> float` instead if signatures differ. Refactor Task 1 scorers to accept `ctx` object:

```python
@dataclass
class ScoreContext:
    student: BlipStudentAdapter
    clip: OpenClipAdapter
    nli_model: object
    image_root: Path
    config: object
```

- [ ] **Step 2: Update tests/test_gate_registry.py for 5 gates**

- [ ] **Step 3: Commit**

---

### Task 5: Wire filter_pseudo.py

**Files:**
- Modify: `filter_pseudo.py`
- Create: `configs/filter_pseudo_grounding.yaml`

- [ ] **Step 1: Add scoring loops for `lp` and `kcons` after xcons**

- [ ] **Step 2: Reuse xcons student answer for LP clean path** (avoid triple forward)

- [ ] **Step 3: Optional `kcons.strata_filter: [external_knowledge]`** — skip KC for non-EK records (score=1.0)

- [ ] **Step 4: JSONL score cache under `cache/gate_scores/`**

- [ ] **Step 5: Integration test `tests/test_grounding_cascade.py` on tiny fixture with gates lp/kcons disabled then enabled**

- [ ] **Step 6: Commit**

---

### Task 6: Config + example script

**Files:**
- Create: `configs/filter_pseudo_grounding.yaml`
- Modify: `examples/filter_synthetic.sh` — add `--config configs/filter_pseudo_grounding.yaml` variant comment

- [ ] Document ablation matrix commands in config header comments

- [ ] Commit

---

### Task 7: Invariant + regression

- [ ] `bash scripts/check_invariants.sh`
- [ ] `pytest tests/ -m "not slow" -v`
- [ ] Final commit on `feat/grounding-gates`

---

## Spec Coverage

| Spec | Task |
|------|------|
| LP-1 | Task 1, 5 |
| KC-1 | Task 2, 3, 5 |
| Gate registry extension | Task 4 |
| Ablation configs | Task 6 |
| Caching | Task 5 |

## Parallel development note

C3 only touches `filtering/` and `filter_pseudo.py`. Safe to develop in parallel with `feat/iterative-seltda` if C2 avoids `filter_pseudo.py` structural changes.

## Merge order (thesis)

1. `feat/pseudo-label-filter` → master (C1)
2. `feat/grounding-gates` rebase → merge (C3)
3. `feat/iterative-seltda` rebase → merge (C2)

C2 and C3 order interchangeable if no conflict in `filter_pseudo.py`.
