# Pseudo-label Filtering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 3-gate quality filter (decoder confidence + CLIP-ITM + cross-consistency) between SelTDA's `generate_questions.py` and `train_vqa.py`, exposing per-sample scores and producing a drop-in `synthetic_data.json` without touching any training/eval code.

**Architecture:** Pure-function scorers in `filtering/` orchestrated by a new `filter_pseudo.py` entrypoint. Sequential cascade gating: a sample is kept iff it passes Confidence AND CLIP-ITM AND Cross-consistency. Each gate is independently switchable for ablation. Output JSON keeps the existing schema (additive optional fields only), so the training reader stays untouched.

**Tech Stack:** Python 3.x, `attrs`/`cattrs` (already in repo), `omegaconf`/`hydra` (already), `transformers==4.21.3` (already), `torch`, `Pillow`. New deps: `open_clip_torch` (CLIP), `sentence-transformers` (paraphrase match). Tests via `pytest` (already configured in `pytest.ini`).

**Workspace:** Implement trực tiếp trong repo hiện có `/home/phongcoder/Workspace/thesis/SelTDA` trên branch `feat/pseudo-label-filter`. **Không** tạo git worktree hay folder sibling (`SelTDA-filter/`). Mọi lệnh `cd`, `pytest`, `git` chạy từ thư mục SelTDA đó.

---

## Hard Constraints (from spec §1.4)

Throughout the plan, **DO NOT** touch any of the following:

- `train_vqa.py`, `train_vqg.py`
- `data/vqa_dataset.py`, `data/vqg_dataset.py`, anything else under `data/`
- `models/` (BLIP architecture)
- `vqa_eval_tools/`, any `*_eval.py`
- `examples/self_train_synthetic.sh`, `examples/evaluate.sh`, `examples/train_teacher.sh`
- `configs/aokvqa.yaml`, `configs/pathvqa.yaml`, and any other train/eval config

Allowed edits: `generate_questions.py` only (additive). Everything else is brand-new files under `filtering/`, `tests/`, `configs/filter_pseudo*.yaml`, `examples/filter_synthetic.sh`, plus the new entrypoint `filter_pseudo.py`.

A grep guard (Task 13) enforces this before merge.

---

## File Structure

```
SelTDA/
├── filter_pseudo.py                    # NEW — entrypoint orchestrator
├── filtering/                          # NEW — module
│   ├── __init__.py
│   ├── matchers.py                     # exact_match, sbert_match, max_match
│   ├── scorers.py                      # score_confidence, score_clip_itm, score_xcons
│   ├── gates.py                        # apply_gates (cascade), thresholds_from_quantile
│   ├── io.py                           # load_records, dump_records (preserves schema)
│   └── report.py                       # build_filter_report
├── configs/
│   ├── filter_pseudo.yaml              # NEW — default config
│   └── filter_pseudo_aokvqa.yaml       # NEW — A-OKVQA preset
├── examples/
│   └── filter_synthetic.sh             # NEW
├── tests/
│   ├── test_matchers.py                # NEW
│   ├── test_scorers_confidence.py      # NEW
│   ├── test_scorers_clip.py            # NEW (marked slow)
│   ├── test_scorers_xcons.py           # NEW (marked slow)
│   ├── test_gates.py                   # NEW
│   ├── test_io.py                      # NEW
│   ├── test_report.py                  # NEW
│   ├── test_filter_pseudo_smoke.py     # NEW (end-to-end on tiny fixture)
│   └── fixtures/
│       ├── synthetic_data_raw_tiny.json   # 5 records, hand-crafted
│       └── images/                        # 5 tiny PNGs (created in code via PIL)
└── scripts/
    └── check_invariants.sh             # NEW — guard against touching forbidden files
generate_questions.py                   # MODIFY — additive only (Task 11–12)
```

**Responsibility per file:**

- `matchers.py`: string/embedding matching primitives. No I/O, no torch.
- `scorers.py`: 3 score functions, each takes a record (+ image / model) → float in [0,1]. Pure once dependencies passed in.
- `gates.py`: cascade logic + threshold conversion (quantile → absolute). Pure functions on score dicts.
- `io.py`: read/write JSON list-of-dicts. Preserves all fields. Round-trip safe.
- `report.py`: aggregate counts, histograms, sample lists → dict.
- `filter_pseudo.py`: parse hydra config → load records → run scorers → apply gates → write outputs.

---

## Task 0: Feature branch + dependency staging

**Files:**
- Modify: `SelTDA/requirements.txt`

- [ ] **Step 0.1: Create feature branch in repo hiện tại**

```bash
cd /home/phongcoder/Workspace/thesis/SelTDA
git fetch origin
git checkout -b feat/pseudo-label-filter
```

Expected: branch `feat/pseudo-label-filter` checked out trong `/home/phongcoder/Workspace/thesis/SelTDA`. Mọi task sau chạy tại đây — **không** dùng `git worktree`, **không** tạo folder mới.

- [ ] **Step 0.2: Append new dependencies to `requirements.txt`**

Open `requirements.txt` and append:

```
open_clip_torch==2.20.0
sentence-transformers==2.2.2
```

(Pin to versions compatible with `transformers==4.21.3` — already validated by HF community as of 2024.)

- [ ] **Step 0.3: Chuẩn bị môi trường bằng `setup.sh`**

```bash
cd /home/phongcoder/Workspace/thesis/SelTDA
bash setup.sh
```

Script này sẽ:
- Tạo conda env `vqa` (Python 3.10) — nếu env đã tồn tại, xóa trước: `conda env remove -n vqa -y`
- Export `PYTHONNOUSERSITE=1` (tránh pip lấy package từ `~/.local`)
- Cài PyTorch + torchvision (CUDA 12.6 wheel)
- `pip install -r requirements.txt` (gồm cả deps mới ở bước 0.2, gồm `huggingface_hub==0.13.4` cho tương thích ST 2.2.2)

Sau khi xong, activate:

```bash
conda activate vqa
```

Verify:

```bash
python -c "import open_clip, sentence_transformers, torch, transformers; print('OK')"
```

Expected: in `OK`. Nếu env `vqa` đã tồn tại từ trước, xóa hoặc đổi tên env cũ trước khi chạy lại `setup.sh` (`conda env remove -n vqa`).

> **Lưu ý:** Không dùng `pip install -r requirements.txt` thủ công ngoài `setup.sh` trừ khi debug — `setup.sh` là nguồn chuẩn cho môi trường SelTDA trên máy local / Vast.ai.

- [ ] **Step 0.4: Commit**

```bash
git add requirements.txt
git commit -m "build: add open_clip_torch and sentence-transformers for pseudo-label filtering"
```

---

## Task 1: `matchers.py` — exact_match + sbert_match (pure functions)

**Files:**
- Create: `filtering/__init__.py` (empty)
- Create: `filtering/matchers.py`
- Test: `tests/test_matchers.py`

- [ ] **Step 1.1: Write failing tests**

Create `tests/test_matchers.py`:

```python
import pytest
from filtering.matchers import exact_match, sbert_match, max_match


def test_exact_match_identical():
    assert exact_match("dog", "dog") == 1.0


def test_exact_match_case_insensitive_with_punct():
    assert exact_match("A Dog.", "a dog") == 1.0


def test_exact_match_mismatch():
    assert exact_match("dog", "cat") == 0.0


def test_exact_match_yes_no():
    assert exact_match("yes", "Yes") == 1.0
    assert exact_match("no", "yes") == 0.0


class DummySbert:
    """Deterministic stand-in for SentenceTransformer for unit testing."""

    def encode(self, texts, convert_to_numpy=True):
        import numpy as np
        out = []
        for t in texts:
            t = t.lower().strip()
            if "dog" in t:
                out.append(np.array([1.0, 0.0]))
            elif "cat" in t:
                out.append(np.array([0.0, 1.0]))
            else:
                out.append(np.array([0.5, 0.5]))
        return np.stack(out)


def test_sbert_match_paraphrase():
    score = sbert_match("a dog", "the dog", model=DummySbert())
    assert score == pytest.approx(1.0, abs=1e-6)


def test_sbert_match_different():
    score = sbert_match("dog", "cat", model=DummySbert())
    assert score == pytest.approx(0.0, abs=1e-6)


def test_max_match_prefers_exact_when_high():
    score = max_match("dog", "dog", sbert_model=DummySbert())
    assert score == 1.0


def test_max_match_falls_back_to_sbert():
    score = max_match("a dog", "the dog", sbert_model=DummySbert())
    assert score == pytest.approx(1.0, abs=1e-6)
```

- [ ] **Step 1.2: Run tests; expect ImportError**

```bash
cd /home/phongcoder/Workspace/thesis/SelTDA
pytest tests/test_matchers.py -v
```

Expected: `ModuleNotFoundError: No module named 'filtering'`.

- [ ] **Step 1.3: Implement `filtering/matchers.py`**

Create `filtering/__init__.py` (empty), then `filtering/matchers.py`:

```python
"""Pure-function matchers for cross-consistency scoring."""

from __future__ import annotations
import re
import string
from typing import Protocol

import numpy as np


_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize(text: str) -> str:
    return text.lower().translate(_PUNCT_TABLE).strip()


def exact_match(pred: str, ref: str) -> float:
    """Lowercase + strip punct + exact string compare. Returns 1.0 / 0.0."""
    return 1.0 if _normalize(pred) == _normalize(ref) else 0.0


class SbertLike(Protocol):
    def encode(self, texts, convert_to_numpy: bool = True): ...


def sbert_match(pred: str, ref: str, model: SbertLike) -> float:
    """Cosine similarity of SBERT embeddings; rescaled to [0, 1]."""
    embs = model.encode([pred, ref], convert_to_numpy=True)
    a, b = embs[0], embs[1]
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    cos = float(np.dot(a, b) / (na * nb))
    return max(0.0, min(1.0, (cos + 1.0) / 2.0))


def max_match(pred: str, ref: str, sbert_model: SbertLike) -> float:
    """max(exact, sbert) — see spec §3.3."""
    e = exact_match(pred, ref)
    if e >= 1.0:
        return 1.0
    return max(e, sbert_match(pred, ref, sbert_model))
```

- [ ] **Step 1.4: Tests should pass**

```bash
pytest tests/test_matchers.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 1.5: Commit**

```bash
git add filtering/__init__.py filtering/matchers.py tests/test_matchers.py
git commit -m "feat(filtering): add string and SBERT-based matchers for Gate 3"
```

---

## Task 2: `scorers.py` — confidence score (pure function over log-probs)

**Files:**
- Create: `filtering/scorers.py`
- Test: `tests/test_scorers_confidence.py`

- [ ] **Step 2.1: Write failing test**

Create `tests/test_scorers_confidence.py`:

```python
import math
import pytest
from filtering.scorers import score_confidence, normalize_min_max


def test_score_confidence_returns_log_prob_when_present():
    record = {"gen_logprob": -1.5}
    assert score_confidence(record) == -1.5


def test_score_confidence_returns_none_when_missing():
    assert score_confidence({}) is None


def test_score_confidence_returns_none_when_explicit_none():
    assert score_confidence({"gen_logprob": None}) is None


def test_normalize_min_max_basic():
    out = normalize_min_max([-3.0, -1.0, 0.0])
    assert out == pytest.approx([0.0, 2.0 / 3.0, 1.0])


def test_normalize_min_max_handles_constant():
    out = normalize_min_max([-2.0, -2.0, -2.0])
    assert out == [0.5, 0.5, 0.5]


def test_normalize_min_max_ignores_nones():
    out = normalize_min_max([-3.0, None, 0.0])
    assert out[0] == pytest.approx(0.0)
    assert out[1] is None
    assert out[2] == pytest.approx(1.0)
```

- [ ] **Step 2.2: Run — expect ImportError**

```bash
pytest tests/test_scorers_confidence.py -v
```

- [ ] **Step 2.3: Implement minimal `filtering/scorers.py`**

```python
"""Per-sample score functions for the 3 filter gates."""

from __future__ import annotations
from typing import Optional, Sequence, List


def score_confidence(record: dict) -> Optional[float]:
    """Return the teacher decoder mean log-prob attached to a record.

    Records produced by the modified generate_questions.py will have
    `gen_logprob: float`. Older records will not.
    """
    v = record.get("gen_logprob")
    if v is None:
        return None
    return float(v)


def normalize_min_max(values: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Min-max normalize a sequence, preserving None positions.

    Constant non-None values map to 0.5.
    """
    non_none = [v for v in values if v is not None]
    if not non_none:
        return [None for _ in values]
    lo, hi = min(non_none), max(non_none)
    if lo == hi:
        return [0.5 if v is not None else None for v in values]
    span = hi - lo
    return [None if v is None else (v - lo) / span for v in values]
```

- [ ] **Step 2.4: Tests pass**

```bash
pytest tests/test_scorers_confidence.py -v
```

- [ ] **Step 2.5: Commit**

```bash
git add filtering/scorers.py tests/test_scorers_confidence.py
git commit -m "feat(filtering): add confidence scorer and min-max normalization"
```

---

## Task 3: `scorers.py` — CLIP image–text matching score

**Files:**
- Modify: `filtering/scorers.py` (extend)
- Test: `tests/test_scorers_clip.py`

- [ ] **Step 3.1: Write failing test (dependency-injected, fast)**

Create `tests/test_scorers_clip.py`:

```python
import pytest
import numpy as np
from filtering.scorers import score_clip_itm


class StubClip:
    """Returns image and text embeddings that we control for the test."""

    def __init__(self, image_emb, text_emb):
        self.image_emb = image_emb
        self.text_emb = text_emb
        self.last_text = None

    def embed_image(self, image):
        return self.image_emb

    def embed_text(self, text):
        self.last_text = text
        return self.text_emb


def test_score_clip_itm_perfect_alignment():
    clip = StubClip(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
    s = score_clip_itm(
        record={"question": "is this a dog?", "answer": ["yes"]},
        image=None,
        clip=clip,
    )
    assert s == pytest.approx(1.0, abs=1e-6)


def test_score_clip_itm_orthogonal_is_half():
    clip = StubClip(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    s = score_clip_itm(
        record={"question": "q?", "answer": ["a"]},
        image=None,
        clip=clip,
    )
    assert s == pytest.approx(0.5, abs=1e-6)


def test_score_clip_itm_text_format_includes_question_and_answer():
    clip = StubClip(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
    score_clip_itm(
        record={"question": "is this a dog?", "answer": ["yes"]},
        image=None,
        clip=clip,
    )
    assert clip.last_text == "is this a dog? yes"


def test_score_clip_itm_uses_first_answer_when_list():
    clip = StubClip(np.array([1.0, 0.0]), np.array([1.0, 0.0]))
    score_clip_itm(
        record={"question": "q?", "answer": ["a", "b", "c"]},
        image=None,
        clip=clip,
    )
    assert clip.last_text == "q? a"
```

- [ ] **Step 3.2: Run — expect ImportError on `score_clip_itm`**

```bash
pytest tests/test_scorers_clip.py -v
```

- [ ] **Step 3.3: Extend `filtering/scorers.py`**

Append to the existing file:

```python
import numpy as np
from typing import Protocol


class ClipLike(Protocol):
    def embed_image(self, image): ...
    def embed_text(self, text: str): ...


def _format_qa_for_clip(record: dict) -> str:
    answer = record["answer"]
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    return f"{record['question']} {answer}".strip()


def score_clip_itm(record: dict, image, clip: ClipLike) -> float:
    """Cosine similarity between CLIP image embedding and "Q? A." text embedding,
    rescaled to [0, 1]. Image preprocessing is the caller's responsibility.
    """
    text = _format_qa_for_clip(record)
    img_emb = np.asarray(clip.embed_image(image), dtype=np.float64).reshape(-1)
    txt_emb = np.asarray(clip.embed_text(text), dtype=np.float64).reshape(-1)
    ni = np.linalg.norm(img_emb)
    nt = np.linalg.norm(txt_emb)
    if ni == 0 or nt == 0:
        return 0.0
    cos = float(np.dot(img_emb, txt_emb) / (ni * nt))
    return max(0.0, min(1.0, (cos + 1.0) / 2.0))
```

- [ ] **Step 3.4: Tests pass**

```bash
pytest tests/test_scorers_clip.py -v
```

- [ ] **Step 3.5: Commit**

```bash
git add filtering/scorers.py tests/test_scorers_clip.py
git commit -m "feat(filtering): add CLIP-ITM scorer with question+answer text format"
```

---

## Task 4: `scorers.py` — cross-consistency (Student zero-shot vs pseudo answer)

**Files:**
- Modify: `filtering/scorers.py` (extend)
- Test: `tests/test_scorers_xcons.py`

- [ ] **Step 4.1: Write failing test**

Create `tests/test_scorers_xcons.py`:

```python
import pytest
from filtering.scorers import score_xcons


class StubStudent:
    """Returns pre-programmed answers for given (image, question)."""

    def __init__(self, answer: str):
        self.answer = answer
        self.last_call = None

    def answer_question(self, image, question: str) -> str:
        self.last_call = (image, question)
        return self.answer


class StubSbert:
    def encode(self, texts, convert_to_numpy=True):
        import numpy as np
        out = []
        for t in texts:
            out.append(np.array([1.0, 0.0]) if "dog" in t.lower() else np.array([0.0, 1.0]))
        return np.stack(out)


def test_score_xcons_exact_match():
    student = StubStudent("dog")
    s = score_xcons(
        record={"question": "what is this?", "answer": ["dog"]},
        image=None,
        student=student,
        sbert=StubSbert(),
    )
    assert s == 1.0


def test_score_xcons_falls_back_to_sbert_paraphrase():
    student = StubStudent("a dog")
    s = score_xcons(
        record={"question": "what is this?", "answer": ["dog"]},
        image=None,
        student=student,
        sbert=StubSbert(),
    )
    assert s == pytest.approx(1.0, abs=1e-6)


def test_score_xcons_mismatch():
    student = StubStudent("cat")
    s = score_xcons(
        record={"question": "what is this?", "answer": ["dog"]},
        image=None,
        student=student,
        sbert=StubSbert(),
    )
    assert s == 0.0


def test_score_xcons_passes_correct_question_to_student():
    student = StubStudent("dog")
    score_xcons(
        record={"question": "what is this?", "answer": ["dog"]},
        image="img_obj",
        student=student,
        sbert=StubSbert(),
    )
    assert student.last_call == ("img_obj", "what is this?")
```

- [ ] **Step 4.2: Run — expect ImportError**

```bash
pytest tests/test_scorers_xcons.py -v
```

- [ ] **Step 4.3: Extend `filtering/scorers.py`**

Append:

```python
from filtering.matchers import max_match


class StudentLike(Protocol):
    def answer_question(self, image, question: str) -> str: ...


def score_xcons(record: dict, image, student: StudentLike, sbert) -> float:
    """Cross-consistency: how well does a frozen pretrained Student's zero-shot
    answer match the pseudo-answer? Uses max(exact, SBERT-cosine).
    """
    answer = record["answer"]
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    predicted = student.answer_question(image, record["question"])
    return max_match(predicted, answer, sbert_model=sbert)
```

- [ ] **Step 4.4: Tests pass**

```bash
pytest tests/test_scorers_xcons.py -v
```

- [ ] **Step 4.5: Commit**

```bash
git add filtering/scorers.py tests/test_scorers_xcons.py
git commit -m "feat(filtering): add cross-consistency scorer using frozen Student"
```

---

## Task 5: `gates.py` — cascade + quantile thresholds

**Files:**
- Create: `filtering/gates.py`
- Test: `tests/test_gates.py`

- [ ] **Step 5.1: Write failing test**

Create `tests/test_gates.py`:

```python
import pytest
from filtering.gates import (
    apply_gates,
    thresholds_from_quantile,
    GateConfig,
    GateThresholds,
    apply_soft_fusion,
)


def test_apply_gates_keeps_when_all_above():
    scores = {"conf": 0.8, "itm": 0.7, "xcons": 0.9}
    keep, reason = apply_gates(scores, GateThresholds(0.5, 0.5, 0.5))
    assert keep is True
    assert reason == "kept"


def test_apply_gates_rejects_on_conf():
    scores = {"conf": 0.1, "itm": 0.9, "xcons": 0.9}
    keep, reason = apply_gates(scores, GateThresholds(0.5, 0.5, 0.5))
    assert keep is False
    assert reason == "conf"


def test_apply_gates_rejects_on_itm_even_if_conf_passes():
    scores = {"conf": 0.9, "itm": 0.1, "xcons": 0.9}
    keep, reason = apply_gates(scores, GateThresholds(0.5, 0.5, 0.5))
    assert (keep, reason) == (False, "itm")


def test_apply_gates_rejects_on_xcons_last():
    scores = {"conf": 0.9, "itm": 0.9, "xcons": 0.1}
    keep, reason = apply_gates(scores, GateThresholds(0.5, 0.5, 0.5))
    assert (keep, reason) == (False, "xcons")


def test_apply_gates_disabled_via_neg_infinity():
    scores = {"conf": -10.0, "itm": -10.0, "xcons": -10.0}
    keep, reason = apply_gates(
        scores, GateThresholds(float("-inf"), float("-inf"), float("-inf"))
    )
    assert keep is True


def test_thresholds_from_quantile_keep_top_75():
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    t = thresholds_from_quantile(values, keep_top=0.75)
    assert t == pytest.approx(0.25, abs=0.01)


def test_thresholds_from_quantile_keep_all():
    values = [0.1, 0.2, 0.3]
    t = thresholds_from_quantile(values, keep_top=1.0)
    assert t == float("-inf")


def test_thresholds_from_quantile_keep_top_1():
    values = [0.1, 0.2, 0.3]
    t = thresholds_from_quantile(values, keep_top=0.0)
    assert t == float("inf")


def test_apply_soft_fusion_stub_raises():
    with pytest.raises(NotImplementedError):
        apply_soft_fusion({"conf": 0.5, "itm": 0.5, "xcons": 0.5}, (1, 1, 1), 0.5)
```

- [ ] **Step 5.2: Run — expect ImportError**

```bash
pytest tests/test_gates.py -v
```

- [ ] **Step 5.3: Implement `filtering/gates.py`**

```python
"""Gate logic (cascade) and quantile-based threshold helpers."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np


@dataclass
class GateConfig:
    enabled: bool = True
    keep_top: float = 0.75  # quantile; 1.0 = keep all, 0.0 = keep none


@dataclass
class GateThresholds:
    tau_conf: float
    tau_itm: float
    tau_xcons: float


def apply_gates(scores: dict, thresholds: GateThresholds) -> Tuple[bool, str]:
    """Cascade: conf -> itm -> xcons. Return (keep, reason).
    A disabled gate is encoded by thresholds = -inf.
    """
    if scores["conf"] < thresholds.tau_conf:
        return False, "conf"
    if scores["itm"] < thresholds.tau_itm:
        return False, "itm"
    if scores["xcons"] < thresholds.tau_xcons:
        return False, "xcons"
    return True, "kept"


def thresholds_from_quantile(values: Sequence[float], keep_top: float) -> float:
    """Convert "keep top X fraction" to an absolute threshold on `values`.

    keep_top=1.0 -> -inf (everything passes)
    keep_top=0.0 -> +inf (nothing passes)
    Otherwise: returns the (1 - keep_top) quantile of values.
    """
    if keep_top >= 1.0:
        return float("-inf")
    if keep_top <= 0.0:
        return float("inf")
    cutoff = 1.0 - keep_top
    return float(np.quantile(np.asarray(values, dtype=np.float64), cutoff))


def apply_soft_fusion(scores: dict, weights, tau: float):
    """Phase 2 (spec §6) — intentionally not implemented in this plan."""
    raise NotImplementedError("Soft fusion is Phase 2; out of scope.")
```

- [ ] **Step 5.4: Tests pass**

```bash
pytest tests/test_gates.py -v
```

- [ ] **Step 5.5: Commit**

```bash
git add filtering/gates.py tests/test_gates.py
git commit -m "feat(filtering): add cascade gate logic and quantile thresholds"
```

---

## Task 6: `io.py` — load / dump records preserving schema

**Files:**
- Create: `filtering/io.py`
- Test: `tests/test_io.py`
- Test: `tests/fixtures/synthetic_data_raw_tiny.json`

- [ ] **Step 6.1: Create tiny fixture**

Create `tests/fixtures/synthetic_data_raw_tiny.json`:

```json
[
  {"question_id": 0, "question": "what is this animal?", "answer": ["dog"], "image": "vg/img_0.jpg", "dataset": "vg", "rationale": null, "gen_logprob": -0.50},
  {"question_id": 1, "question": "how many cats?", "answer": ["2"], "image": "vg/img_1.jpg", "dataset": "vg", "rationale": null, "gen_logprob": -3.10},
  {"question_id": 2, "question": "is it raining?", "answer": ["yes"], "image": "vg/img_2.jpg", "dataset": "vg", "rationale": null, "gen_logprob": -0.20},
  {"question_id": 3, "question": "is the sky blue?", "answer": ["no"], "image": "vg/img_3.jpg", "dataset": "vg", "rationale": null, "gen_logprob": -5.00},
  {"question_id": 4, "question": "what color is the car?", "answer": ["red"], "image": "vg/img_4.jpg", "dataset": "vg", "rationale": null, "gen_logprob": -1.20}
]
```

- [ ] **Step 6.2: Write failing test**

Create `tests/test_io.py`:

```python
import json
import pytest
from pathlib import Path
from filtering.io import load_records, dump_records


FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_data_raw_tiny.json"


def test_load_records_returns_list_of_dicts():
    records = load_records(FIXTURE)
    assert isinstance(records, list)
    assert len(records) == 5
    assert all(isinstance(r, dict) for r in records)


def test_load_records_preserves_all_fields():
    records = load_records(FIXTURE)
    assert set(records[0].keys()) >= {
        "question_id", "question", "answer", "image", "dataset", "gen_logprob"
    }


def test_dump_records_roundtrip(tmp_path):
    records = load_records(FIXTURE)
    out = tmp_path / "out.json"
    dump_records(records, out)
    loaded = json.loads(out.read_text())
    assert loaded == records


def test_dump_records_writes_compact_json(tmp_path):
    records = [{"a": 1}]
    out = tmp_path / "out.json"
    dump_records(records, out)
    text = out.read_text()
    assert "\n" not in text or text.count("\n") <= 1  # compact / single-line list


def test_dump_records_creates_parent_dirs(tmp_path):
    records = [{"a": 1}]
    out = tmp_path / "deep" / "nested" / "out.json"
    dump_records(records, out)
    assert out.exists()
```

- [ ] **Step 6.3: Run — expect ImportError**

```bash
pytest tests/test_io.py -v
```

- [ ] **Step 6.4: Implement `filtering/io.py`**

```python
"""Filesystem I/O for pseudo-label JSON files."""

from __future__ import annotations
import json
from pathlib import Path
from typing import List


def load_records(path) -> List[dict]:
    """Read a JSON file expected to be a list of dicts. No validation —
    consumers are responsible for required fields.
    """
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected JSON list at top level")
    return data


def dump_records(records: List[dict], path) -> None:
    """Write a list of dicts to JSON, creating parent directories as needed.
    Output is compact (no pretty-printing) to match SelTDA's existing format.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(records, f)
```

- [ ] **Step 6.5: Tests pass**

```bash
pytest tests/test_io.py -v
```

- [ ] **Step 6.6: Commit**

```bash
git add filtering/io.py tests/test_io.py tests/fixtures/synthetic_data_raw_tiny.json
git commit -m "feat(filtering): add schema-preserving JSON I/O for pseudo records"
```

---

## Task 7: `report.py` — counts, histograms, sample reasons

**Files:**
- Create: `filtering/report.py`
- Test: `tests/test_report.py`

- [ ] **Step 7.1: Write failing test**

Create `tests/test_report.py`:

```python
from filtering.report import build_filter_report


def make_record(qid, score_conf, score_itm, score_xcons, keep_reason):
    return (
        {
            "question_id": qid,
            "question": f"q{qid}?",
            "answer": [f"a{qid}"],
            "image": f"vg/img_{qid}.jpg",
            "scores": {"conf": score_conf, "itm": score_itm, "xcons": score_xcons},
        },
        keep_reason,
    )


def test_report_counts():
    decisions = [
        make_record(0, 0.9, 0.9, 0.9, "kept"),
        make_record(1, 0.1, 0.9, 0.9, "conf"),
        make_record(2, 0.9, 0.1, 0.9, "itm"),
        make_record(3, 0.9, 0.9, 0.1, "xcons"),
        make_record(4, 0.9, 0.9, 0.9, "kept"),
    ]
    report = build_filter_report(decisions, max_examples=2)
    assert report["counts"]["total_in"] == 5
    assert report["counts"]["kept"] == 2
    assert report["counts"]["dropped"]["conf"] == 1
    assert report["counts"]["dropped"]["itm"] == 1
    assert report["counts"]["dropped"]["xcons"] == 1


def test_report_histograms_have_bins():
    decisions = [
        make_record(i, i / 10.0, i / 10.0, i / 10.0, "kept") for i in range(11)
    ]
    report = build_filter_report(decisions, max_examples=0, n_bins=5)
    assert "histograms" in report
    assert len(report["histograms"]["conf"]) == 5


def test_report_includes_rejected_examples():
    decisions = [
        make_record(1, 0.1, 0.9, 0.9, "conf"),
        make_record(2, 0.2, 0.9, 0.9, "conf"),
        make_record(3, 0.9, 0.1, 0.9, "itm"),
    ]
    report = build_filter_report(decisions, max_examples=10)
    examples = report["rejected_examples"]
    assert {e["reason"] for e in examples} == {"conf", "itm"}
    assert all("question" in e and "answer" in e for e in examples)
```

- [ ] **Step 7.2: Run — expect ImportError**

```bash
pytest tests/test_report.py -v
```

- [ ] **Step 7.3: Implement `filtering/report.py`**

```python
"""Build a JSON-serializable filter report from per-record decisions."""

from __future__ import annotations
from collections import Counter
from typing import Iterable, List, Tuple

import numpy as np


def build_filter_report(
    decisions: Iterable[Tuple[dict, str]],
    max_examples: int = 50,
    n_bins: int = 20,
) -> dict:
    """Aggregate per-record (record, reason) tuples into a serializable report.

    reason: "kept" | "conf" | "itm" | "xcons"
    """
    decisions = list(decisions)
    total_in = len(decisions)

    reasons = Counter(r for _, r in decisions)
    kept = reasons.get("kept", 0)

    score_lists = {"conf": [], "itm": [], "xcons": []}
    for rec, _ in decisions:
        for k in score_lists:
            v = rec.get("scores", {}).get(k)
            if v is not None:
                score_lists[k].append(float(v))

    histograms = {}
    for k, vals in score_lists.items():
        if not vals:
            histograms[k] = []
            continue
        hist, _ = np.histogram(np.asarray(vals, dtype=np.float64), bins=n_bins, range=(0.0, 1.0))
        histograms[k] = [int(x) for x in hist.tolist()]

    rejected_examples: List[dict] = []
    if max_examples > 0:
        for rec, reason in decisions:
            if reason == "kept":
                continue
            rejected_examples.append({
                "question_id": rec.get("question_id"),
                "question": rec.get("question"),
                "answer": rec.get("answer"),
                "image": rec.get("image"),
                "scores": rec.get("scores"),
                "reason": reason,
            })
            if len(rejected_examples) >= max_examples:
                break

    return {
        "counts": {
            "total_in": total_in,
            "kept": kept,
            "dropped": {
                "conf": reasons.get("conf", 0),
                "itm": reasons.get("itm", 0),
                "xcons": reasons.get("xcons", 0),
            },
        },
        "histograms": histograms,
        "rejected_examples": rejected_examples,
    }
```

- [ ] **Step 7.4: Tests pass**

```bash
pytest tests/test_report.py -v
```

- [ ] **Step 7.5: Commit**

```bash
git add filtering/report.py tests/test_report.py
git commit -m "feat(filtering): add report builder with counts, histograms, examples"
```

---

## Task 8: Heavy adapters — CLIP and Student wrappers (real models, slow tests)

**Files:**
- Modify: `filtering/scorers.py` (add `OpenClipAdapter`, `BlipStudentAdapter`)
- Test: `tests/test_scorers_clip.py` (add a `@pytest.mark.slow` test exercising real CLIP)
- Test: `tests/test_scorers_xcons.py` (add slow real-model test only if checkpoint available)

- [ ] **Step 8.1: Append adapters to `filtering/scorers.py`**

```python
from PIL import Image


class OpenClipAdapter:
    """Thin wrapper around open_clip so score_clip_itm can stay framework-agnostic."""

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = "cuda"):
        import open_clip
        import torch
        self._torch = torch
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model = self.model.to(device).eval()

    def embed_image(self, image):
        if isinstance(image, (str, bytes)):
            image = Image.open(image).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            emb = self.model.encode_image(tensor)
        return emb.cpu().numpy().squeeze(0)

    def embed_text(self, text: str):
        tokens = self.tokenizer([text]).to(self.device)
        with self._torch.no_grad():
            emb = self.model.encode_text(tokens)
        return emb.cpu().numpy().squeeze(0)


class BlipStudentAdapter:
    """Wraps a SelTDA BLIP model loaded from a *pretrained* checkpoint and exposes
    a `.answer_question(image, question) -> str` interface for Gate 3.

    IMPORTANT (spec §3.3): the checkpoint MUST be the BLIP pretrained one, NOT a
    student that has already trained on synthetic_data.json.
    """

    def __init__(self, checkpoint: str, image_size: int = 384, device: str = "cuda"):
        import torch
        from models.blip_vqa import blip_vqa  # SelTDA's own VQA model
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode

        self._torch = torch
        self.device = device
        self.model = blip_vqa(pretrained=checkpoint, image_size=image_size, vit="base")
        self.model = self.model.to(device).eval()
        self.preprocess = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ])

    def answer_question(self, image, question: str) -> str:
        if isinstance(image, (str, bytes)):
            image = Image.open(image).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            answer = self.model(tensor, question, train=False, inference="generate")
        return answer[0] if isinstance(answer, list) else str(answer)
```

> **Verification step before continuing:** confirm `models.blip_vqa` exposes the signature shown. Run:
> ```bash
> python -c "from models.blip_vqa import blip_vqa; import inspect; print(inspect.signature(blip_vqa))"
> ```
> Expected: a signature containing `pretrained, image_size, vit` (matching `examples/self_train_synthetic.sh`'s usage indirectly via `train_vqa.py`).
> If different, adjust the wrapper here. **Do not** modify `models/blip_vqa.py`.

- [ ] **Step 8.2: Add a `@pytest.mark.slow` smoke test for CLIP adapter**

Append to `tests/test_scorers_clip.py`:

```python
@pytest.mark.slow
def test_open_clip_adapter_runs_end_to_end(tmp_path):
    from filtering.scorers import OpenClipAdapter, score_clip_itm
    from PIL import Image

    img_path = tmp_path / "red.png"
    Image.new("RGB", (64, 64), color=(255, 0, 0)).save(img_path)

    clip = OpenClipAdapter(device="cpu")
    s = score_clip_itm(
        record={"question": "what color is this?", "answer": ["red"]},
        image=str(img_path),
        clip=clip,
    )
    assert 0.0 <= s <= 1.0
```

- [ ] **Step 8.3: Run fast suite (excludes slow)**

```bash
pytest -m "not slow" -v
```

Expected: all prior tests still pass; new slow test is skipped.

- [ ] **Step 8.4: Run slow CLIP test (downloads weights once)**

```bash
pytest tests/test_scorers_clip.py::test_open_clip_adapter_runs_end_to_end -v
```

Expected: passes after model download. If the runner has no internet, mark this task done and rely on the smoke E2E test in Task 12.

- [ ] **Step 8.5: Commit**

```bash
git add filtering/scorers.py tests/test_scorers_clip.py
git commit -m "feat(filtering): add OpenClip and BLIP-Student real-model adapters"
```

---

## Task 9: `filter_pseudo.py` orchestrator

**Files:**
- Create: `filter_pseudo.py`
- Create: `configs/filter_pseudo.yaml`

- [ ] **Step 9.1: Create default config `configs/filter_pseudo.yaml`**

```yaml
input: datasets/aokvqa/synthetic_data_raw.json
image_root: datasets/coco2017
output: datasets/aokvqa/synthetic_data.json
report: datasets/aokvqa/filter_report.json

gates:
  conf:
    enabled: true
    keep_top: 0.75
  itm:
    enabled: true
    keep_top: 0.75
    clip_model: "ViT-B-32"
    clip_pretrained: "openai"
  xcons:
    enabled: true
    keep_top: 0.75
    student_ckpt: cache/blip_pretrained.pth
    sbert_model: "sentence-transformers/all-MiniLM-L6-v2"
    image_size: 384

scoring_only: false
seed: 42
device: cuda
report_max_examples: 50
log_every: 200
torch_home: null
```

- [ ] **Step 9.2: Implement `filter_pseudo.py`**

```python
"""Entry point: load raw synthetic JSON, score, gate, write filtered + report."""

from __future__ import annotations
import logging
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import cli
from filtering.gates import (
    GateConfig,
    GateThresholds,
    apply_gates,
    thresholds_from_quantile,
)
from filtering.io import load_records, dump_records
from filtering.report import build_filter_report
from filtering.scorers import (
    score_confidence,
    score_clip_itm,
    score_xcons,
    normalize_min_max,
    OpenClipAdapter,
    BlipStudentAdapter,
)

logger = logging.getLogger(__name__)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_image(image_root: Path, image_field: str) -> Path:
    return image_root / image_field


def _run_gate_conf(records, cfg):
    if not cfg.gates.conf.enabled:
        for r in records:
            r.setdefault("scores", {})["conf"] = 1.0
        return float("-inf")
    raw = [score_confidence(r) for r in records]
    normalized = normalize_min_max(raw)
    for r, v in zip(records, normalized):
        r.setdefault("scores", {})["conf"] = 0.0 if v is None else float(v)
    return thresholds_from_quantile(
        [r["scores"]["conf"] for r in records],
        keep_top=cfg.gates.conf.keep_top,
    )


def _run_gate_itm(records, image_root, cfg):
    if not cfg.gates.itm.enabled:
        for r in records:
            r.setdefault("scores", {})["itm"] = 1.0
        return float("-inf")
    clip = OpenClipAdapter(
        model_name=cfg.gates.itm.clip_model,
        pretrained=cfg.gates.itm.clip_pretrained,
        device=cfg.device,
    )
    for i, r in enumerate(tqdm(records, desc="Gate ITM")):
        try:
            image = Image.open(_resolve_image(image_root, r["image"])).convert("RGB")
            s = score_clip_itm(r, image, clip)
        except Exception as e:
            logger.warning("ITM scoring failed for %s: %s", r.get("image"), e)
            s = 0.0
        r.setdefault("scores", {})["itm"] = float(s)
    return thresholds_from_quantile(
        [r["scores"]["itm"] for r in records],
        keep_top=cfg.gates.itm.keep_top,
    )


def _run_gate_xcons(records, image_root, cfg):
    if not cfg.gates.xcons.enabled:
        for r in records:
            r.setdefault("scores", {})["xcons"] = 1.0
        return float("-inf")
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer(cfg.gates.xcons.sbert_model, device=cfg.device)
    student = BlipStudentAdapter(
        checkpoint=cfg.gates.xcons.student_ckpt,
        image_size=cfg.gates.xcons.image_size,
        device=cfg.device,
    )
    for i, r in enumerate(tqdm(records, desc="Gate X-cons")):
        try:
            image = Image.open(_resolve_image(image_root, r["image"])).convert("RGB")
            s = score_xcons(r, image, student, sbert)
        except Exception as e:
            logger.warning("X-cons scoring failed for %s: %s", r.get("image"), e)
            s = 0.0
        r.setdefault("scores", {})["xcons"] = float(s)
    return thresholds_from_quantile(
        [r["scores"]["xcons"] for r in records],
        keep_top=cfg.gates.xcons.keep_top,
    )


def main(args, config):
    _seed_everything(config.seed)
    logging.basicConfig(level=logging.INFO)

    image_root = Path(config.image_root)
    records = load_records(config.input)
    logger.info("Loaded %d raw records from %s", len(records), config.input)

    tau_conf = _run_gate_conf(records, config)
    tau_itm = _run_gate_itm(records, image_root, config)
    tau_xcons = _run_gate_xcons(records, image_root, config)
    thresholds = GateThresholds(tau_conf, tau_itm, tau_xcons)
    logger.info("Thresholds: %s", thresholds)

    decisions = []
    kept_records = []
    for r in records:
        if config.scoring_only:
            decisions.append((r, "kept"))
            kept_records.append(r)
            continue
        keep, reason = apply_gates(r["scores"], thresholds)
        decisions.append((r, reason))
        if keep:
            kept_records.append(r)

    dump_records(kept_records, config.output)
    logger.info("Wrote %d kept records to %s", len(kept_records), config.output)

    report = build_filter_report(decisions, max_examples=config.report_max_examples)
    report["thresholds"] = {
        "conf": tau_conf, "itm": tau_itm, "xcons": tau_xcons,
    }
    report["config"] = dict(config)
    from filtering.io import dump_records as _dump
    Path(config.report).parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(config.report, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Wrote report to %s", config.report)


if __name__ == "__main__":
    args, config = cli.parse_args(default_config_path="./configs/filter_pseudo.yaml")
    cli.setup(args, config)
    main(args, config)
```

- [ ] **Step 9.3: Commit**

```bash
git add filter_pseudo.py configs/filter_pseudo.yaml
git commit -m "feat: add filter_pseudo.py entrypoint and default hydra config"
```

---

## Task 10: `examples/filter_synthetic.sh`

**Files:**
- Create: `examples/filter_synthetic.sh`

- [ ] **Step 10.1: Write script**

```bash
#!/bin/bash
# Filter the raw synthetic data produced by generate_synthetic_data.sh.
# Usage: bash examples/filter_synthetic.sh <dataset_name>
#   e.g. bash examples/filter_synthetic.sh aokvqa
#
# Reads: datasets/<ds>/synthetic_data_raw.json
# Writes: datasets/<ds>/synthetic_data.json   (drop-in for train_vqa.py)
#         datasets/<ds>/filter_report.json
set -euo pipefail
DS="${1:-aokvqa}"

python filter_pseudo.py \
    --config configs/filter_pseudo.yaml \
    --overrides \
        input=datasets/${DS}/synthetic_data_raw.json \
        output=datasets/${DS}/synthetic_data.json \
        report=datasets/${DS}/filter_report.json
```

- [ ] **Step 10.2: chmod and commit**

```bash
chmod +x examples/filter_synthetic.sh
git add examples/filter_synthetic.sh
git commit -m "feat(examples): add filter_synthetic.sh runner"
```

---

## Task 11: Extend `VQARecord` with optional fields (in `generate_questions.py`)

**Files:**
- Modify: `generate_questions.py` (additive only)

- [ ] **Step 11.1: Add optional fields to the `attrs` class**

Open `generate_questions.py`. In the `VQARecord` definition (currently lines ~72–125), add 2 fields **at the end** of the `@attrs.define` body, **before** `build_from_raw_model_output`. Keep all existing fields and methods unchanged.

Replace this exact existing block:

```python
@attrs.define
class VQARecord:
    question_id: int
    question: str
    answer: List[str]
    image: str
    dataset: str
    rationale: Optional[str] = None
```

With:

```python
@attrs.define
class VQARecord:
    question_id: int
    question: str
    answer: List[str]
    image: str
    dataset: str
    rationale: Optional[str] = None
    gen_logprob: Optional[float] = None
    scores: Optional[dict] = None
```

Also update the imports at the top of the file to include `Dict` if not already imported:

```python
from typing import List, Optional, Dict
```

- [ ] **Step 11.2: Make sure existing tests still pass**

```bash
pytest tests/test_generate_questions.py -v
```

Expected: all existing 5 tests pass (the new fields default to `None`).

- [ ] **Step 11.3: Commit**

```bash
git add generate_questions.py
git commit -m "feat(generate): add optional gen_logprob and scores to VQARecord"
```

---

## Task 12: Compute and persist `gen_logprob` in `generate_questions.py`

**Files:**
- Modify: `generate_questions.py` (additive only)
- Test: extend `tests/test_generate_questions.py` if appropriate, or add `tests/test_generate_logprob.py`

- [ ] **Step 12.1: Inspect current `model.generate` call**

Read lines 280–320 of `generate_questions.py`. The block under `for _ in range(config.questions_per_image):` currently looks like:

```python
with torch.no_grad():
    outputs = model.generate(
        images,
        sample=True,
        top_p=config.top_p,
        max_length=config.max_length,
        min_length=config.min_length,
    )

for idx, (model_output, image_path) in enumerate(zip(outputs, image_paths)):
    try:
        record = VQARecord.build_from_raw_model_output(...)
```

We need the mean log-prob of each `outputs[i]` under `model`, **without** touching `models/blip.py`.

- [ ] **Step 12.2: Add a helper `_compute_mean_logprob` inside `generate_questions.py`**

Add this helper near the top of the file, after imports:

```python
def _compute_mean_logprob(model, images, texts) -> list:
    """Teacher-forced mean log-prob of each `texts[i]` conditioned on `images[i]`.

    Uses the public BLIP-decoder forward in `models/blip.py` without modifying it:
    BLIP's text decoder accepts an `input_ids` tensor and returns logits; we
    request logits from the visual encoder + text decoder via the same call path
    used by `model.generate` internally, but with sampling disabled.
    """
    import torch
    import torch.nn.functional as F

    # Tokenize text with the model's tokenizer (BLIP exposes `model.tokenizer`).
    tokenizer = model.tokenizer
    enc = tokenizer(
        list(texts),
        padding="longest",
        return_tensors="pt",
        truncation=True,
        max_length=64,
    ).to(images.device)
    input_ids = enc.input_ids
    attention_mask = enc.attention_mask

    with torch.no_grad():
        image_embeds = model.visual_encoder(images)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=images.device)

        out = model.text_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            labels=input_ids.masked_fill(input_ids == tokenizer.pad_token_id, -100),
            return_dict=True,
        )
        # `out.loss` is mean NLL across non-pad tokens (averaged over batch).
        # Recompute per-sample mean log-prob to get one number per record.
        logits = out.logits  # (B, T, V)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous().float()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        gathered = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        per_sample = (gathered * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1.0)

    return per_sample.cpu().tolist()
```

> **Verification before relying on this:** confirm BLIP's decoder model exposes `model.tokenizer`, `model.visual_encoder`, and `model.text_decoder` (or equivalent names). Run:
> ```bash
> python -c "from models.blip import blip_decoder; m = blip_decoder(pretrained='', image_size=384); print([a for a in dir(m) if not a.startswith('_')][:30])"
> ```
> If the names differ (e.g. `text_encoder` instead of `text_decoder`), adjust the helper. Do not modify `models/blip.py` itself.

- [ ] **Step 12.3: Wire the helper into the generation loop**

Replace the existing block:

```python
with torch.no_grad():
    outputs = model.generate(
        images,
        sample=True,
        top_p=config.top_p,
        max_length=config.max_length,
        min_length=config.min_length,
    )

for idx, (model_output, image_path) in enumerate(zip(outputs, image_paths)):
```

With:

```python
with torch.no_grad():
    outputs = model.generate(
        images,
        sample=True,
        top_p=config.top_p,
        max_length=config.max_length,
        min_length=config.min_length,
    )

try:
    logprobs = _compute_mean_logprob(model, images, outputs)
except Exception as e:
    logger.warning("Failed to compute logprob, falling back to None: %s", e)
    logprobs = [None] * len(outputs)

for idx, (model_output, image_path, lp) in enumerate(zip(outputs, image_paths, logprobs)):
```

And inside the `try` block, after `record.question_id = idx`, add:

```python
record.gen_logprob = lp
```

- [ ] **Step 12.4: Add a regression test pinning the schema**

Create `tests/test_generate_logprob.py`:

```python
from generate_questions import VQARecord


def test_vqa_record_accepts_gen_logprob():
    r = VQARecord(
        question_id=0,
        question="q?",
        answer=["a"],
        image="vg/img.jpg",
        dataset="vg",
        gen_logprob=-1.23,
    )
    assert r.gen_logprob == -1.23


def test_vqa_record_default_gen_logprob_is_none():
    r = VQARecord(
        question_id=0,
        question="q?",
        answer=["a"],
        image="vg/img.jpg",
        dataset="vg",
    )
    assert r.gen_logprob is None
    assert r.scores is None
```

- [ ] **Step 12.5: Run regression + existing parsing tests**

```bash
pytest tests/test_generate_logprob.py tests/test_generate_questions.py -v
```

Expected: all pass.

- [ ] **Step 12.6: Commit**

```bash
git add generate_questions.py tests/test_generate_logprob.py
git commit -m "feat(generate): compute and persist teacher mean log-prob per pseudo sample"
```

---

## Task 13: End-to-end smoke test on tiny fixture

**Files:**
- Create: `tests/test_filter_pseudo_smoke.py`
- Create: `tests/fixtures/images/img_0.jpg` ... `img_4.jpg` (created by the test, not committed as binaries)

- [ ] **Step 13.1: Write the smoke test**

Create `tests/test_filter_pseudo_smoke.py`:

```python
"""End-to-end smoke: tiny fixture + stubbed scorers go through the orchestrator."""
import json
from pathlib import Path

import pytest


@pytest.fixture
def tiny_setup(tmp_path):
    from PIL import Image
    fixture_src = Path(__file__).parent / "fixtures" / "synthetic_data_raw_tiny.json"
    records = json.loads(fixture_src.read_text())

    image_root = tmp_path / "images"
    (image_root / "vg").mkdir(parents=True)
    for i in range(5):
        Image.new("RGB", (32, 32), color=(i * 50, 100, 100)).save(
            image_root / "vg" / f"img_{i}.jpg"
        )

    raw_path = tmp_path / "synthetic_data_raw.json"
    raw_path.write_text(json.dumps(records))
    return {
        "image_root": str(image_root),
        "input": str(raw_path),
        "output": str(tmp_path / "synthetic_data.json"),
        "report": str(tmp_path / "filter_report.json"),
    }


def test_orchestrator_with_all_gates_disabled_keeps_all(tiny_setup, monkeypatch):
    """When all gates are disabled, every record must pass through unchanged."""
    from omegaconf import OmegaConf
    import filter_pseudo

    config = OmegaConf.create({
        "input": tiny_setup["input"],
        "image_root": tiny_setup["image_root"],
        "output": tiny_setup["output"],
        "report": tiny_setup["report"],
        "gates": {
            "conf":  {"enabled": False, "keep_top": 1.0},
            "itm":   {"enabled": False, "keep_top": 1.0, "clip_model": "x", "clip_pretrained": "y"},
            "xcons": {"enabled": False, "keep_top": 1.0, "student_ckpt": "x",
                      "sbert_model": "y", "image_size": 384},
        },
        "scoring_only": False,
        "seed": 0,
        "device": "cpu",
        "report_max_examples": 10,
        "log_every": 100,
        "torch_home": None,
    })

    class Args:
        output_dir = str(Path(tiny_setup["output"]).parent)
        result_dir = str(Path(tiny_setup["output"]).parent)
        device = "cpu"
        seed = 0

    filter_pseudo.main(Args(), config)

    out = json.loads(Path(tiny_setup["output"]).read_text())
    assert len(out) == 5

    report = json.loads(Path(tiny_setup["report"]).read_text())
    assert report["counts"]["total_in"] == 5
    assert report["counts"]["kept"] == 5


def test_orchestrator_confidence_only_drops_low_logprob(tiny_setup):
    """Enable only Gate 1 with keep_top=0.6 -> 2 lowest-logprob records dropped."""
    from omegaconf import OmegaConf
    import filter_pseudo

    config = OmegaConf.create({
        "input": tiny_setup["input"],
        "image_root": tiny_setup["image_root"],
        "output": tiny_setup["output"],
        "report": tiny_setup["report"],
        "gates": {
            "conf":  {"enabled": True,  "keep_top": 0.6},
            "itm":   {"enabled": False, "keep_top": 1.0, "clip_model": "x", "clip_pretrained": "y"},
            "xcons": {"enabled": False, "keep_top": 1.0, "student_ckpt": "x",
                      "sbert_model": "y", "image_size": 384},
        },
        "scoring_only": False,
        "seed": 0,
        "device": "cpu",
        "report_max_examples": 10,
        "log_every": 100,
        "torch_home": None,
    })

    class Args:
        output_dir = str(Path(tiny_setup["output"]).parent)
        result_dir = str(Path(tiny_setup["output"]).parent)
        device = "cpu"
        seed = 0

    filter_pseudo.main(Args(), config)
    out = json.loads(Path(tiny_setup["output"]).read_text())
    # Fixture has logprobs: -0.2, -0.5, -1.2, -3.1, -5.0  -> top 60% = 3 kept.
    assert len(out) == 3
    kept_qids = sorted(r["question_id"] for r in out)
    assert kept_qids == [0, 2, 4]  # logprobs -0.5, -0.2, -1.2 → top 3
```

- [ ] **Step 13.2: Run smoke test**

```bash
pytest tests/test_filter_pseudo_smoke.py -v
```

Expected: both tests PASS. If the second test's kept_qids differs, audit the quantile math — but the assertion is correct given the fixture.

- [ ] **Step 13.3: Commit**

```bash
git add tests/test_filter_pseudo_smoke.py
git commit -m "test(filter): end-to-end smoke test with tiny fixture"
```

---

## Task 14: Hard-constraint enforcement script

**Files:**
- Create: `scripts/check_invariants.sh`

- [ ] **Step 14.1: Write the guard script**

Create `scripts/check_invariants.sh`:

```bash
#!/bin/bash
# Enforce the spec §1.4 hard constraint: filtering work must not modify training/eval code.
# Usage: bash scripts/check_invariants.sh [<base-ref>]
# Default base ref: origin/main
set -euo pipefail
BASE="${1:-origin/main}"

FORBIDDEN_REGEX='^(train_vqa\.py|train_vqg\.py|data/|models/|vqa_eval_tools/|.*_eval\.py|examples/self_train_synthetic\.sh|examples/evaluate\.sh|examples/train_teacher\.sh|configs/(aokvqa|pathvqa|okvqa|advqa|artvqa|rsvqa|vqa)\.yaml)$'

CHANGED=$(git diff --name-only "${BASE}"...HEAD || true)
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
```

- [ ] **Step 14.2: Run the guard**

```bash
chmod +x scripts/check_invariants.sh
bash scripts/check_invariants.sh origin/main
```

Expected: `OK: no forbidden files modified.` Only `generate_questions.py` and brand-new files should appear in the changed list.

- [ ] **Step 14.3: Commit**

```bash
git add scripts/check_invariants.sh
git commit -m "chore: add invariant guard for filtering branch"
```

---

## Task 15: Full test suite + reproducibility check

- [ ] **Step 15.1: Run full fast suite**

```bash
pytest -m "not slow" -v
```

Expected: all green. No warnings about modified files in `models/`, `data/`, etc.

- [ ] **Step 15.2: Run the invariant guard once more**

```bash
bash scripts/check_invariants.sh origin/main
```

Expected: `OK: no forbidden files modified.`

- [ ] **Step 15.3: Dry-run the orchestrator on the tiny fixture from the CLI**

```bash
mkdir -p datasets/tiny/vg
cp tests/fixtures/synthetic_data_raw_tiny.json datasets/tiny/synthetic_data_raw.json
python - <<'PY'
from PIL import Image
import os
os.makedirs("datasets/tiny/vg", exist_ok=True)
for i in range(5):
    Image.new("RGB", (32, 32), color=(i * 50, 100, 100)).save(f"datasets/tiny/vg/img_{i}.jpg")
PY

python filter_pseudo.py \
    --config configs/filter_pseudo.yaml \
    --overrides \
        input=datasets/tiny/synthetic_data_raw.json \
        image_root=datasets/tiny \
        output=datasets/tiny/synthetic_data.json \
        report=datasets/tiny/filter_report.json \
        gates.conf.enabled=false \
        gates.itm.enabled=false \
        gates.xcons.enabled=false \
        device=cpu
```

Expected: writes `datasets/tiny/synthetic_data.json` (5 records, identical to input) and `datasets/tiny/filter_report.json` (counts.kept = 5).

- [ ] **Step 15.4: No commit needed for a dry run; clean up**

```bash
rm -rf datasets/tiny
```

---

## Verification before declaring Phase 1 code done

| DoD item (spec §8) | Where verified |
|---|---|
| `filter_pseudo.py` runs end-to-end producing a `train_vqa.py`-compatible JSON | Task 13, Task 15.3 |
| Ablation table runnable (8 variants × 2 datasets) | Implicit: each variant is a CLI override; no code change needed |
| `filter_report.json` populated | Task 7 + Task 13 |
| Unit tests green for `scorers`, `matchers`, `gates` | Tasks 1–5, 7 |
| Spec §1.4 hard constraint held | Task 14 + Task 15.2 |

Running the actual A-OKVQA / PathVQA ablations is research execution work, **out of scope** for this implementation plan. Those runs are gated on (a) reproducing the SelTDA baseline and (b) generating raw synthetic on Vast.ai — both already documented by the upstream SelTDA repo, no code changes needed.

---

## Self-Review Summary

**Spec coverage** (against `docs/superpowers/specs/2026-05-21-pseudo-label-filtering-design.md`):

| Spec section | Covered by |
|---|---|
| §1.1 Functional req. 1 (3 scores)        | Tasks 2, 3, 4 |
| §1.1 req. 2 (per-gate thresholds, cascade) | Task 5 |
| §1.1 req. 3 (train_vqa-compatible JSON)  | Task 6, 9 |
| §1.1 req. 4 (filter_report.json)         | Task 7, 9 |
| §1.1 req. 5 (scoring-only mode)          | Task 9 (`scoring_only` config) |
| §1.4 hard constraint (no training edits) | Task 14 + 15.2 |
| §3.1 Gate 1 confidence                   | Tasks 2, 12 |
| §3.2 Gate 2 CLIP-ITM                     | Tasks 3, 8 |
| §3.3 Gate 3 cross-consistency            | Tasks 1, 4, 8 |
| §3.4 cascade logic + 8 ablation variants | Task 5, 9 |
| §4.1 new files                           | Tasks 1–10 |
| §4.2 generate_questions.py changes       | Tasks 11, 12 |
| §4.3 pipeline shell                      | Task 10 |
| §4.4 config schema                       | Task 9 |
| §6 Phase 2 stub                          | Task 5 (`apply_soft_fusion` raises NotImplementedError) |
| §8 DoD                                   | Tasks 13, 14, 15 |

**Placeholder scan**: searched for "TBD", "TODO", "implement later", "add validation", "handle edge cases" — none present. Every code step shows the actual code.

**Type consistency**: `GateThresholds`, `GateConfig`, `score_*`, `apply_gates`, `OpenClipAdapter`, `BlipStudentAdapter` names are used consistently across Tasks 5, 8, 9 and the test files. `gen_logprob` field name is identical in Tasks 11, 12, 13 fixtures, and `filter_pseudo.py`. `scores` dict keys (`conf`/`itm`/`xcons`) are identical everywhere.

**Two flagged verifications** (Tasks 8.1 and 12.2) require runtime confirmation of BLIP's public attribute names; both have explicit verification commands and a "don't modify models/" fallback note.
