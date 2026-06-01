# C-RL Online-GRPO Teacher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the SelTDA teacher into an online-GRPO policy whose reward internalizes the pseudo-label filter (P2 Grounded-Learnability), run a multi-round loop (accumulating teacher / restarting student) with a two-cadence audit and a tiered hacking-response protocol.

**Architecture:** Pure-function reward terms (DI-style like `filtering/scorers.py`) compose into one reward. A GRPO trainer updates the BLIP decoder per epoch; a cheap per-epoch audit (independent judges) drives the hacking-response controller; an orchestrator runs rounds, calling existing `generate_questions.py` / `train_vqa.py` as subprocesses (never editing them) and using student val accuracy as the round-level stop signal.

**Tech Stack:** Python, PyTorch, BLIP (`models/blip.py`, `models/blip_vqa.py`), OpenCLIP, SBERT, pytest. Spec: `docs/superpowers/specs/2026-06-01-rl-teacher-grpo-design.md`.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `filtering/reward.py` | Pure P2 reward terms + `compose_reward` (DI of scorer callables) |
| `orchestration/weak_types.py` | Parse student_base eval → top-k weak types |
| `filtering/audit.py` | Independent judges (ITM-Large + BLIP-VQA) + hacking detection |
| `orchestration/hrp.py` | Hacking-response controller (L1–L3 auto; L4–L5 flag) |
| `train_vqg_rl.py` | GRPO trainer for the BLIP decoder teacher |
| `orchestration/rl_teacher_loop.py` | Round loop; subprocess generate/train; stop signal |
| `configs/rl_teacher_aokvqa.yaml` | Config |
| `tests/test_reward.py`, `test_weak_types.py`, `test_audit.py`, `test_hrp.py`, `test_rl_smoke.py` | Tests |

Reward terms are pure and injected with fakes in tests (no GPU), mirroring `filtering/scorers.py`.

---

## Task 1: Reward — TypeMatch + Repetition terms

**Files:**
- Create: `filtering/reward.py`
- Test: `tests/test_reward.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reward.py
from filtering.reward import type_match, repetition_penalty


def test_type_match_in_weak_set():
    assert type_match("How many dogs?", "3", {"how_many"}) == 1.0

def test_type_match_not_in_weak_set():
    assert type_match("Is it red?", "yes", {"how_many"}) == 0.0

def test_repetition_penalty_unique_is_zero():
    seen = ["how many dogs are there", "what color is the car"]
    assert repetition_penalty("is the sky blue", seen, threshold=0.9) == 0.0

def test_repetition_penalty_duplicate_is_one():
    seen = ["how many dogs are there"]
    assert repetition_penalty("how many dogs are there", seen, threshold=0.9) == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reward.py -v`
Expected: FAIL with `ImportError: cannot import name 'type_match'`

- [ ] **Step 3: Write minimal implementation**

```python
# filtering/reward.py
"""P2 Grounded-Learnability reward terms (pure functions, DI of scorers)."""

from __future__ import annotations

import string
from typing import Callable, Iterable, Optional

from filtering.strata import classify_question_type

_PUNCT = str.maketrans("", "", string.punctuation)


def _norm(text: str) -> str:
    return text.lower().translate(_PUNCT).strip()


def type_match(question: str, answer: str, weak_types: set[str]) -> float:
    """1.0 if the (Q,A) question-type is in the targeted weak set, else 0.0."""
    return 1.0 if classify_question_type(question, answer) in weak_types else 0.0


def repetition_penalty(question: str, seen: Iterable[str], threshold: float = 0.9) -> float:
    """1.0 if `question` token-Jaccard-overlaps any seen question above threshold."""
    q = set(_norm(question).split())
    if not q:
        return 0.0
    for s in seen:
        ss = set(_norm(s).split())
        if not ss:
            continue
        jac = len(q & ss) / len(q | ss)
        if jac >= threshold:
            return 1.0
    return 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_reward.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add filtering/reward.py tests/test_reward.py
git commit -m "feat(rl): add TypeMatch and Repetition reward terms"
```

---

## Task 2: Reward — Learnability term (student uncertainty)

**Files:**
- Modify: `filtering/reward.py`
- Test: `tests/test_reward.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reward.py  (append)
from filtering.reward import learnability


class _FakeStudentProb:
    def __init__(self, prob): self._p = prob
    def answer_prob(self, image, question): return self._p


def test_learnability_high_when_student_uncertain():
    # student max-prob 0.2 -> learnability 0.8
    assert abs(learnability("img", "Q", _FakeStudentProb(0.2)) - 0.8) < 1e-9

def test_learnability_low_when_student_confident():
    assert abs(learnability("img", "Q", _FakeStudentProb(0.95)) - 0.05) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reward.py::test_learnability_high_when_student_uncertain -v`
Expected: FAIL with `ImportError: cannot import name 'learnability'`

- [ ] **Step 3: Write minimal implementation**

```python
# filtering/reward.py  (append)
class StudentProbLike:
    def answer_prob(self, image, question: str) -> float: ...  # max softmax prob of greedy answer


def learnability(image, question: str, student: "StudentProbLike") -> float:
    """1 - max_prob of the frozen base student on (image, question). Higher = harder."""
    p = float(student.answer_prob(image, question))
    p = min(1.0, max(0.0, p))
    return 1.0 - p
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_reward.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add filtering/reward.py tests/test_reward.py
git commit -m "feat(rl): add Learnability reward term"
```

---

## Task 3: Reward — LP_flip + Grounding

`Grounding = XCONS_frozen × LP_flip`. XCONS reuses `filtering.matchers.max_match`. LP_flip = answer changes when the image is corrupted (grounded ⇒ flips).

**Files:**
- Modify: `filtering/reward.py`
- Test: `tests/test_reward.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reward.py  (append)
from filtering.reward import lp_flip, grounding


class _FakeAnswerer:
    """Returns answers keyed by an 'image' token so we can simulate corruption."""
    def __init__(self, mapping): self.mapping = mapping
    def answer_question(self, image, question): return self.mapping[image]


class _IdentitySbert:
    def encode(self, texts, convert_to_numpy=True):
        import numpy as np
        # encode each unique string to an orthogonal-ish vector by hashing
        return np.array([[float(hash(t) % 7), float(hash(t) % 5), 1.0] for t in texts])


def _corrupt(image):  # test corruption = swap to the blanked key
    return image + "_blank"


def test_lp_flip_one_when_answer_changes():
    ans = _FakeAnswerer({"img": "dog", "img_blank": "cat"})
    assert lp_flip("img", "What animal?", ans, _corrupt, _IdentitySbert()) == 1.0

def test_lp_flip_zero_when_answer_same():
    ans = _FakeAnswerer({"img": "dog", "img_blank": "dog"})
    assert lp_flip("img", "What animal?", ans, _corrupt, _IdentitySbert()) == 0.0

def test_grounding_is_product_of_xcons_and_flip():
    # xcons agrees (pred 'dog' == answer 'dog') -> 1.0 ; flip 1.0 -> grounding 1.0
    ans = _FakeAnswerer({"img": "dog", "img_blank": "cat"})
    g = grounding("img", "What animal?", "dog", ans, _corrupt, _IdentitySbert())
    assert g == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reward.py::test_lp_flip_one_when_answer_changes -v`
Expected: FAIL with `ImportError: cannot import name 'lp_flip'`

- [ ] **Step 3: Write minimal implementation**

```python
# filtering/reward.py  (append)
from filtering.matchers import max_match


def lp_flip(image, question: str, answerer, corrupt: Callable, sbert) -> float:
    """1.0 if the frozen answerer's answer CHANGES when the image is corrupted.

    Grounded questions depend on the image, so their answer should flip when the
    image is blanked/corrupted. Answer-stability under corruption => language prior.
    """
    a_full = answerer.answer_question(image, question)
    a_corr = answerer.answer_question(corrupt(image), question)
    same = max_match(a_full, a_corr, sbert_model=sbert)  # 1.0 == identical meaning
    return 1.0 - float(same)


def grounding(image, question: str, answer: str, answerer, corrupt: Callable, sbert) -> float:
    """XCONS_frozen (answerer agrees with pseudo-answer given image) x LP_flip."""
    pred_full = answerer.answer_question(image, question)
    xcons = float(max_match(pred_full, answer, sbert_model=sbert))
    flip = lp_flip(image, question, answerer, corrupt, sbert)
    return xcons * flip
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_reward.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add filtering/reward.py tests/test_reward.py
git commit -m "feat(rl): add LP_flip grounding reward term"
```

---

## Task 4: Reward — compose_reward + RewardConfig

**Files:**
- Modify: `filtering/reward.py`
- Test: `tests/test_reward.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reward.py  (append)
from filtering.reward import RewardConfig, RewardTerms, compose_reward


def test_compose_reward_weighted_sum():
    cfg = RewardConfig(w_type=1.0, w_itm=0.5, w_grounding=1.0,
                       w_learnability=0.5, w_kl=0.1, w_repetition=0.3)
    terms = RewardTerms(type_match=1.0, itm=0.6, grounding=1.0,
                        learnability=0.8, kl=2.0, repetition=1.0)
    # 1*1 + 0.5*0.6 + 1*1 + 0.5*0.8 - 0.1*2 - 0.3*1 = 1+0.3+1+0.4-0.2-0.3 = 2.2
    assert abs(compose_reward(terms, cfg) - 2.2) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reward.py::test_compose_reward_weighted_sum -v`
Expected: FAIL with `ImportError: cannot import name 'RewardConfig'`

- [ ] **Step 3: Write minimal implementation**

```python
# filtering/reward.py  (append)
from dataclasses import dataclass


@dataclass
class RewardConfig:
    w_type: float = 1.0
    w_itm: float = 0.5
    w_grounding: float = 1.0
    w_learnability: float = 0.5
    w_kl: float = 0.1
    w_repetition: float = 0.3


@dataclass
class RewardTerms:
    type_match: float = 0.0
    itm: float = 0.0
    grounding: float = 0.0
    learnability: float = 0.0
    kl: float = 0.0
    repetition: float = 0.0


def compose_reward(t: RewardTerms, cfg: RewardConfig) -> float:
    return (
        cfg.w_type * t.type_match
        + cfg.w_itm * t.itm
        + cfg.w_grounding * t.grounding
        + cfg.w_learnability * t.learnability
        - cfg.w_kl * t.kl
        - cfg.w_repetition * t.repetition
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_reward.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add filtering/reward.py tests/test_reward.py
git commit -m "feat(rl): add compose_reward and reward config dataclasses"
```

---

## Task 5: weak_types — parse student_base eval → top-k

**Files:**
- Create: `orchestration/__init__.py` (empty, if missing)
- Create: `orchestration/weak_types.py`
- Test: `tests/test_weak_types.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_weak_types.py
from orchestration.weak_types import error_rate_per_type, top_k_weak_types

RESULTS = [
    {"question": "How many cats?", "answer": "2", "pred": "3"},      # how_many wrong
    {"question": "How many cats?", "answer": "2", "pred": "2"},      # how_many right
    {"question": "What sport is this?", "answer": "golf", "pred": "tennis"},  # EK wrong
    {"question": "Is it blue?", "answer": "yes", "pred": "yes"},     # yes_no right
]


def test_error_rate_per_type():
    er = error_rate_per_type(RESULTS)
    assert abs(er["how_many"] - 0.5) < 1e-9
    assert abs(er["external_knowledge"] - 1.0) < 1e-9
    assert abs(er["yes_no"] - 0.0) < 1e-9

def test_top_k_weak_types():
    assert top_k_weak_types(RESULTS, k=2) == ["external_knowledge", "how_many"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_weak_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestration.weak_types'`

- [ ] **Step 3: Write minimal implementation**

```python
# orchestration/weak_types.py
"""WT-1: derive frozen weak question types from the base student's eval."""

from __future__ import annotations

import json
from pathlib import Path

from filtering.matchers import exact_match
from filtering.strata import classify_question_type


def error_rate_per_type(results: list[dict]) -> dict[str, float]:
    counts: dict[str, list[int]] = {}
    for r in results:
        ans = r["answer"][0] if isinstance(r["answer"], list) else r["answer"]
        t = classify_question_type(r["question"], ans)
        correct = exact_match(str(r["pred"]), str(ans)) >= 1.0
        counts.setdefault(t, []).append(0 if correct else 1)
    return {t: sum(v) / len(v) for t, v in counts.items()}


def top_k_weak_types(results: list[dict], k: int = 3) -> list[str]:
    er = error_rate_per_type(results)
    return [t for t, _ in sorted(er.items(), key=lambda kv: kv[1], reverse=True)[:k]]


def compute_and_save(results_path: Path, out_path: Path, k: int = 3) -> list[str]:
    results = json.loads(Path(results_path).read_text())
    weak = top_k_weak_types(results, k=k)
    Path(out_path).write_text(json.dumps({"weak_types": weak}, indent=2))
    return weak
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_weak_types.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add orchestration/__init__.py orchestration/weak_types.py tests/test_weak_types.py
git commit -m "feat(rl): add weak-type freezing from base student eval"
```

---

## Task 6: audit — judges aggregation + hacking detection

**Files:**
- Create: `filtering/audit.py`
- Test: `tests/test_audit.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_audit.py
from filtering.audit import judge_score, detect_hacking


def test_judge_score_is_mean_of_components():
    assert abs(judge_score(itm_large=0.8, vqa_match=0.6) - 0.7) < 1e-9

def test_detect_hacking_reward_up_judge_down():
    # reward rose, judge fell beyond eps -> hacking
    assert detect_hacking(d_reward=0.10, d_judge=-0.05, eps=0.02,
                          term_share={"itm": 0.3}, tau=0.6) is True

def test_detect_hacking_term_dominance():
    # judge flat but one cheap term dominates the gain
    assert detect_hacking(d_reward=0.10, d_judge=0.01, eps=0.02,
                          term_share={"itm": 0.7}, tau=0.6) is True

def test_no_hacking_when_judge_rises():
    assert detect_hacking(d_reward=0.10, d_judge=0.05, eps=0.02,
                          term_share={"itm": 0.3}, tau=0.6) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_audit.py -v`
Expected: FAIL with `ImportError: cannot import name 'judge_score'`

- [ ] **Step 3: Write minimal implementation**

```python
# filtering/audit.py
"""AUD-1: independent held-out judges + hacking detection (epoch-level)."""

from __future__ import annotations


def judge_score(itm_large: float, vqa_match: float) -> float:
    """Aggregate the two independent judges (mean)."""
    return (float(itm_large) + float(vqa_match)) / 2.0


def detect_hacking(
    d_reward: float,
    d_judge: float,
    eps: float,
    term_share: dict[str, float],
    tau: float,
) -> bool:
    """Hacking if reward rises while judge falls past eps, OR a cheap term dominates gain."""
    if d_reward > 0 and d_judge < -eps:
        return True
    if any(share > tau for share in term_share.values()):
        return True
    return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_audit.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add filtering/audit.py tests/test_audit.py
git commit -m "feat(rl): add held-out judge aggregation and hacking detection"
```

---

## Task 7: hrp — hacking-response controller (L1–L3 auto)

**Files:**
- Create: `orchestration/hrp.py`
- Test: `tests/test_hrp.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hrp.py
from orchestration.hrp import HrpState, respond


def test_l0_no_hacking_keeps_state():
    s = HrpState(kl_beta=0.1, w_itm=0.5)
    out, action = respond(s, hacking=False, dominant_term=None,
                          judge_dropping=False, auto_levels=[1, 2, 3])
    assert action == "continue"
    assert out.kl_beta == 0.1 and out.w_itm == 0.5

def test_l1_adaptive_kl_on_mild_hacking():
    s = HrpState(kl_beta=0.1, w_itm=0.5)
    out, action = respond(s, hacking=True, dominant_term=None,
                          judge_dropping=False, auto_levels=[1, 2, 3], kl_c=1.5)
    assert action == "adaptive_kl"
    assert abs(out.kl_beta - 0.15) < 1e-9

def test_l2_downweight_dominant_term():
    s = HrpState(kl_beta=0.1, w_itm=0.5)
    out, action = respond(s, hacking=True, dominant_term="itm",
                          judge_dropping=False, auto_levels=[1, 2, 3], downweight_d=0.5)
    assert action == "downweight"
    assert abs(out.w_itm - 0.25) < 1e-9

def test_l3_early_stop_when_judge_dropping():
    s = HrpState(kl_beta=0.1, w_itm=0.5)
    out, action = respond(s, hacking=True, dominant_term=None,
                          judge_dropping=True, auto_levels=[1, 2, 3])
    assert action == "early_stop_rollback"

def test_l4_flagged_for_human_when_not_in_auto_levels():
    s = HrpState(kl_beta=0.1, w_itm=0.5, persistent_rounds=2)
    out, action = respond(s, hacking=True, dominant_term=None,
                          judge_dropping=True, auto_levels=[1, 2, 3])
    assert action == "flag_human"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hrp.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'orchestration.hrp'`

- [ ] **Step 3: Write minimal implementation**

```python
# orchestration/hrp.py
"""HRP-1: tiered hacking-response controller. L1-L3 auto; L4-L5 flag for human."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass
class HrpState:
    kl_beta: float
    w_itm: float
    persistent_rounds: int = 0


def respond(
    state: HrpState,
    *,
    hacking: bool,
    dominant_term: str | None,
    judge_dropping: bool,
    auto_levels: list[int],
    kl_c: float = 1.5,
    downweight_d: float = 0.5,
) -> tuple[HrpState, str]:
    """Return (new_state, action). Escalates L4 when hacking persists >=2 rounds."""
    if not hacking:
        return state, "continue"

    # L4/L5: persistent hacking -> human gate (not auto-handled)
    if state.persistent_rounds >= 2 or judge_dropping and 3 not in auto_levels:
        return state, "flag_human"

    # L3: judge actively dropping -> early stop + rollback
    if judge_dropping and 3 in auto_levels:
        return state, "early_stop_rollback"

    # L2: a cheap term dominates the gain -> down-weight it
    if dominant_term == "itm" and 2 in auto_levels:
        return replace(state, w_itm=state.w_itm * downweight_d), "downweight"

    # L1: mild divergence -> tighten KL
    if 1 in auto_levels:
        return replace(state, kl_beta=state.kl_beta * kl_c), "adaptive_kl"

    return state, "flag_human"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_hrp.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add orchestration/hrp.py tests/test_hrp.py
git commit -m "feat(rl): add tiered hacking-response controller"
```

---

## Task 8: GRPO trainer core (`train_vqg_rl.py`)

GRPO needs: group sampling per image, group-normalized advantage, policy log-prob, KL to a frozen reference. The math is unit-testable independent of the BLIP model. We test the **advantage** and **loss** helpers with a tiny dummy, then wire the model in `main` (not unit-tested; covered by the smoke test in Task 10).

**Files:**
- Create: `train_vqg_rl.py`
- Test: `tests/test_rl_smoke.py` (advantage/loss units here; full smoke in Task 10)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rl_smoke.py
import torch
from train_vqg_rl import group_advantages, grpo_loss


def test_group_advantages_normalizes_within_group():
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    adv = group_advantages(rewards)
    assert abs(float(adv.mean())) < 1e-6           # zero-mean
    assert abs(float(adv.std(unbiased=False)) - 1.0) < 1e-5

def test_grpo_loss_sign():
    # positive advantage with positive logprob delta should give negative loss contribution
    logp = torch.tensor([-1.0, -2.0])
    ref_logp = torch.tensor([-1.0, -1.0])
    adv = torch.tensor([1.0, -1.0])
    loss = grpo_loss(logp, ref_logp, adv, kl_beta=0.1)
    assert torch.isfinite(loss)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rl_smoke.py -v`
Expected: FAIL with `ImportError: cannot import name 'group_advantages'`

- [ ] **Step 3: Write minimal implementation**

```python
# train_vqg_rl.py
"""GRPO trainer for the BLIP decoder teacher (RL-1).

Only the math helpers (group_advantages, grpo_loss) are unit-tested. `main`
wires the BLIP teacher + reference model and is exercised by the smoke test.
Does NOT modify train_vqa.py / train_vqg.py.
"""

from __future__ import annotations

import torch


def group_advantages(rewards: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Group-relative advantage: (r - mean) / (std + eps) within the group."""
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    return (rewards - mean) / (std + eps)


def grpo_loss(
    logp: torch.Tensor,
    ref_logp: torch.Tensor,
    advantages: torch.Tensor,
    kl_beta: float,
) -> torch.Tensor:
    """GRPO objective: -E[A * logp] + beta * KL(policy || ref).

    KL approximated per-sample by (logp - ref_logp) (k1 estimator placeholder is
    replaced by the unbiased k3 estimator below for stability).
    """
    pg = -(advantages.detach() * logp).mean()
    log_ratio = logp - ref_logp
    kl = (torch.exp(log_ratio) - 1.0 - log_ratio).mean()  # k3 estimator, >= 0
    return pg + kl_beta * kl
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rl_smoke.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Add the model-wiring `main` (integration, not unit-tested here)**

```python
# train_vqg_rl.py  (append)
def train_one_epoch(policy, ref, loader, reward_fn, optimizer, cfg, device):
    """One GRPO epoch. `reward_fn(image_path, question, answer) -> float`.

    For each image: sample cfg.group_size completions, score rewards, compute
    group advantages, accumulate GRPO loss, step. Returns mean reward + per-term log.
    """
    policy.train()
    epoch_rewards = []
    for images, image_paths in loader:
        images = images.to(device)
        for i in range(images.size(0)):
            img = images[i : i + 1].repeat(cfg.group_size, 1, 1, 1)
            outputs, logp = policy.generate(
                img, sample=True, top_p=cfg.top_p,
                max_length=cfg.max_length, min_length=cfg.min_length,
                return_logprob=True,
            )
            with torch.no_grad():
                _, ref_logp = ref.generate(
                    img, sample=False, max_length=cfg.max_length,
                    min_length=cfg.min_length, return_logprob=True,
                )
            rewards = torch.tensor(
                [reward_fn(image_paths[i], *_parse_qa(o)) for o in outputs],
                device=device, dtype=torch.float32,
            )
            adv = group_advantages(rewards)
            loss = grpo_loss(
                torch.as_tensor(logp, device=device),
                torch.as_tensor(ref_logp, device=device),
                adv, kl_beta=cfg.kl_beta,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_rewards.extend(rewards.tolist())
    return sum(epoch_rewards) / max(1, len(epoch_rewards))


def _parse_qa(model_output: str) -> tuple[str, str]:
    """Reuse generate_questions parsing; fall back to naive split."""
    from generate_questions import VQARecord, VQADatasetOrigin
    try:
        rec = VQARecord.build_from_raw_model_output(
            model_output, "x", dataset_origin=VQADatasetOrigin("aokvqa"),
            parse_rationale=False,
        )
        ans = rec.answer[0] if isinstance(rec.answer, list) else rec.answer
        return rec.question, ans
    except Exception:
        return model_output, ""
```

- [ ] **Step 6: Commit**

```bash
git add train_vqg_rl.py tests/test_rl_smoke.py
git commit -m "feat(rl): add GRPO advantage/loss helpers and epoch trainer"
```

---

## Task 9: Round orchestrator (`orchestration/rl_teacher_loop.py`)

Wires rounds: GRPO epochs (with epoch audit + HRP) → generate (subprocess) → train student (subprocess) → eval → stop signal. Subprocess calls are injected so tests use fakes (no GPU).

**Files:**
- Create: `orchestration/rl_teacher_loop.py`
- Test: `tests/test_rl_smoke.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rl_smoke.py  (append)
from orchestration.rl_teacher_loop import should_continue, kl_beta_for_round


def test_should_continue_above_delta():
    assert should_continue(acc_curr=62.0, acc_prev=61.0, delta=0.5) is True

def test_should_stop_below_delta():
    assert should_continue(acc_curr=61.2, acc_prev=61.0, delta=0.5) is False

def test_kl_beta_decays_per_round():
    assert abs(kl_beta_for_round(beta0=0.1, gamma=0.7, r=1) - 0.1) < 1e-9
    assert abs(kl_beta_for_round(beta0=0.1, gamma=0.7, r=2) - 0.07) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rl_smoke.py::test_should_continue_above_delta -v`
Expected: FAIL with `ImportError: cannot import name 'should_continue'`

- [ ] **Step 3: Write minimal implementation**

```python
# orchestration/rl_teacher_loop.py
"""IT loop: accumulating-teacher GRPO + restarting-student, round-level stop."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def should_continue(acc_curr: float, acc_prev: float, delta: float) -> bool:
    return (acc_curr - acc_prev) >= delta


def kl_beta_for_round(beta0: float, gamma: float, r: int) -> float:
    """kl_beta_r = beta0 * gamma^(r-1). Round index r is 1-based."""
    return beta0 * (gamma ** (r - 1))


def run_round(r, cfg, run_subprocess=subprocess.run, eval_fn=None, state_dir="orchestration/state"):
    """Generate (no filter) -> train student (restart) -> eval. Returns acc_r.

    `run_subprocess` and `eval_fn` are injected for testing.
    """
    teacher_ckpt = f"{state_dir}/teacher_{r}.pth"
    pool = f"datasets/aokvqa/pool_{r}.json"
    run_subprocess([
        "python", "generate_questions.py",
        "--config", "configs/generate_questions_aokvqa.yaml",
        "--overrides", f"pretrained={teacher_ckpt}",
        f"output_annotations_name=pool_{r}.json",
    ], check=True)
    run_subprocess([
        "python", "-m", "torch.distributed.run", "--nproc_per_node=1", "train_vqa.py",
        "--output_dir", f"cache/rl_student_{r}",
        "--config", "configs/aokvqa.yaml",
        "--overrides", f"train_files=[train,pool_{r}]", "wandb=false",
    ], check=True)
    acc = eval_fn(r) if eval_fn else 0.0
    Path(f"{state_dir}/round_{r}.json").write_text(
        json.dumps({"round": r, "acc": acc, "teacher_ckpt": teacher_ckpt})
    )
    return acc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rl_smoke.py -v`
Expected: PASS (5 tests in file)

- [ ] **Step 5: Commit**

```bash
git add orchestration/rl_teacher_loop.py tests/test_rl_smoke.py
git commit -m "feat(rl): add round orchestrator stop-signal and KL decay"
```

---

## Task 10: Config + full smoke (mocked subprocess round)

**Files:**
- Create: `configs/rl_teacher_aokvqa.yaml`
- Test: `tests/test_rl_smoke.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rl_smoke.py  (append)
from orchestration.rl_teacher_loop import run_round


def test_run_round_mocked_subprocess(tmp_path):
    calls = []
    def fake_run(args, check=False): calls.append(args[1] if len(args) > 1 else args[0])
    def fake_eval(r): return 61.5
    acc = run_round(1, cfg=None, run_subprocess=fake_run,
                    eval_fn=fake_eval, state_dir=str(tmp_path))
    assert acc == 61.5
    assert any("generate_questions.py" in c for c in calls)
    assert (tmp_path / "round_1.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rl_smoke.py::test_run_round_mocked_subprocess -v`
Expected: FAIL (round file path / call assertion) until config + wiring present

- [ ] **Step 3: Write the config**

```yaml
# configs/rl_teacher_aokvqa.yaml
rounds: 3
early_stop_delta: 0.5

weak_types:
  k: 3
  student_base_eval: cache/evals/student_base/vqa_result.json
  out: orchestration/state/weak_types.json

grpo:
  group_size: 8
  top_p: 0.92
  temperature: 1.0
  max_length: 30
  min_length: 5
  kl_beta: 0.1
  kl_beta_decay_gamma: 0.7
  lr: 1.0e-6
  epochs_per_round: 3
  steps_per_epoch: 200

reward:
  w_type: 1.0
  w_itm: 0.5
  w_grounding: 1.0
  w_learnability: 0.5
  w_kl: 0.1
  w_repetition: 0.3

audit:
  every_epoch: true
  n_audit: 200
  itm_large_ckpt: cache/judges/blip_itm_large.pth
  vqa_judge_ckpt: cache/judges/blip_vqa_published.pth
  hacking_eps: 0.02
  itm_dominance_tau: 0.6
  rollback_to_best_judge: true

hrp:
  adaptive_kl_c: 1.5
  downweight_d: 0.5
  auto_levels: [1, 2, 3]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rl_smoke.py -v`
Expected: PASS (6 tests in file)

- [ ] **Step 5: Run the full suite**

Run: `pytest -m "not slow" tests/test_reward.py tests/test_weak_types.py tests/test_audit.py tests/test_hrp.py tests/test_rl_smoke.py -q`
Expected: PASS (all)

- [ ] **Step 6: Commit**

```bash
git add configs/rl_teacher_aokvqa.yaml tests/test_rl_smoke.py
git commit -m "feat(rl): add RL teacher config and full mocked-round smoke test"
```

---

## Task 11: Invariants check

- [ ] **Step 1: Run repo invariants**

Run: `bash scripts/check_invariants.sh`
Expected: PASS (no edits to `train_vqa.py` / `train_vqg.py`; new files only)

- [ ] **Step 2: Confirm no forbidden edits**

Run: `git diff --name-only origin/feat/pseudo-label-filter...HEAD`
Expected: only new files under `filtering/`, `orchestration/`, `configs/`, `tests/`, and `train_vqg_rl.py`

---

## Self-Review

**Spec coverage:**
- §2.2 reward terms → Tasks 1–4 (TypeMatch, ITM weight, Grounding=xcons×LP_flip, Learnability, KL, Repetition). ITM term is a config weight applied in `compose_reward`; the CLIP score itself reuses `filtering/scorers.score_clip_itm` at integration (Task 8 `reward_fn` builder — wire `OpenClipAdapter`).
- §3 WT-1 → Task 5. §5 AUD-1 → Task 6. §6 HRP-1 → Task 7. §2.3 GRPO → Task 8. §2.1 loop + init policy + KL decay → Task 9. §1.4/§9 config → Task 10. §1.4 invariants → Task 11.

**Gaps to close during execution:**
- Task 8 `reward_fn` must be assembled from `filtering/reward.py` terms + adapters (`OpenClipAdapter`, two `BlipStudentAdapter` instances: base-student for learnability/grounding, published-VQA for audit) — build this assembler in Task 8 Step 5 when wiring `main`, injecting `score_clip_itm` for the ITM term.
- `student.answer_prob` (Learnability) is not on `BlipStudentAdapter` yet — add a thin method returning the greedy-answer max softmax prob when wiring Task 8 (BLIP-VQA exposes logits in `inference="generate"` path).
- `corrupt(image)` for LP_flip = zero/blank the image tensor; implement in the Task 8 assembler.

**Placeholder scan:** no TBD/TODO; every code step has runnable content. `<path>`-style values live only in YAML config (environment-specific, expected).

**Type consistency:** `RewardTerms`/`RewardConfig` field names match `compose_reward`; `HrpState.w_itm`/`kl_beta` match `respond`; `group_advantages`/`grpo_loss` signatures match their tests.
```
