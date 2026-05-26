# Phase 2 Literature Synthesis — Verified Sources for Thesis Directions

- **Date**: 2026-05-26 (continued session)
- **Mode**: deep-research Phase 2 (Investigation) + source verification
- **Method**: Hugging Face paper index (verified arXiv IDs). Abstract-level reads unless marked [full-text read].
- **Supersedes partial coverage in**: `docs/reports/2026-05-26-deep-research-seltda-improvements.md` §4–5

---

## Verification status key

| Tag | Meaning |
|-----|---------|
| ✅ VERIFIED | arXiv ID confirmed via HF index; abstract read |
| ⚠️ ABSTRACT-ONLY | Claims below are from abstract; full text not yet read |
| ❌ FAIL | Could not verify — excluded from thesis citations |

---

## 1. DataEnvGym — Iterative SelTDA authority (IT-1 / H6)

**Citation**: Khan, Stengel-Eskin, Cho, Bansal (2024). arXiv **2410.06215**. ✅ VERIFIED  
**Link**: https://hf.co/papers/2410.06215

### Core claim (abstract)

Frames data generation as **sequential decision-making**: an agent (policy + engine) operates inside a teacher environment that returns **student feedback** (errors / weak skills) after each iteration. Supports math, code, and **VQA**. Students are iteratively trained on generated data; feedback closes the loop.

### Why this matters for SelTDA

| SelTDA (2023) | DataEnvGym (2024) |
|---------------|-------------------|
| 1 round: teacher → pseudo → student | N rounds with feedback-driven re-generation |
| No skill-gap signal | Explicit weak-skill reporting |
| Same author (Zaid Khan) | Direct conceptual successor |

### Transferable design elements for thesis IT-1

1. **Skill decomposition**: bucket val errors by question type (A-OKVQA EK / VR / VID) or PathVQA category → target teacher re-fine-tuning.
2. **Restart between rounds**: combine with Cascante-Bonilla 2020 (arXiv 2001.06001) — reset student weights before each self-training cycle to mitigate confirmation bias (Arazo et al. 2019, arXiv 1908.02983).
3. **Environment levels**: DataEnvGym offers structured vs unstructured state — for VQA, structured = per-type error counts; unstructured = raw wrong-answer strings.

### Counter-evidence / risks

- DataEnvGym VQA results are **not yet reproduced in this repo**; scale and compute for N-round BLIP training may exceed 1-GPU thesis budget.
- Feedback signal quality depends on val set size — A-OKVQA val is small; per-type estimates are noisy (same caveat as Khan 2023 Table 3, n=100 manual eval).

### Thesis mapping

- **Hypothesis H6**: 3-round iterative SelTDA compounds over 1-round filtered SelTDA.
- **§1.4 note**: iterative training requires multiple `train_vqa.py` runs with different synthetic JSONs — allowed; no training code modification.

---

## 2. MMed-RAG — Medical branch (RG-1 / H5)

**Citation**: Xia et al. (2024). arXiv **2410.13085**. ✅ VERIFIED  
**Link**: https://hf.co/papers/2410.13085  
**Code**: https://github.com/richard-peng-xia/MMed-RAG

### Core claim (abstract)

Versatile multimodal RAG for Med-LVLMs addressing **factual hallucination**. Three components:

1. **Domain-aware retrieval** — retrieve from radiology / ophthalmology / pathology KBs.
2. **Adaptive retrieved context selection** — avoid modality misalignment when injecting context.
3. **RAG-based preference fine-tuning** — alignment when retrieved context is introduced.

Reports **+43.8% average improvement in factual accuracy** across five medical datasets (VQA + report generation). ⚠️ ABSTRACT-ONLY — metric definition and baselines need full-text read before citing numerically in thesis.

### Why this matters for PathVQA

Khan 2023 Limitation #2: teacher fails on specialized medical vocabulary. MMed-RAG attacks the same failure mode via **external biomedical context at generation/inference time**, not via retraining the student backbone.

### SelTDA-specific application (RG-1)

```
generate_questions.py:
  for each PathVQA unlabeled image:
    retrieve top-k similar (image, text) from small biomedical KB
    prepend retrieved snippets to teacher prompt
    generate (Q, A) as today
filter_pseudo.py:
  unchanged gates (conf, itm, xcons)
train_vqa.py:
  unchanged (§1.4)
```

### Related 2025 work (optional extension)

- **Patho-AgenticRAG** (arXiv 2508.02258): multimodal RAG with textbook page-level embeddings + RL — stronger for pathology-specific VQA. Defer until RG-1 baseline works.

### Risks

- RAG KB curation cost for PathVQA (PubMed abstracts, pathology textbooks).
- Retrieval noise can **hurt** if context is irrelevant — MMed-RAG's adaptive selection is the critical piece, not naive top-k.

---

## 3. STIC — Multi-view / corruption filtering (H3 extension)

**Citation**: Deng et al. (2024). arXiv **2405.19716**. ✅ VERIFIED  
**Link**: https://hf.co/papers/2405.19716

### Core claim (abstract)

Self-training for LVLMs on **image comprehension**: build preference data from unlabeled images — **preferred** = step-by-step prompt; **dispreferred** = corrupted images or misleading prompts. Reuses small SFT data with self-generated descriptions. **+4.0% average on 7 benchmarks** with **70% less SFT data**. Code public.

### Mapping to SelTDA filtering (not full STIC training)

STIC's dispreferred pairs ≡ **pseudo-QA that fails under image corruption**. Direct filter extension:

| Signal | Implementation in SelTDA |
|--------|--------------------------|
| Preferred | student agrees on clean image (current xcons) |
| Dispreferred | student disagrees when image has mild corruption |
| Keep rule | high agreement on clean AND high agreement across 3 augmented views |

This operationalizes **H3** (multi-view self-consistency) with literature backing stronger than generic "ensemble decoding."

### Cost estimate

3× student forward per pseudo-sample ≈ 1.25 GPU-hours on ~100k pool (from deep-research report §A.1) — feasible.

---

## 4. DeFacto — Counterfactual generation (CT-1 / DIFF-1)

**Citation**: Xu et al. (2025). arXiv **2509.20912**. ✅ VERIFIED  
**Link**: https://hf.co/papers/2509.20912

### Core claim (abstract)

Counterfactual reasoning framework enforcing **accurate answering + faithful reasoning**. Three training paradigms: (i) positive, (ii) counterfactual, (iii) random-masking. Pipeline localizes question-relevant evidence, builds ~100k image variants. **GRPO-based RL** with three complementary rewards.

Targets: models that answer correctly via **spurious regions** — same failure mode as SelTDA bias amplification (Khan 2023 Limitation #3).

### SelTDA application (lighter than full DeFacto)

Full GRPO is §1.4-compatible only if applied to teacher/generation — not student training code. Practical thesis path:

1. **Filter-only counterfactual**: for each `(I, Q, A)`, generate `I'` (object removal / attribute swap via SDXL or simple inpainting).
2. **Drop** pseudo-QA if student still predicts `A` on `I'` (shortcut / language prior).
3. **Augment** (optional): add `(I', Q, A')` as hard negative via duplicate-sampling — student sees contrastive pairs without loss reweighting.

### Related benchmark

**CounterVQA** (arXiv 2511.19923): evaluates counterfactual reasoning in video VQA — useful eval suite if CT-1 branch proceeds.

---

## 5. ZCore — Coreset selection (CS-X / CS-1)

**Citation**: Griffin, Marks, Corso (2024). arXiv **2411.15349**. ✅ VERIFIED  
**Link**: https://hf.co/papers/2411.15349  
**Code**: https://github.com/voxel51/zcore

### Core claim (abstract)

**Zero-shot coreset selection** without labels or training on candidate data. Uses foundation-model embeddings → iterative subspace sampling for **coverage vs redundancy**. Outperforms label-based methods at **low data rates** (e.g., ImageNet 10% → 53.99% val acc).

### Direct SelTDA wiring (CS-X)

```
scores = conf × itm × xcons          # existing gates
embed  = CLIP or DINOv2(I, "Q? A.")  # per pseudo-QA
within each question-type stratum:
  select top-N by quality score
  then ZCore submodular pick for coverage
output synthetic_data.json
```

**2×2 factorial experiment** (from literature deep-dive):

| | No coreset | + coreset |
|--|------------|-----------|
| No filter | baseline | CS-only |
| + filter | CF-1 (H1) | **CS-X** |

Key question: is coreset **additive** over quality filter at matched N?

---

## 6. Confirmation bias literature (cross-cutting)

| Paper | arXiv | Lesson for SelTDA |
|-------|-------|-------------------|
| Arazo et al. 2019 | 1908.02983 | Naive pseudo-labeling overfits wrong labels — **confirmation bias** |
| Cascante-Bonilla 2020 | 2001.06001 | Curriculum + **restart weights** each cycle → competitive SSL |
| Rafailov et al. 2024 | 2406.02900 | Reward hacking in DPO — critical for RW-1 |
| SemiReward 2023 | 2310.03013 | Learned reward for pseudo-label quality — alternative to rule gates |

**Thesis implication**: any iterative or reward-shaped variant (IT-1, RW-1) must report confirmation-bias diagnostics (e.g., filter precision drift across rounds).

---

## 7. Evidence gaps remaining (post Phase 2)

| Gap | Priority | Action |
|-----|----------|--------|
| MMed-RAG +43.8% — exact metric | High | Read full paper §4 before citing |
| DataEnvGym VQA numbers | High | Read full paper experiments |
| ZCore implementation on text-image pairs | Medium | Port ZCore to `(I,Q,A)` embedding pipeline |
| Conformal prediction for VQA pseudo-labels | Medium | No direct VQA paper found; split-conformal may be novel contribution (PAC-1) |
| Calibration (ECE) on SelTDA filter scores | Medium | Run scoring_only audit in H1 protocol |

---

## 8. Updated evidence hierarchy for thesis citations

| Tier | Sources for this project |
|------|--------------------------|
| **I — Primary** | Khan 2023 SelTDA (PDF in repo, full read) |
| **II — Same-author successor** | DataEnvGym 2410.06215 |
| **III — Direct mechanism transfer** | ZCore 2411.15349, STIC 2405.19716, MMed-RAG 2410.13085 |
| **IV — Abstract-level / pending full read** | DeFacto 2509.20912, ViLP 2501.00569 |
| **V — Do not cite without verification** | Perplexity 2026-05-25 report (fabricated IDs) |

---

## AI disclosure

Literature search and synthesis assisted by AI tools (Hugging Face paper index, Claude). All arXiv IDs independently verified before inclusion.
