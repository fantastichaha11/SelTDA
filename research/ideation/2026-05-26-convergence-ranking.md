# Convergence Ranking — 14+ Candidates → Thesis Portfolio

- **Date**: 2026-05-26 (continued session)
- **Skill**: brainstorming-research-ideas Phase 2 (Converge)
- **Input**: `2026-05-26-divergent-brainstorm.md` (14 candidates) + `2026-05-26-creative-thinking-deepening.md` (5 transformational) + literature evidence pass

---

## Filter criteria applied

| Filter | Kill criterion |
|--------|----------------|
| **F10 Explain-It** | Cannot state in two sentences → demote |
| **F1 Problem-First** | No genuine pain point → drop |
| **F7 Simplicity** | Complex method with no ablation path → defer |
| **F8 Stakeholder** | No clear beneficiary → drop |
| **Feasibility** | Violates §1.4 without workaround → flag or drop |
| **Evidence** | No verified literature support → defer |
| **Missing baseline** | Must include random-subsample-at-matched-N for all filter claims |

---

## Full candidate scorecard

Scores 1–5 (5 = best). **Total** = weighted sum (Evidence×2 + Feasibility×2 + Novelty + Impact + §1.4-fit).

| ID | Name | Evid | Feas | Nov | Imp | §1.4 | Total | Verdict |
|----|------|------|------|-----|-----|------|-------|---------|
| CF-1 | Cascade filter | 5 | 5 | 3 | 4 | 5 | **32** | **Phase 1 — execute H1** |
| TS-1 | Type-stratified thresholds | 4 | 5 | 4 | 4 | 5 | **31** | **Phase 1 — bundle with H1** |
| CS-X | Coreset + stratified | 5 | 4 | 4 | 4 | 5 | **31** | **Phase 1.5 — after H1** |
| IT-1 | Iterative SelTDA | 5 | 3 | 5 | 5 | 4 | **30** | **Phase 2** |
| RG-1 | RAG medical teacher | 4 | 3 | 5 | 5 | 5 | **29** | **Phase 2 parallel (PathVQA)** |
| CT-1 | Counterfactual filter/aug | 4 | 3 | 5 | 4 | 5 | **28** | **Phase 3 — robustness** |
| RW-1 | Reward-shaped VQG (DPO) | 4 | 3 | 5 | 4 | 4 | **27** | **Phase 3 — high risk/reward** |
| AL-1 | Active image selection | 3 | 4 | 4 | 3 | 5 | **26** | Defer |
| EB-1 | Energy-based filter | 3 | 3 | 5 | 3 | 5 | **25** | Defer unless H1 fails |
| PAC-1 | PAC-style bounds | 2 | 4 | 5 | 3 | 5 | **25** | Appendix / theory chapter |
| JUDGE-1 | GPT-4V judge | 3 | 2 | 4 | 4 | 5 | **24** | Mini-ablation only (cost) |
| GROUND-1 | SAM-grounded VQG | 3 | 3 | 4 | 3 | 5 | **24** | Defer |
| MULTI-1 | Multi-task teacher | 3 | 2 | 4 | 3 | 3 | **22** | Defer (scope) |
| PROV-1 | Provenance hashing | 2 | 5 | 3 | 2 | 5 | **22** | Engineering note |
| DIFF-1 | Diffusion counterfactual | 4 | 2 | 4 | 3 | 5 | **22** | Merge into CT-1 |
| DPO-1 | (= RW-1) | — | — | — | — | — | — | Alias |
| MC-1 | MCMC-SelTDA | 3 | 2 | 5 | 3 | 4 | **23** | Theory follow-up to IT-1 |
| PV-1 | Pivot-VQG (2-stage Q then A) | 3 | 4 | 4 | 3 | 5 | **26** | Fast follower if CF-1 weak |

---

## Tier assignments

### Tier A — Must run (thesis core)

1. **CF-1 + TS-1** — H1 protocol locked; adds per-type quantiles (cheap, principled).
2. **Random-subsample baseline** — mandatory control at matched kept-set size (F7 gap from brainstorm).
3. **CS-X** — 2×2 factorial after H1 winner known.

### Tier B — Strong chapters if time

4. **IT-1** — 3-round loop + curriculum restart; DataEnvGym citation + Khan same-author narrative.
5. **RG-1 / H4+H5** — PathVQA direct training + MMed-RAG-style retrieval at generation.

### Tier C — Differentiation / robustness

6. **CT-1** — counterfactual drop/augment for bias (Limitation #3).
7. **H3 multi-view** — STIC-inspired corruption agreement (extends xcons).

### Tier D — Optional / appendix

8. **RW-1** — publishable even with negative result if reward hacking characterized (Rafailov 2024).
9. **PAC-1** — theory bound linking filter precision → student error.
10. **EB-1** — only if logistic calibration beats 3-gate and EBM adds ECE gains.

### Tier E — Cut unless trivial to add

JUDGE-1, GROUND-1, MULTI-1, PROV-1 as standalone chapters.

---

## Recommended thesis narrative (3 contributions)

> **Contribution 1 (Filtering)**: We show that cascade pseudo-label filtering with type-stratified thresholds and coreset selection raises A-OKVQA accuracy above unfiltered SelTDA and enables higher synthetic ratios without noise dilution (H1, H2, CS-X).

> **Contribution 2 (Iteration)**: We adapt DataEnvGym-style student-feedback loops to open-ended VQA, with curriculum restarts mitigating confirmation bias, yielding compound gains over single-round SelTDA (H6, IT-1).

> **Contribution 3 (Domain)**: We train SelTDA directly on PathVQA with RAG-augmented teacher generation, closing much of the gap left by the original paper's zero-shot transfer (H4, H5, RG-1).

CT-1 / counterfactual robustness slots as **§4 robustness study** or fourth contribution if results are strong on VQA-CE / AdVQA.

---

## Two-sentence pitches (winners only)

**CS-X**: SelTDA filtering keeps high-confidence samples that are often near-duplicates, wasting the synthetic budget. We combine type-stratified quality gates with ZCore-style submodular coreset selection so each kept pseudo-QA maximizes coverage per unit noise.

**IT-1**: SelTDA stops after one teacher-student round even though validation errors reveal exactly which skills are weak. We close the loop with DataEnvGym-inspired re-teaching targeted at weak question types, restarting student weights each round to prevent confirmation bias.

**RG-1**: SelTDA's PathVQA gain was only +1.67% because the model was never trained on pathology and the teacher lacks medical vocabulary. We retrieve domain-aware biomedical context at generation time (MMed-RAG-style) and train the student directly on PathVQA with filtered pseudo-QA.

---

## Kill list (with reasons)

| ID | Reason dropped from core |
|----|--------------------------|
| MULTI-1 | Scope explosion; 4 datasets × filtering × iteration |
| JUDGE-1 | API cost + non-reproducibility for main results |
| PROV-1 | Regulatory framing without clinical deployment path |
| GROUND-1 | SAM pipeline adds heavy deps; marginal over ITM gate |
| DIFF-1 | Merged into CT-1 (same counterfactual mechanism) |

---

## Next actions (ordered)

1. Fix **xcons leakage** (pretrained-only student judge).
2. Generate `synthetic_data_raw.json`.
3. Run H1 ablation matrix + random-subsample control.
4. Implement TS-1 (per-type quantiles) as config flag in `filter_pseudo.py`.
5. Wire ZCore embedding pipeline for CS-X factorial.
6. Parallel: PathVQA direct-training data prep for RG-1.
