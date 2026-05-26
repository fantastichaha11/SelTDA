# Creative Session 2 — Janusian Synthesis & New Candidates

- **Date**: 2026-05-26 (continued session)
- **Skill**: creative-thinking-for-research (Frameworks 3, 8, 2)
- **Trigger**: Post-convergence gaps — tensions not yet resolved in ranked portfolio

---

## Framework 8 — Janusian resolutions (new syntheses)

### J-1: Filter as gate AND filter as curriculum

**Tension**: Strict filtering ↓ quantity; curriculum needs easy→hard progression.

**Synthesis**: Do not choose — **two-stage pool**:
- Stage A: keep top 90% by conf (lenient) for round-1 student warmup (few epochs via duplicate-sampling only easy half).
- Stage B: apply full C+I+X at 0.75 for main training mix.

§1.4 compatible via **two synthetic JSON files** concatenated in `train_files` without code change.

**Test**: Does staged pool beat single-threshold at matched total steps?

---

### J-2: Teacher ensemble vs single strong teacher

**Tension**: C15 says ensemble reduces variance; compute says one teacher only.

**Synthesis**: **Cheap disagreement gate** — run 2 teachers (different seeds, same arch), keep pseudo-QA only when answers match (SBERT > τ). No ensemble training — ensemble **only at generation** for filtering. Cost: 2× generation, 0× extra training.

Maps to **Teacher-Council-lite** from creative session 1.

---

### J-3: Zero-shot PathVQA vs direct train

**Tension**: Paper's zero-shot is weak; direct train may overfit small PathVQA train.

**Synthesis**: **Two-stage domain transfer** — (1) SelTDA on A-OKVQA with filter → strong general student; (2) **freeze vision**, fine-tune text head only on filtered PathVQA pseudo. Reduces overfit, tests Limitation #2 in isolation.

Requires checking if §1.4 allows — **no train_vqa.py change** means use existing config knobs only (`truncate_train`, `train_files`). If freeze not exposed → document as future work.

---

### J-4: RAG helps generation vs RAG hurts if misaligned

**Synthesis from MMed-RAG**: Use retrieval confidence as **fourth gate** — drop pseudo-QA when retrieved context similarity < τ (adaptive context selection idea). Filter becomes C+I+X+**R**.

---

## Framework 3 — Structural analogies (new)

### Analogy: Unit tests in software CI

| Software CI | SelTDA |
|-------------|--------|
| Unit test must pass | xcons: student must reproduce A |
| Integration test | multi-view agreement across augmentations |
| Fuzz testing | counterfactual image I' |
| Code coverage | coreset coverage over embedding space |
| Flaky test quarantine | low-confidence bucket excluded |

**Candidate UT-1**: Package filter gates as **"pseudo-QA test suite"** with named tests — strong systems narrative for thesis intro.

---

### Analogy: Peer review

| Peer review | SelTDA filter |
|-------------|---------------|
| Reviewer 1 (confidence) | conf gate |
| Reviewer 2 (visual grounding) | ITM gate |
| Reviewer 3 (replication) | xcons gate |
| Meta-reviewer | soft fusion / coreset |
| Desk reject | cascade early exit |

**Insight**: Report **which gate rejects most** per question type — diagnostic table for paper (not yet in spec report.json).

---

## Framework 2 — Problem reformulation (new angles)

| Original | Reformulation | Candidate |
|----------|---------------|-----------|
| "Improve SelTDA accuracy" | "Find the noise saturation point of synthetic VQA data" | **SAT-1**: characterize ratio-accuracy curve under filter; contribution = precise saturation boundary |
| "Filter bad pseudo-QA" | "Maximize information per synthetic sample" | CS-X (already ranked #1) |
| "Fix PathVQA" | "Measure vocabulary gap between COCO teacher and pathology terms" | **LEX-1**: lexicon overlap analysis + RAG ablation |

---

## New candidates from session 2

| ID | Pitch | Priority |
|----|-------|----------|
| **J-1** | Staged lenient→strict synthetic pools as curriculum | Medium — fast ablation |
| **J-2** | Two-seed teacher disagreement gate | Medium — 2× gen cost |
| **J-4** | RAG-confidence fourth gate | High if RG-1 runs |
| **UT-1** | "Test suite" framing + per-gate reject analytics | Low cost — paper writing |
| **SAT-1** | Saturation curve characterization | High — explains Khan Tab.4 mechanistically |
| **LEX-1** | Medical vocabulary gap analysis | High for PathVQA chapter |

---

## Integration with ranked portfolio

- **SAT-1** merges into H2 experiments (ratio sweep) — not a separate direction.
- **J-4** extends RG-1 — implement as optional gate in `filter_pseudo.py`.
- **UT-1** extends `filtering/report.py` — add per-gate reject counts by question type.
- **J-1** optional Phase 1.5 if H1 shows threshold sensitivity.

---

## Self-check (creative-thinking quality)

- [x] Structural analogies (CI, peer review) — not surface metaphors
- [x] Janusian syntheses hold both poles simultaneously
- [x] Each candidate has testable prediction
- [x] §1.4 compatibility checked for J-1, J-2, J-4
