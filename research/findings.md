# Research Findings — SelTDA Improvements

**Seeded 2026-05-26.** Full prior synthesis lives at `docs/reports/2026-05-26-deep-research-seltda-improvements.md`. This file is the lightweight project memory updated each outer-loop cycle.

## Research Question

How can we extend SelTDA (Khan et al., CVPR 2023) — a self-training pipeline that generates pseudo (Q,A) from unlabeled images via a BLIP teacher — to substantially exceed its baseline on A-OKVQA and especially PathVQA, given that ~30% of the teacher-generated QA pairs are noisy by the authors' own manual evaluation?

## Current Understanding (Bootstrap snapshot)

SelTDA's published A-OKVQA gain of +5.0 absolute points (57.1 → 62.1 val) comes from teacher-generated pseudo-QA that is **measurably noisy** (Khan 2023, Table 3 — only 62% answer-correct on external-knowledge type, 70% on visual-reasoning, 88% on visual-identification). The paper's ablation (Table 4) shows performance peaks at synth:real = 2:1 and **degrades above 4:1** — strong evidence that adding more pseudo-data without filtering hurts.

PathVQA in the original paper got only +1.67 because it was **zero-shot transfer from A-OKVQA, not direct training**. Direct training of SelTDA on PathVQA is a fair experiment never run in the original paper.

The authors themselves list four limitations (Section 5): noisy pseudo-QA, failure on specialized vocabulary, bias amplification, and untried with billion-parameter VLMs. This is, in effect, an author-endorsed checklist for follow-up work.

## Key Results

None yet — bootstrap only. Baseline A-OKVQA: 57.11 (no synthetic). Published SelTDA: 60.01 (unfiltered, synth:real=2:1). Both from Khan 2023 Table 4.

## Patterns and Insights

**2026-05-26 continued session (deep-research Phase 2–3 + converge + creative session 2):**

1. **DataEnvGym (arXiv 2410.06215)** is the authoritative iterative successor — same first author as SelTDA; IT-1/H6 should cite this as conceptual foundation, not ad-hoc multi-round training.
2. **ZCore (arXiv 2411.15349)** is the lowest-risk algorithmic addition — zero-shot coreset on unlabeled pools matches SelTDA's pseudo-pool exactly; public code at voxel51/zcore.
3. **Random-subsample-at-matched-N** is the single most important missing baseline (brainstorm F7) — without it, filter gains may be confounded with smaller dataset size.
4. **STIC (arXiv 2405.19716)** gives literature backing for H3 multi-view filtering via corruption/dispreferred pairs — stronger narrative than generic temperature ensemble.
5. **MMed-RAG (arXiv 2410.13085)** claims +43.8% factual accuracy on medical tasks (abstract-only) — verify full text before citing; RG-1 remains high-value for PathVQA.
6. **DeFacto (arXiv 2509.20912)** supports CT-1 counterfactual branch for Limitation #3 (bias/shortcut) — full GRPO likely out of scope; filter-only counterfactual drop is §1.4-safe.
7. **Janusian syntheses** (session 2): staged lenient→strict pools (J-1), RAG-confidence fourth gate (J-4), saturation curve characterization (SAT-1) integrate into existing phases without new hypotheses.
8. **Thesis portfolio converged** to 3 contributions: (1) filter+coreset+type-stratified, (2) iterative DataEnvGym-style loop, (3) PathVQA direct + RAG teacher.

## Compute efficiency (2026-05-26)

**Filter chậm** vì `xcons` chạy BLIP-VQA beam-search **51k lần** (batch=1), `itm` chạy CLIP 51k lần, ảnh đọc 2 vòng — gate `conf` gần như miễn phí. Thực tế **~3–6h** cho full pool, không ~45 phút. Mitigation: `scoring_only=true` một lần rồi sweep threshold offline; tắt xcons cho pilot; batch/cache (cải tiến code).

**Train student chậm** vì mỗi experiment = **10 epoch × ~34–68k samples**, eval cuối, và §1.4 bắt mỗi filter variant một run `train_vqa.py` riêng. H1 naive ≈ **210h train** (7×3×10h). Mitigation: successive halving (proxy `max_epoch=3`, 1 seed) → full train chỉ top-2; reuse published/cache unfiltered checkpoint; random-subsample baseline; CE-01..18 trong `research/notes/2026-05-26-compute-efficiency.md`.

## Lessons and Constraints

- **§1.4 invariant** (from `docs/superpowers/specs/2026-05-21-pseudo-label-filtering-design.md`): do NOT modify `train_vqa.py`, `train_vqg.py`, `data/`, `models/` (except `models/blip.py` scoped change for `return_logprob`), `vqa_eval_tools/`, or any train/eval configs. Only generation, filtering, and orchestration code is in scope. All experiments must respect this.
- **Fair comparison rule**: when comparing filtered vs unfiltered SelTDA, hold everything else constant (same train_vqa.py, same config, same seed). The only difference is the `synthetic_data.json` produced by filter step.
- **Soft-weighting workaround**: since loss-side weighting would violate §1.4, soft scores must be implemented as duplicate-sampling (frequency ∝ weight) in the filter output, not as a per-sample weight in training.
- **Sample-size caveat for Khan 2023 Table 3**: the 62/70/88% answer-correctness is from n=100 manual eval with wide 95% CI. Don't treat as gospel; redo the human-judge eval on our own filtered/dropped samples.
- **Previous Perplexity report (2026-05-25) contains fabricated arXiv IDs** for some papers (e.g., "RISE", "R-C2"). Do not trust uncited claims from that report; use only papers cross-verified in the 2026-05-26 deep research report.

## Ideation Output (2026-05-26 overnight)

Applied brainstorming-research-ideas (10 frameworks) + creative-thinking-for-research (8 frameworks). Total: 17 named candidates ranked by evidence-weighted novelty × feasibility × §1.4 compatibility. See `research/ideation/` and `research/literature/`.

**Top 3 (post-evidence)** — unchanged core; see extended catalog for 28 ideas total:
1. **CS-X Coreset-Stratified Filter** — combine ZCore-style submodular coreset selection (arXiv 2411.15349, Nov 2024) with per-question-type quantile filtering. Lowest risk, highest direct mapping from existing 2024 algorithm. **Promoted to thesis Section 2.**
2. **IT-1 Iterative-SelTDA** + curriculum + restart — DataEnvGym (Khan 2024, arXiv 2410.06215) supports the framework; Cascante-Bonilla 2020 supports the restart move.
3. **RW-1 Reward-Shaped VQG** — DPO of teacher with filter score as reward. Must include held-out judge + KL budget + reward-hacking measurement framework (Rafailov 2024 arXiv 2406.02900 documents the failure mode). Janusian reframe positions RW-1 as a *measurement instrument* — publishable even with negative headline result.

**Extended research (2026-05-26 session 2)**: 45+ papers surveyed; **28 named ideas** with detailed mechanism descriptions in `research/ideation/2026-05-26-extended-ideas-catalog.md`. Highlights: ViLP language-prior gate (LP-1), CycleReward gate (CR-1), BARE two-stage VQG, Maieutic filter, NovelSum diversity metric, SAT-1 saturation curves.

**SOTA landscape (2026-05-26 Firecrawl)**: Non-paper sources (Wizwand, CodeSOTA, OK-VQA official, DataEnvGym site) calibrate expectations. A-OKVQA MC SOTA HinD-CoT-Know **87.2%** vs SelTDA **62.1% DA val** — different metric/backbone; QACap **73.4% DA val** is fairer literature reference. PathVQA **92.7% overall** (VILA-M3) vs SelTDA **26.76%** is **not directly comparable** (yes/no dominates overall; free-form ~13%). **+10 SOTA-informed ideas** (IDEA-29..38): KC-1 knowledge gate, SS-1 SemIAug compose, HAL-1 medical hallucination filter, MCR-1 stratified metrics. See `research/literature/2026-05-26-sota-landscape-firecrawl.md`.

**Transformational five** (could anchor a stronger thesis if execution time allows): EB-1 (Energy-Based), RW-1, MC-1 (MCMC-SelTDA), CS-X, PV-1 (Pivot-VQG).

## Open Questions

1. Does filter precision (% of kept pseudo-QA that are correct on human judge) correlate with final student accuracy gain? Spec §5.4 anticipates this but no data yet.
2. Does optimal synth:real ratio shift when filter is on? Khan 2023 found peak at 2:1; with filtering we predict ≥4:1 still gains. **SAT-1** frames this as saturation-curve characterization.
3. Is single-gate cascade (cheapest) within 1 absolute point of full 3-gate cascade? If yes, the cheaper gate alone could be enough for production.
4. Does RAG-augmented teacher on PathVQA improve answer-correctness ≥10 points on human judge? Required to justify the medical branch. **LEX-1** vocabulary-gap analysis should precede RG-1.
5. How many rounds of iterative SelTDA before gain saturates? DataEnvGym shows iterative wins for math/code; VQA numbers need full-text read.
6. Is coreset **additive** over quality filter at matched N? (CS-X 2×2 factorial — key question from literature pass.)
7. Does staged lenient→strict pool (J-1) beat single threshold? Cheap ablation if H1 shows threshold sensitivity.
8. MMed-RAG +43.8% — what metric and baseline? Full-text read required before thesis citation.
9. Does KC-1 (knowledge-consistency gate) close gap to QACap 73.4% on EK subset without full LLM pipeline?
10. Is SemIAug (+1.4% BLIP) additive with filtered SelTDA (SS-1)?

## Optimization Trajectory

| run_id | hypothesis | A-OKVQA val | PathVQA test | delta vs baseline | wall_time_min | summary |
|---|---|---|---|---|---|---|
| (baseline) | — | 57.11 | — | 0 | — | BLIP-base, no synthetic (Khan 2023 Tab.4) |
| (published SelTDA) | — | 60.01 | 26.76 (zs) | +2.90 / +1.67 | — | BLIP-base + SelTDA unfiltered, synth:real=2:1 |

Will append measured runs as inner loop executes.
