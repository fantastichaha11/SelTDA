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

(Will be populated after first outer-loop cycle.)

## Lessons and Constraints

- **xcons judge leakage risk** (discovered 2026-05-26 during repo audit): `configs/filter_pseudo.yaml` currently sets `xcons.student_ckpt = cache/student_weights/checkpoint_09.pth`. Per `CLAUDE.md` that checkpoint is the **published SelTDA student**, which was trained on synthetic data. Using it as the xcons judge means a sample's pseudo-answer that matches the model's training distribution will get a high score regardless of correctness — exactly the leakage that spec §3.3 warns against. Must replace with BLIP pretrained-only weights before any H1 run produces interpretable numbers.
- **§1.4 invariant** (from `docs/superpowers/specs/2026-05-21-pseudo-label-filtering-design.md`): do NOT modify `train_vqa.py`, `train_vqg.py`, `data/`, `models/` (except `models/blip.py` scoped change for `return_logprob`), `vqa_eval_tools/`, or any train/eval configs. Only generation, filtering, and orchestration code is in scope. All experiments must respect this.
- **Fair comparison rule**: when comparing filtered vs unfiltered SelTDA, hold everything else constant (same train_vqa.py, same config, same seed). The only difference is the `synthetic_data.json` produced by filter step.
- **Soft-weighting workaround**: since loss-side weighting would violate §1.4, soft scores must be implemented as duplicate-sampling (frequency ∝ weight) in the filter output, not as a per-sample weight in training.
- **Sample-size caveat for Khan 2023 Table 3**: the 62/70/88% answer-correctness is from n=100 manual eval with wide 95% CI. Don't treat as gospel; redo the human-judge eval on our own filtered/dropped samples.
- **Previous Perplexity report (2026-05-25) contains fabricated arXiv IDs** for some papers (e.g., "RISE", "R-C2"). Do not trust uncited claims from that report; use only papers cross-verified in the 2026-05-26 deep research report.

## Open Questions

1. Does filter precision (% of kept pseudo-QA that are correct on human judge) correlate with final student accuracy gain? Spec §5.4 anticipates this but no data yet.
2. Does optimal synth:real ratio shift when filter is on? Khan 2023 found peak at 2:1; with filtering we predict ≥4:1 still gains.
3. Is single-gate cascade (cheapest) within 1 absolute point of full 3-gate cascade? If yes, the cheaper gate alone could be enough for production.
4. Does RAG-augmented teacher on PathVQA improve answer-correctness ≥10 points on human judge? Required to justify the medical branch.
5. How many rounds of iterative SelTDA before gain saturates? DataEnvGym shows iterative wins for math/code; VQA evidence-gap.

## Optimization Trajectory

| run_id | hypothesis | A-OKVQA val | PathVQA test | delta vs baseline | wall_time_min | summary |
|---|---|---|---|---|---|---|
| (baseline) | — | 57.11 | — | 0 | — | BLIP-base, no synthetic (Khan 2023 Tab.4) |
| (published SelTDA) | — | 60.01 | 26.76 (zs) | +2.90 / +1.67 | — | BLIP-base + SelTDA unfiltered, synth:real=2:1 |

Will append measured runs as inner loop executes.
