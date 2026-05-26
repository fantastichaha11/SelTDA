# Research Log — SelTDA Improvements

Chronological record of research decisions and actions. Append-only.

| # | Date | Type | Summary |
|---|------|------|---------|
| 1 | 2026-05-21 | bootstrap | Spec drafted at `docs/superpowers/specs/2026-05-21-pseudo-label-filtering-design.md` — 3-gate cascade filtering (conf, itm, xcons) with §1.4 invariant: cannot modify training code. |
| 2 | 2026-05-25 | bootstrap | First literature pass via Perplexity → `docs/reports/2026-05-25-perplexity-filtering-and-paper-directions.html`. Partially unreliable (some fabricated arXiv IDs). Kept as historical artifact only. |
| 3 | 2026-05-26 | bootstrap | Full SelTDA paper read end-to-end. Extracted Tables 1, 3, 4, 5, 6, 7 numbers. Confirmed: 30% pseudo-QA noisy, peak synth:real=2:1, PathVQA was zero-shot (+1.67 only), authors' Section-5 limitations form an endorsed roadmap. |
| 4 | 2026-05-26 | bootstrap | Deep research v2 at `docs/reports/2026-05-26-deep-research-seltda-improvements.md` — verified-arXiv survey of filtering / iteration / RAG / counterfactual directions. 30+ HF-indexed references, abstract-level claims flagged. Top finding: DataEnvGym (arXiv 2410.06215) is the direct iterative successor by the same author (Zaid Khan). |
| 5 | 2026-05-26 | bootstrap | Autoresearch skill installed via Orchestra Research npm package. Workspace seeded at `research/`. 7 hypotheses (H1-H7) formed from the 5-phase roadmap. Continuous /loop set up at 20-minute interval (cron `3,23,43 * * * *`, job `7c4f57bf`, session-only, 7-day auto-expire). |
| 6 | 2026-05-26 | inner-loop | Repo audit before H1 launch: `filtering/` module is implemented (7 files, ~330 LOC), `filter_pseudo.py` entry point ready, `configs/filter_pseudo.yaml` present. No `synthetic_data_raw.json` on disk yet. **Pre-run blocker found**: xcons gate config points at SelTDA-trained student → information leakage (violates spec §3.3). Logged in findings.md Lessons. |
| 7 | 2026-05-26 | inner-loop | H1 (cascade filter) protocol locked at `research/experiments/H1-cascade-filter/protocol.md` — 8 cascade variants × 3 seeds × synth:real=2:1, plus threshold sweep on best variant, plus scoring-only calibration audit, plus 100-sample qualitative audit. Compute budget ≈ 260h × A5000 ≈ $130 ≈ 11 days. Pre-commit (this commit) precedes any run. |
| 8 | 2026-05-26 02:35-03:00 | bootstrap | Cycle 1 — **brainstorming-research-ideas skill** applied. All 10 frameworks (F1-F10) walked through. 110 idea fragments produced, consolidated to 14 named candidates with two-sentence pitches. Written to `research/ideation/2026-05-26-divergent-brainstorm.md`. New candidates beyond deep-research-2026-05-26: AL-1 (Active), CS-1 (Coreset), PAC-1 (Bounds), TS-1 (Type-Stratified), JUDGE-1 (Strong-VLM judge), GROUND-1 (SAM), PROV-1 (Provenance), MULTI-1 (Multi-task), DPO-1 (RL teacher). |
| 9 | 2026-05-26 03:00-03:15 | bootstrap | Cycle 2 — **creative-thinking-for-research skill** applied. 8 cognitive frameworks. 5 transformational candidates surfaced: EB-1 (Energy-Based), RW-1 (Reward-Shaped VQG), MC-1 (MCMC-SelTDA), CS-X (Coreset-Stratified, Janusian synthesis of CS-1 and TS-1), PV-1 (Pivot-VQG, from NEG(C2)). Written to `research/ideation/2026-05-26-creative-thinking-deepening.md`. |
| 10 | 2026-05-26 03:15-03:30 | bootstrap | Cycle 3 — literature evidence pass on EB-1, RW-1, CS-X via HF paper-search. Confirmatory + counter-evidence + implications. Top candidate ranking shifted: CS-X promoted to #1 (ZCore arXiv 2411.15349 directly applicable); RW-1 demoted to #3 (Rafailov 2024 arXiv 2406.02900 shows reward hacking in DPO — must mitigate via held-out judge + KL budget). Written to `research/literature/2026-05-26-EB-RW-CS-deep-dive.md`. |
| 11 | 2026-05-26 03:30-03:45 | bootstrap | Cycle 4 — deep-dive on RW-1 design with hacking mitigation. Janusian reframe: filter is simultaneously reward and adversary; held-out judge becomes measurement instrument, not regularizer. Detailed algorithm pseudo-code + required baselines (BoN-only, filtered-without-DPO, random-subsample, no-KL drift). Written to `research/ideation/2026-05-26-RW1-design-deep-dive.md`. Bonus find: Karan & Du 2025 (arXiv 2510.14901) "Reasoning with Sampling" provides MCMC-for-base-model-improvement precedent for MC-1. |
| 12 | 2026-05-26 04:01 | conclude | **Autoresearch loop stopped on schedule** per user request (run until 4am). CronDelete on job 7c4f57bf. Final synthesis HTML at `research/to_human/2026-05-26-overnight-summary.html`. State of project: bootstrap complete, ideation phase complete, H1 protocol locked (pre-run), xcons-leakage blocker identified, top-12 candidate ranking established. Next step requires Vast.ai compute + xcons fix before first experiment. |

<!-- Entry types:
  bootstrap    — initial scoping, literature search, hypothesis formation
  inner-loop   — experiment run and result
  outer-loop   — synthesis, reflection, direction decision
  pivot        — change in research direction
  report       — progress presentation generated
  conclude     — decision to finalize and write paper
-->
