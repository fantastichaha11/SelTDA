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

<!-- Entry types:
  bootstrap    — initial scoping, literature search, hypothesis formation
  inner-loop   — experiment run and result
  outer-loop   — synthesis, reflection, direction decision
  pivot        — change in research direction
  report       — progress presentation generated
  conclude     — decision to finalize and write paper
-->
