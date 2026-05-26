# H1 — Cascade filter (conf + ITM + xcons) on A-OKVQA pseudo-QA

**Protocol locked**: 2026-05-26 (before any data generation or filter run)
**Hypothesis**: Cascading `conf + ITM + xcons` filtering at quantile `keep_top ∈ {0.5, 0.75, 0.9, 1.0}` per gate, applied to SelTDA pseudo-QA on A-OKVQA at `synth:real = 2:1`, yields A-OKVQA val accuracy strictly greater than the unfiltered SelTDA baseline of 60.01% (Khan 2023 Table 4 row 3).
**Prediction (confirmatory)**: Best filter variant (likely `C+I+X` at `keep_top=0.75` per gate) reaches ≥ 61.5 on A-OKVQA val (≥+1.5 abs gain over unfiltered).
**Falsifier**: If no filter variant beats unfiltered SelTDA by ≥ 0.5 absolute points on A-OKVQA val with 3-seed averaging, H1 is refuted. We pivot to H3 (multi-view consistency) or H6 (iterative SelTDA).

## What changes vs. baseline

| Aspect | Unfiltered baseline (SelTDA paper Tab.4 row 3) | H1 |
|---|---|---|
| Pseudo-QA file consumed by `train_vqa.py` | `synthetic_data_raw.json` (51k pairs from 17k unlabeled images, top-p=0.92 nucleus) | `synthetic_data.json` produced by `filter_pseudo.py` |
| Filter | None | Cascade `conf -> ITM -> xcons` per `filtering/gates.py` |
| Threshold mode | n/a | quantile, `keep_top` per gate |
| `train_vqa.py` | unchanged | unchanged (§1.4 hard invariant) |
| Config / seeds / steps | identical | identical |

## Why this is worth testing

1. The spec was written before any score-distribution audit. We don't yet know if `keep_top=0.75` is the right operating point.
2. Khan 2023 Table 4 shows synth:real degrades above 2:1 — consistent with noise dilution. If filtering helps, we should also be able to push the ratio higher (sub-hypothesis H2, tested next).
3. Author-stated noise (Tab.3 of paper): EK 38%, VR 30%, VID 12% answer-incorrect. Filtering targets this directly.

## Pre-run blockers (must resolve before generating data)

| # | Blocker | Owner action |
|---|---|---|
| 1 | ~~**xcons leakage**~~ | **Resolved on server** (user). |
| 2 | No compute env with torch in the current local shell. | Use Vast.ai 1×A5000 per spec §5.6. Or set up local torch env. |
| 3 | `synthetic_data_raw.json` does not exist on disk yet. | Run `examples/generate_synthetic_data.sh` first. ~3h on A5000. |
| 4 | Tests don't run locally because no torch — can't verify implementation cleanly before generation. | Add minimal `pytest -m "not slow"` smoke test that uses stub adapter (no model load). Or run tests on remote compute. |

## Experimental design

### Ablation matrix (Phase 1a: keep ratio fixed at 2:1)

Run all 8 cascade variants from spec §3.4 plus 3 baselines, **all with synth:real = 2:1** (51k synth + 17k real), 3 seeds:

| # | Variant | Conf | ITM | xcons | Expected role |
|---|---|---|---|---|---|
| 0a | No-synthetic | — | — | — | Lower bound (BLIP-base only, no synthetic) |
| 0b | Unfiltered SelTDA | off | off | off | Khan-2023 published number (60.01%) |
| 1 | C | on | off | off | Confidence-only |
| 2 | I | off | on | off | ITM-only |
| 3 | X | off | off | on | Cross-consistency only |
| 4 | C+I | on | on | off | No xcons (cheap) |
| 5 | C+X | on | off | on | No ITM |
| 6 | I+X | off | on | on | No confidence |
| 7 | **C+I+X** | on | on | on | Full cascade |

`keep_top` = 0.75 for every "on" gate. 3 seeds. Report mean ± stdev.

### Threshold sweep (Phase 1b: only on best variant from 1a)

Sweep `keep_top ∈ {0.5, 0.75, 0.9, 1.0}` on whichever single gate dominated 1a, holding the other two at 0.75. 1 seed initial pass, 3 seeds for the winner.

### Calibration audit (parallel to 1a)

Use `scoring_only=true` mode (already supported by `filter_pseudo.py` per CLAUDE.md): compute scores on the full raw pool without dropping. Plot histograms for `s_conf`, `s_itm`, `s_xcons`. Compute ECE if a small labeled subset is available. This gives us evidence of whether the threshold defaults are sane.

### Qualitative audit (spec §5.4)

Random 100 kept + 100 dropped from variant 7. Human-judge (annotator = phong) → label `correct / incorrect / ambiguous`. Compute filter precision/recall vs. judgment. Report in `analysis.md`.

## Metrics

Primary: **A-OKVQA validation accuracy** (open-ended, official metric via `train_vqa.py` evaluator).
Secondary:
- # synthetic pairs retained
- Filter precision/recall vs human judge (100/100 sample)
- ECE before/after temperature scaling (if implementable on labeled val)
- Per-question-type accuracy on A-OKVQA val

## Compute budget

> **Revised 2026-05-26** after code audit — see `research/notes/2026-05-26-compute-efficiency.md`.

| Step | Naive (old) | Revised strategy |
|---|---|---|
| Generation (raw 51k pseudo) | 3h | 3h (unchanged) |
| Filter scoring (`scoring_only=true`, once) | 0.75h × 7 = 5h | **~4h once** (xcons dominates; was underestimated) |
| Offline threshold → 8 JSONs | — | CPU, minutes |
| Train screen (7 var × 1 seed × 3 epoch) | — | **~12–15h** |
| Train confirm (top-2 × 3 seeds × 10 epoch) | 210h | **~60h** |
| Unfiltered baseline retrain | 30h | **0h** (use published 60.01% or `cache/student_weights/checkpoint_09.pth`) |
| **Phase 1a total** | **~260h** | **~85–100h** (~4 days 1 GPU) |

**Do not** run `filter_pseudo.py` seven times for seven variants — run once with all scores, then apply gates offline (CE-01).

## Pre-commit verification

Before running the FIRST experiment, this protocol is committed to git (`research(protocol): H1 cascade-filter`). The hypothesis status in `research/research-state.yaml` will transition from `pending` to `active` only when the first experiment actually starts.

## What success looks like

A figure in `to_human/H1-results.html` showing:
- 9 bars (no-synth, unfiltered, 7 cascade variants), error bars = 3 seeds
- A horizontal line at unfiltered = 60.01
- Filter precision/recall table

If the best variant clears 61.5 (prediction), commit `research(results): H1 cascade-filter — supported, best=C+I+X +X.X over unfiltered`. Move on to H2.

If nothing clears 60.5, run outer-loop reflection — likely **PIVOT** to H3 (multi-view) or H6 (iterative).
