# Literature Deep-Dive — Evidence for/against EB-1, RW-1, CS-X

- **Date**: 2026-05-26 ~03:05 local
- **Sources**: Hugging Face paper index (verified arXiv IDs). Abstract-level reads. Full text TBD.
- **Companion to**: `research/ideation/2026-05-26-creative-thinking-deepening.md`

This file rapidly tests the three transformational candidates from creative-thinking against actual published literature. Each candidate gets: (1) confirmatory evidence; (2) **counter-evidence** (most important); (3) implications for the SelTDA thesis.

---

## EB-1 Energy-Based SelTDA

### Confirmatory evidence
- **EBMs for Continual Learning** (Li & Du et al., 2020, arXiv [2011.12216](https://arxiv.org/abs/2011.12216)). EBMs trained with contrastive divergence reduce interference. Confirms that EBM-as-filter is a viable architecture pattern.
- **Improved Universal Sentence Embeddings with EBM** (Jiang et al., 2022, arXiv [2203.06875](https://arxiv.org/abs/2203.06875)). Energy-based Hinge loss improves domain-shift robustness for sentence embeddings — relevant to xcons gate's SBERT scoring.
- **Curriculum Labeling** (Cascante-Bonilla et al., 2020, arXiv [2001.06001](https://arxiv.org/abs/2001.06001)). Pseudo-labeling + curriculum + restart achieves competitive SSL — confirms curriculum is a valid SelTDA add-on (compatible with EB-1 staging).

### ⚠️ Counter-evidence (must address in thesis)
- **EBMs vs. CL for VQA** (Shevchenko et al., 2022, arXiv [2206.14355](https://arxiv.org/abs/2206.14355)). Direct apples-to-apples: self-supervised CL **outperforms** EBMs for training VQA representations. Specifically tested on CLEVR + systematic generalization + OOD detection + calibration.
  - **Why this doesn't kill EB-1**: their EBM is for *representation learning* (replacing the VQA backbone). My EB-1 is for *filter scoring* (replacing the 3-gate cascade). Different application of EBM machinery. But the paper's finding that EBMs lag CL on calibration is a yellow flag — exactly the property I want to leverage. **Need to read full text** before claiming EB-1 advantage.

### Implications
| Decision | Impact on EB-1 |
|---|---|
| EB-1 must explicitly frame itself as **filter calibration**, not representation learning. | Avoids being killed by Shevchenko 2022. |
| EB-1 must include a baseline = **logistic-regression-over-3-gate-scores** (calibrated). | If logistic beats EB-1, the energy framing adds nothing. |
| EB-1 should leverage Curriculum Labeling-style restart-after-each-round to avoid confirmation bias drift. | Cheap addition, paper-citable. |
| EB-1 should report calibration metrics (ECE, NLL) not just accuracy. | EBM's claim-to-fame is calibration; must measure it. |

---

## RW-1 Reward-Shaped VQG

### Confirmatory evidence
- **Filtered DPO** (Morimura et al., 2024, arXiv [2404.13846](https://arxiv.org/abs/2404.13846)). Confirms text quality in DPO preference data matters MORE than for reward-model RLHF. Implies: filter score *as preference data* should work well for VQG-DPO. Quality monitoring is built in.
- **West-of-N** (Pace et al., 2024, arXiv [2401.12086](https://arxiv.org/abs/2401.12086)). Synthetic preference data via Best-of-N improves reward models. Directly applicable: use the 3-gate filter to score N teacher samples per image, take best as "preferred", worst as "rejected", DPO the teacher.
- **D3PO** (Yang et al., 2023, arXiv [2311.13231](https://arxiv.org/abs/2311.13231)). DPO for diffusion without reward model. Confirms DPO-on-generators is viable beyond LLMs.
- **RLP — Reward Learning on Policy** (Lang et al., 2024, arXiv [2403.19279](https://arxiv.org/abs/2403.19279)). Reward refinement using policy samples — directly maps to "filter-as-reward keeps adapting".

### ⚠️ Counter-evidence (must address)
- **Scaling Laws for Reward Model Overoptimization in DAA** (Rafailov et al., 2024, arXiv [2406.02900](https://arxiv.org/abs/2406.02900)). Reward over-optimization / **reward hacking is empirically demonstrated in DPO**, similar to classical RLHF. KL budget matters. **This is the #1 failure mode RW-1 must mitigate.**
- **New Desiderata for DPO** (Hu, He, Wipf, 2024, arXiv [2407.09072](https://arxiv.org/abs/2407.09072)). DPO has limitations in interpolating between reference and preferences — yields low-quality responses if not bounded. Likely to bite RW-1 (teacher drifts to filter-gaming).

### Implications
| Decision | Impact on RW-1 |
|---|---|
| RW-1 **must** include a held-out judge (separate filter or GPT-4V) to detect reward hacking. | Without this, RW-1 is undefendable. |
| RW-1 **must** report KL divergence from baseline VQG_IC at every DPO step. | Provides the "reward hacking failure mode" data the paper needs. |
| Reward-hacking as a finding becomes **its own contribution**: "We show filter-as-reward induces specific failure modes; characterize them." This pivots a possibly-negative result into a positive one. | Defensive framing. |
| RW-1's compute footprint: DPO of a 600M-param VQG_IC on 50k preference pairs ≈ 8-12h A5000 — feasible. | 1-GPU compatible. |
| Filter-as-reward dataset construction = West-of-N style (cheap). | No human annotation needed. |

---

## CS-X Coreset-Stratified Filter

### Confirmatory evidence
- **ZCore — Zero-Shot Coreset Selection** (Griffin et al., 2024, arXiv [2411.15349](https://arxiv.org/abs/2411.15349)). Selects representative subsets from **unlabeled** pools using foundation-model embeddings. *Specifically iterates subspace sampling for coverage vs. redundancy.* Directly applicable to SelTDA pseudo-pool. Achieves SOTA at low data rates.
- **TAGCOS** (Zhang et al., 2024, arXiv [2407.15235](https://arxiv.org/abs/2407.15235)). Task-agnostic gradient-clustered coreset selection for instruction tuning — drops to ~5% data with maintained performance.
- **Coverage-centric Coreset Selection** (Zheng et al., 2022, arXiv [2210.15809](https://arxiv.org/abs/2210.15809)). Distribution coverage + importance — handles high pruning rates. Precisely the regime SelTDA filtering operates in.
- **INGENIOUS** (Renduchintala et al., 2023, arXiv [2305.06677](https://arxiv.org/abs/2305.06677)). Submodular optimization for LM pretraining data — confirms submodularity scales to deep-learning data selection.

### ⚠️ Counter-evidence
- None found directly. Coreset selection is well-established and consistently positive. Lowest risk of the three.
- **One caveat**: TAGCOS uses gradient clustering — requires gradients from student model. For SelTDA this means a non-trivial student forward+backward per pseudo-sample. Cost may be prohibitive at 50k+ pool. → use **ZCore-style embedding clustering** (no gradients) for the SelTDA implementation.

### Implications
| Decision | Impact on CS-X |
|---|---|
| Combine CS-X with TS-1 (Type-Stratified): apply ZCore-style coreset *within each question type*. | Yields per-stratum diversity, not just global coverage. Original contribution. |
| Pre-compute BLIP / DINOv2 / CLIP embeddings of all 51k pseudo-QA → cluster → diversity score → multiply by quality score. | Cheap engineering; total cost ≈ 1 GPU-hour. |
| Headline experiment: **at matched kept-set size N, does filter+coreset beat filter-only? Does it beat coreset-only?** | The 2×2 factorial answers whether coreset is additive over quality filter — the key research question. |
| CS-X is the lowest-risk highest-paper-output candidate. Should be Phase 1.5 of the thesis (after CF-1 + TS-1). | Schedule decision. |

---

## Cross-cutting insights from this literature pass

1. **Reward hacking is a published, real phenomenon** (Rafailov 2024). RW-1 cannot ignore it. The mitigation pattern from RLHF literature transfers: KL budget + held-out judge + monitoring. This is **the strongest mitigation case I can make** for RW-1 in the thesis.
2. **ZCore (Nov 2024) is the most directly transferable algorithm** in this entire literature scan. It does *exactly* what SelTDA filtering needs: zero-shot coreset selection on an unlabeled pool. Code presumably public. Wraps the spec's filtering module nicely.
3. **Curriculum + restart-each-round is a cheap addition** to any iterative variant (IT-1, RW-1). Cascante-Bonilla 2020 shows restart prevents confirmation bias accumulation across rounds. Should be in the spec for IT-1.
4. **Calibration metrics (ECE, NLL) are missing from SelTDA's published evaluation.** Whatever direction wins, adding these metrics is a thesis-level contribution because no SelTDA-adjacent paper reports them.
5. **VQA-specific evidence against pure EBMs** (Shevchenko 2022) means EB-1 must be *carefully framed* — filter-calibration EBM, not representation-EBM. This is a critical narrative point.

---

## Re-ranking the top-12 after this evidence pass

| New Rank | ID | Why this rank shifted |
|---|---|---|
| 🥇 | **CS-X Coreset-Stratified-Filter** | Strongest literature support (ZCore 2024 directly applicable). Lowest risk. Cheap. Compounds with TS-1. **Promote to Phase 1.5.** |
| 🥈 | **IT-1 Iterative-SelTDA** (with curriculum + restart) | Cascante-Bonilla 2020 supports curriculum-restart. DataEnvGym is by SelTDA's author. Schedulable as Phase 3. |
| 🥉 | **RW-1 Reward-Shaped-VQG** (with KL budget + held-out judge) | High novelty but high risk. Reward hacking must be measured. Could yield "positive finding + characterized failure mode" double contribution. |
| 4 | **CF-1 + TS-1 Cascade-Filter + Type-Stratified** | Foundation. Run Phase 1 regardless. |
| 5 | **EB-1 Energy-Based-SelTDA** | Reframe to filter-calibration; baseline against logistic regression. Risk of being beaten by simpler calibration. |
| 6 | **MC-1 MCMC-SelTDA** | Theory thesis chapter; depends on having iterative results from IT-1 first. |
| 7 | **RG-1 RAG-SelTDA-Medical** | High value but largely independent track (medical branch). Schedule alongside Phase 1. |
| 8 | **PV-1 Pivot-VQG** | Cheap to test; could be folded into IT-1's multi-stage generation. |
| 9-12 | DIFF-1, GROUND-1, JUDGE-1, AL-1 | Defer until Phase 3+ unless one becomes a fast follower. |

---

## What to read next (priority, full-text)

1. **ZCore arXiv 2411.15349** — implementation details for SelTDA wiring.
2. **Filtered DPO arXiv 2404.13846** — quality-monitoring template for RW-1.
3. **Rafailov 2024 arXiv 2406.02900** — reward hacking failure mode characterization (for RW-1 mitigation chapter).
4. **DataEnvGym arXiv 2410.06215** — Khan's iterative successor (for IT-1).
5. **MMed-RAG arXiv 2410.13085** — for RG-1 if PathVQA branch becomes Phase 2.
