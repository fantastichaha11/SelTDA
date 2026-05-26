# Divergent Brainstorm — SelTDA Improvement Space

- **Date**: 2026-05-26, ~02:35 local
- **Skill applied**: `brainstorming-research-ideas` (Orchestra Research, 10-framework workflow)
- **Mode**: Diverge — generate without filtering. Converge phase happens in a separate file.
- **Constraint**: respect spec §1.4 (no train/eval code changes). Ideas violating this are flagged `[§1.4-breaker]` so they remain in the candidate pool for an alternate "no-constraint" thesis.

## F1 — Problem-First vs Solution-First reframing

**SelTDA-as-stated is solution-first**: "we have a self-training method; what can it accomplish?" Reframing as problem-first surfaces sharper questions.

| Reframe | Two-sentence form |
|---|---|
| PF-A | *PathVQA accuracy at 26.76% (zero-shot SelTDA) is clinically unusable. What method actually clears 60%+ on PathVQA without 100× more annotations?* |
| PF-B | *Authors admit 30% of pseudo-QA are wrong (Tab.3). At what filter precision does the student gain peak — and is the current 3-gate spec close to it?* |
| PF-C | *Numerical questions surged +29.81% via SelTDA because "how-many" generates naturally — but compositional ("how many red cars to the left of the bus") didn't. Why?* |
| PF-D | *VQA-CE (counterexamples) is the only "+1.9%" small gain in robustness — the shortcut-learning failure mode isn't really fixed. What does fix it?* |
| SF-A | *BLIP "knows things it cannot say" (paper's motivating experiment). What is the maximum knowledge extraction rate from a fixed VLM?* |
| SF-B | *DataEnvGym shows iterative student-feedback teacher works on math/code. Can it scale to open-ended VQA with the same compute budget?* |

**Output**: 6 candidate framings. PF-B and SF-A are the most defensible — they don't presuppose a method.

---

## F2 — Abstraction Ladder

Current focus: "Filter pseudo-QA from a BLIP VQG teacher to improve a BLIP student on A-OKVQA / PathVQA."

| Direction | Variant | Note |
|---|---|---|
| UP | "How do we select training data from a noisy generator to maximize student performance?" | Generalizes beyond VQA → distillation curation theory. |
| UP-UP | "What is the optimal data-budget allocation between real and synthetic for any data-scarce SL task?" | Scaling-law shaped. PAC-like bounds possible. |
| DOWN | "Does filter precision/recall on yes/no questions differ from open-ended in a way requiring different gates?" | Concrete experiment. |
| DOWN-DOWN | "For PathVQA staining-type questions specifically (H&E vs IHC), what filter signal transfers?" | Hyper-specific PathVQA carving. |
| SIDEWAYS-1 | "Code generation pipelines filter LLM output with **execution-based** signals (does the code run?). What's the VQA equivalent of 'execution'?" → cycle-consistency = closest analog. |
| SIDEWAYS-2 | "Dense prediction (segmentation) uses **soft pseudo-labels + entropy regularization**. SelTDA uses hard labels only — what changes with soft?" | [§1.4-breaker for true soft training, but duplicate-sampling proxy is fine] |
| SIDEWAYS-3 | "Speech (Whisper) self-training uses **confidence-weighted CTC loss**. Equivalent in VQA = sequence-level weighting per token." | [§1.4-breaker] |

**Output**: 7 variants. UP-UP enables theoretical contribution; DOWN-DOWN enables clinical-publication path; SIDEWAYS-1 (execution-as-cycle-consistency) reframes existing xcons gate compellingly.

---

## F3 — Tension Hunting

Pairs of "everyone wants both" desiderata where the resolution = the contribution.

| Tension | Resolution-shape candidate |
|---|---|
| **Filter strictness ↔ data quantity** | Soft weighting OR active selection on top of strict filter to recover quantity (composition with F9). |
| **Teacher fluency ↔ visual grounding** | Inject visual scaffolding (SAM masks, DINOv2 regions, dense captions) into teacher prompt → ground without re-training. |
| **Diversity ↔ quality** | Per-question-type stratified filtering with type-specific thresholds. Quality threshold per stratum, not global. |
| **Generality ↔ specialization** | Multi-task SelTDA: one VQG_IC trained jointly on A-OKVQA + PathVQA + ArtVQA + RSVQA → conditioned on domain token. |
| **Reproducibility ↔ progress** | Hash-based provenance of every pseudo-QA: `sha256(image_id + decoding_seed + teacher_ckpt_hash)`. Reproducible to the sample. |
| **Compute ↔ rigor** | Sequential testing with confidence intervals — stop seed-3 if seed-2 result is already significant. Cuts 33% compute. |
| **Honesty ↔ self-training amplification** | Pseudo-QA gets a confidence-weighted token mask: the student is told "this label is 0.7-reliable" via duplication rate. |
| **Bias correction ↔ accuracy** | Counterfactual paired generation: for every (I,Q,A), generate (I', Q, A') where I' is the diffusion-counterfactual of I. |

**Output**: 8 tensions, each maps to a concrete resolution idea.

---

## F4 — Cross-Pollination (analogy transfer)

Fields whose techniques map structurally onto pseudo-label selection.

| Source field | Technique | Map onto SelTDA |
|---|---|---|
| **Active learning** | BADGE, BAIT, BALD (informativeness scoring) | Pick *informative* pseudo-QA not just *confident*. Maximize student loss-gradient diversity. |
| **Conformal prediction** | Split-conformal selective classification | Replace `keep_top=0.75` with adaptive threshold giving finite-sample coverage guarantee. |
| **Robust statistics** | Huber loss, M-estimators, MoM | Down-weight score outliers in filter aggregation (resilient to long-tail noise). |
| **Coreset selection** | k-Center, k-Means++, Submodular maximization | Pick representative subset to maximize coverage; prevents over-sampling redundant easy pairs. |
| **Curriculum learning** | Self-paced, OneShot curriculum | Train easy→hard by filter score. SelTDA does no curriculum. |
| **Mixup / CutMix** | Image-region mixup | Generate paired pseudo-QA on mixup images → harder negatives for student. |
| **Distillation** | KD with temperature, attention transfer, intermediate layer matching | SelTDA uses only hard labels. Soft-target distillation = §1.4-breaker. |
| **Game theory (GAN/self-play)** | Generator-discriminator dynamics | Filter as discriminator, teacher as generator — alternating training. Mutual improvement. |
| **Bandits / RL** | UCB, Thompson sampling on data selection | Allocate generation budget across image clusters / question types. |
| **Information theory** | BALD, EIG, mutual-information criteria | Filter score = `I(student_pred ; label \| pseudo)` instead of pure confidence. |
| **Cognitive science (metacognition)** | "Knowing what you don't know" calibration | Reframes uncertainty quantification with strong narrative for paper intro. |
| **Robotics (sim2real)** | DANN / DR / RandConv | Pretrain student on synthetic only, then fine-tune on real (reverse the SelTDA mixing). |
| **Test-time training** | TTT, MEMO, T3A | At deployment, adapt to incoming pathology slide via entropy minimization. |
| **Diffusion generation** | SDXL, ControlNet | Generate counterfactual ablated images for robustness training. |
| **Programming languages** | Type systems & contracts | Declarative QA schemas (object: ?, attribute: ?, relation: ?) enforce well-formedness. |
| **Crowdsourcing literature** | Dawid-Skene, MACE | Multi-rater agreement (multiple teacher generations) → posterior over true label. |

**Output**: 16 cross-pollinated technique candidates. **BADGE/active learning** and **conformal prediction** are most underused in SelTDA-adjacent literature — likely strongest novelty.

---

## F5 — What Changed Since 2023?

SelTDA was finalized late 2022 / early 2023. Two-and-a-half years later:

| Change | Implication for SelTDA |
|---|---|
| **Open-source MLLMs explode**: LLaVA-1.5/1.6/NeXT, Qwen-VL/2-VL, InternVL, Florence-2, IDEFICS-3 | Teacher backbone choice is no longer "BLIP or pay $$". A 7B-13B teacher fits 1 GPU with PEFT. |
| **PEFT/LoRA standard**: QLoRA, LongLoRA, DoRA | Train teacher VQG efficiently even at 13B+. Adapter swap = cheap multi-domain teacher. |
| **GRPO/DPO turnkey**: TRL, OpenRLHF | Filter score = reward. Train teacher with RL to produce only high-score pseudo-QA. |
| **GPT-4V / Claude 3.5 Sonnet as judge** | Replace SBERT exact/cosine match in xcons gate with strong VLM judge. Higher human correlation. |
| **Vision foundation models**: SAM-2, DINOv2, Florence-2 grounding | Scaffold question generation around detected regions/objects → grounded by construction. |
| **Diffusion VLMs**: LLaDA-MedV (Aug 2025) | Non-AR teacher may generate structurally distinct pseudo-QA. Diversity boost. |
| **Long-context models** (1M tokens) | Pack many retrieved contexts cheaply. Enable RAG-VQA without truncation. |
| **Multi-modal CoT benchmarks**: MathVista, MM-Vet, MMMU | Standardized reasoning evaluation. SelTDA paper used none of these. |
| **Synthetic data scaling laws** (Phi-3, LLaMA-3 reports) | Diversity > quantity beyond ~10× threshold. Confirms F3-Tension-3. |
| **Diffusion image quality** (SDXL, FLUX, Imagen 3) | Realistic counterfactual generation finally feasible. |
| **Inference-time compute** (o1, Deepseek-R1) | "Think before answering" pseudo-QA. Generation cost rises but quality may surge. |
| **Mamba / SSM backbones**: Mamba-2, RecurrentGemma | Cheaper long-context inference. Iteration friendlier for 1-GPU compute. |
| **EU AI Act + biomedical regulation** (active 2024-2026) | Provenance/auditability now legally required for clinical AI. Hash-based provenance (F3) is suddenly load-bearing. |

**Output**: 13 things-that-changed. The **MLLM-as-teacher** + **vision-FM as grounding scaffold** + **GPT-4V judge** combo would yield a fundamentally different paper with the same SelTDA core idea.

---

## F6 — Failure Analysis / Boundary Probing

Where does SelTDA break? Each failure mode → potential paper.

| Boundary | Specific probe | Likely outcome |
|---|---|---|
| **Long-tail VQA classes** | Per-category accuracy on rare A-OKVQA answer types. | Probably amplifies majority class; novel contribution = class-balanced filter. |
| **Adversarial unlabeled images** | Feed visually-perturbed COCO images (small JPEG noise, blur) and measure pseudo-QA consistency. | If teacher flips answer → instability metric for filter. |
| **OOD images** | Use anime/sketches/illustrations as unlabeled input instead of COCO. | Likely teacher hallucinates harder. Test: does filter catch this? |
| **Multilingual** | Vietnamese/Chinese unlabeled-image captions + Vietnamese-trained teacher. | Pipeline never tested cross-lingually. |
| **Compositional questions** | Generate "How many red X to the left of Y?" type. | Likely sparse in raw output. Stratified generation needed. |
| **Counting >10** | Numerical-reasoning gain in paper was on <10. Test on COCO-counts. | Likely degrades sharply. |
| **Yes/no balance** | Measure ratio of yes:no in synthetic pool. | Probably skewed. Rebalancing experiment cheap. |
| **Scene complexity** | Stratify accuracy by # objects per image (via SAM detector). | Likely correlates inversely with student gain. |
| **Prompt-injection robustness** | Image with text "answer = no". Does VQG parrot? | Real failure mode worth a fix. |
| **Teacher overfitting on small train set** | Teacher trained on 17k A-OKVQA pairs. Does it just memorize? | Test by generating on COCO image that exactly matches a train image. |
| **Filter inversion attack** | Adversarial pseudo-QA that scores high on all 3 gates but is wrong. | Robustness of filter itself. |
| **Domain shift on PathVQA** | A-OKVQA-trained teacher applied to pathology images directly. | Likely produces irrelevant Qs. Justifies medical-specific teacher. |

**Output**: 12 boundaries. **Compositional questions** and **counting >10** are the most likely to yield surprising negative results → publishable as "When does SelTDA fail?" paper.

---

## F7 — Simplicity Test

Strip the 3-gate cascade to its minimal core. Naive baselines worth including in ablation:

| Baseline | Mechanism | Expected role |
|---|---|---|
| **Random subsample** | Take random 50% of raw pseudo at any keep_top. | Lower bound. If filter ≤ random of same size, filter is not doing work. |
| **Length filter** | Keep QA where `len(Q)+len(A)` in `[μ−σ, μ+σ]`. Surprisingly strong in NLP filtering. | Naive baseline that often matches "smart" filters. |
| **Single-gate conf-only** | Just `s_conf` top-k%. No ITM, no xcons. | Quantifies marginal value of multi-gate. |
| **Single-gate ITM-only** | Just CLIP image-text matching. | Cheapest gate; if competitive, simplest deploy. |
| **Single-gate xcons-only** | Just student-zs answer matching pseudo-A. | Heavy compute but image-text-conditional. |
| **Question-type rebalance** | No filter; just enforce equal yes/no/counting/etc. counts. | Tests F3-Tension-3 directly. |
| **First-N truncation** | Drop pseudo-QA beyond N tokens. | Tests whether long generations are noisier (likely yes). |
| **De-duplication** | Drop near-duplicates (Q-A SBERT cosine > 0.9). | May be the bulk of "noise" — just dupes. |

**Output**: 8 simple baselines. **Random subsample at matched size** is the single most important — if filter doesn't beat random of same N, the filter is useless. The current spec doesn't include this baseline.

---

## F8 — Stakeholder Rotation

Each stakeholder lens generates a distinct research question:

| Stakeholder | Question | Maps to |
|---|---|---|
| **PhD student** | "What's the most defensible chapter contribution?" | Cascade-filter + iterative SelTDA — clear, novel, 1-GPU feasible. |
| **Industry practitioner** | "What's the best accuracy / cost ratio?" | PEFT + single-gate cheap filter + run on Vast.ai for $130. |
| **Pathologist** | "When can I use this on real slides?" | PathVQA-direct + RAG + groundable explanations (GEMeX-style). |
| **Ethicist** | "Where does bias amplification happen?" | Counterfactual augmentation + per-attribute fairness eval. |
| **Adversary** | "How do I fool the system?" | Adversarial pseudo-QA injection / prompt-injection benchmark. |
| **Reproducibility advocate** | "Can I exactly reproduce numbers?" | Hash-based provenance; published synth-data integrity check. |
| **Compute steward** | "Cut compute 4×?" | Coreset selection + early-stopping seeds + amortized scoring. |
| **Theorist** | "Why does filtering work?" | PAC-style bound: student error ≤ f(filter-precision, recall, dataset-size). |
| **Regulator** | "Audit trail for clinical deployment?" | EU AI Act compliance: explainable filter decisions + per-sample lineage. |
| **Educator** | "What's an accessible tutorial demo?" | A-OKVQA filter demo in Colab, no GPU needed. |

**Output**: 10 stakeholder framings. **Theorist (PAC bound)** and **Regulator (auditability)** are most underexplored in SelTDA-adjacent literature.

---

## F9 — Composition / Decomposition

**Compositions** (X + Y → emergent capability):

| Combo | Emergent capability |
|---|---|
| SelTDA + RAG (MMed-RAG style) | Medical-grounded teacher. Addresses Limitation #2. |
| SelTDA + counterfactual augmentation | Robust + bias-mitigated. Addresses Limitation #3. |
| SelTDA + multi-view consistency + soft weighting | Strict quality + high retention. Addresses Limitation #1. |
| SelTDA + active-learning image selection | Pick which COCO image to generate from based on student weakness. |
| SelTDA + diffusion image generation | Generate paired (image, Q, A) where image itself is synthetic — infinite scale. |
| SelTDA + GPT-4V judge | Replace SBERT/CLIP-ITM with strong VLM judge for filter scoring. |
| SelTDA + curriculum | Easy pseudo-QA first, hard later. |
| SelTDA + DPO | Filter score = reward; train teacher to maximize filter-pass rate. |
| SelTDA + SAM grounding | Teacher prompt includes detected object masks → fewer hallucinations. |
| SelTDA + iterative + RAG + counterfactual | "Maximal-SelTDA" — combine everything. Risky but high-ceiling. |

**Decompositions** (split entangled component):

| Decomp | Reveals |
|---|---|
| Separate question generation from answer generation (two model heads). | Maybe Q-gen is the noisy step, not A-gen. Different fix. |
| Per-question-type filters with type-specific thresholds. | Yes/no may need 0.9, open-ended 0.5. |
| Split scoring (cheap CPU) from filtering (set thresholds offline). | Recompute thresholds without re-scoring. Speeds ablation. |
| Decompose teacher into perceiver (visual encoding) and language (decoding) heads. | Maybe visual encoding is the bottleneck. |
| Decompose "noisy pseudo-QA" into (a) wrong Q, (b) wrong A, (c) both — different remedies. | Targets root cause not symptom. |
| Decompose self-training into: warmup → high-confidence-only → broader → exploratory. | Curriculum-shaped self-training. |

**Output**: 16 combinations/decompositions. **Per-question-type filters with type-specific thresholds** is undervalued — cheap, principled, immediately implementable.

---

## F10 — Explain-It Test (two-sentence pitch for each top candidate)

| ID | Pitch |
|---|---|
| **CF-1** "Cascade-Filter" | SelTDA-trained students inherit ~30% noise from their teacher's hallucinated pseudo-QA, diluting data-scarce gains. We add a learned three-gate filter (confidence, image-text matching, cycle-consistency) that drops noisy pairs before training, recovering 1.5-3 absolute points on A-OKVQA. |
| **IT-1** "Iterative-SelTDA" | Self-training stops after one round even though student errors point at exactly which question types still need data. We close that loop with student-feedback-driven teacher re-fine-tuning over 3 rounds, achieving compound gains and converging more quickly than a single large generation pass. |
| **RG-1** "RAG-SelTDA-Medical" | SelTDA fails on medical VQA because the teacher's vocabulary is too generic for pathology. We inject retrieved biomedical context into the teacher's prompt at generation time, closing the medical-vocabulary gap without changing the student backbone. |
| **CT-1** "Counter-SelTDA" | Self-training amplifies the teacher's statistical shortcuts (e.g., always answering "yes" to underdetermined questions). We pair every pseudo-Q with a counterfactual image generated via diffusion, forcing the student to ground answers in visual evidence rather than language priors. |
| **CS-1** "Coreset-SelTDA" | Filtering by confidence wastes data when many high-score samples are near-duplicates. We add submodular coreset selection on top of quality filtering to maximize informational coverage per sample. |
| **AL-1** "Active-SelTDA" | Naive SelTDA generates pseudo-QA uniformly over unlabeled images, but student errors cluster on specific image types. We adaptively pick which unlabeled images to generate from using a BADGE-style informativeness score, reducing total generation cost by 3×. |
| **TS-1** "Type-Stratified-Filter" | One global filter threshold misweights question types — yes/no needs strict, open-ended needs lenient. We compute per-question-type quantiles, retaining proportional pool sizes per stratum and per-type accuracy gains. |
| **PAC-1** "Filter-Bounds" | No prior work derives finite-sample bounds linking filter precision to student error. We provide a PAC-style bound `student_error ≤ f(precision, recall, |D|)` and empirically validate it across A-OKVQA, PathVQA, RSVQA. |
| **JUDGE-1** "GPT-Judge-Filter" | Current xcons gates use SBERT cosine — weak proxy for semantic correctness. We replace it with GPT-4V or Claude 3.5 as judge (used sparingly, with caching) and show ≥5% better filter precision/recall on a 200-sample human-judge benchmark. |
| **GROUND-1** "SAM-Grounded-VQG" | Hallucinated answers come from un-grounded generation. We inject SAM-detected object masks + DINOv2 region tokens into the teacher's prompt so every (Q,A) is conditional on identified regions. Expect ≥10pt drop in factuality-error rate. |
| **DIFF-1** "Diffusion-Counterfactual" | Robustness gains on VQA-CE are only +1.9% with vanilla SelTDA. We pair every (I,Q,A) with (I',Q,A') where I' is SDXL-counterfactual edit (object removal, attribute swap), training the student on both. Expect VQA-CE gain ≥+10%. |
| **MULTI-1** "Multi-Task-SelTDA" | Training one teacher per dataset is wasteful — most VQA datasets share visual reasoning primitives. We jointly train one VQG_IC across A-OKVQA + PathVQA + ArtVQA + RSVQA with a domain-token conditioner, sharing 80% of weights. |
| **DPO-1** "DPO-Teacher" | Filter score is a reward signal — but it's used only post-hoc to discard data. We instead train the teacher via DPO with filter score as the preference signal, so the teacher learns to produce high-quality pseudo-QA upfront. |
| **PROV-1** "Provenance-SelTDA" | EU AI Act 2024 requires per-sample lineage for clinical AI. We hash-stamp every pseudo-QA with `sha256(image_id ‖ teacher_ckpt ‖ decoding_seed)` and publish full lineage. Sets a reproducibility standard for medical self-training pipelines. |

**Output**: 14 candidates with two-sentence pitches.

---

## Diverge Summary

**Total candidates generated**: 6 (F1) + 7 (F2) + 8 (F3) + 16 (F4) + 13 (F5) + 12 (F6) + 8 (F7) + 10 (F8) + 16 (F9) + 14 (F10) = **110 idea fragments**, consolidating into **14 named candidate research directions** (F10 list).

**The 14 named candidates** (alphabetical):
1. AL-1 — Active-SelTDA (image selection by informativeness)
2. CF-1 — Cascade-Filter (current spec direction)
3. CS-1 — Coreset-SelTDA (submodular coverage)
4. CT-1 — Counter-SelTDA (counterfactual diffusion augmentation)
5. DIFF-1 — Diffusion-Counterfactual (VQA-CE robustness)
6. DPO-1 — DPO-Teacher (filter score as reward)
7. GROUND-1 — SAM-Grounded-VQG (visual scaffolding)
8. IT-1 — Iterative-SelTDA (DataEnvGym successor)
9. JUDGE-1 — GPT-Judge-Filter (strong-VLM judge)
10. MULTI-1 — Multi-Task-SelTDA (joint cross-dataset)
11. PAC-1 — Filter-Bounds (theoretical contribution)
12. PROV-1 — Provenance-SelTDA (regulatory-grade lineage)
13. RG-1 — RAG-SelTDA-Medical (MMed-RAG-style)
14. TS-1 — Type-Stratified-Filter (per-type thresholds)

**Next step (separate file)**: Converge phase — apply Explain-It / Problem-First / Simplicity / Stakeholder / Feasibility filters to rank these 14, eliminate weak ones, pick top 3-5 for refinement.

---

## Reflection on this brainstorm session

- The deep-research report (2026-05-26) covered 4 directions (A/B/C/D = Filter / Iterative / RAG / Counterfactual). This brainstorm expands to **14 directions**, of which **10 are genuinely new** vs. the deep-research report.
- New high-novelty candidates surfaced by the framework approach: **CS-1 (Coreset)**, **AL-1 (Active)**, **PAC-1 (Bounds)**, **TS-1 (Type-Stratified)**, **JUDGE-1 (Strong-VLM judge)**, **GROUND-1 (SAM)**, **PROV-1 (Provenance)**, **MULTI-1 (Multi-task)**, **DPO-1 (RL teacher)**.
- F4 (cross-pollination) and F8 (stakeholder) were the most generative frameworks for this problem space. F7 (simplicity) revealed an important *missing baseline*: random-subsample-of-matched-size.
- Anti-pattern check: most candidates are "improve filtering" — i.e., the same problem framing. F5 (what-changed) and F6 (failure analysis) push toward genuinely different framings (better teacher, harder benchmarks).
