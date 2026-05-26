# Creative Thinking Deepening — SelTDA Paradigm Transformation

- **Date**: 2026-05-26, ~02:50 local
- **Skill applied**: `creative-thinking-for-research` (Orchestra Research, 8 cognitive frameworks)
- **Companion to**: `2026-05-26-divergent-brainstorm.md` (14 candidates from operational brainstorm)
- **Goal**: Push beyond exploratory/combinational creativity into **transformational** ideas — ones that change the rules of the SelTDA paradigm itself, not just its parameters.

---

## Phase 1 — Map the Space (Constraint Manipulation + Adjacent Possible)

### F4 — SelTDA's Constraints, Reclassified

The published SelTDA pipeline carries many constraints. Most have never been challenged.

| # | Constraint | Type | Negotiable? |
|---|---|---|---|
| C1 | The teacher and student must share the same backbone (BLIP). | **Soft** | Yes — different backbones could be more efficient or stronger. |
| C2 | The teacher generates `(Q,A)` jointly as one autoregressive sequence. | **Hidden** | Yes — could be two stages (Q first, then A from Q). |
| C3 | The unlabeled image distribution must match the labeled distribution. | **Soft** | Yes — out-of-distribution images could supply harder diversity. |
| C4 | One teacher trained once, then used N times to generate pseudo-QA. | **Hidden** | Yes — teacher could be retrained mid-generation based on student feedback (DataEnvGym). |
| C5 | Filtering is a *post-hoc selector*, not part of training. | **Hidden** | Yes — filter could be a *reward model* training the teacher in-loop. |
| C6 | Pseudo-QA is a single "ground truth" used as a hard label. | **Hidden** | Yes — could be a *distribution* over plausible answers. |
| C7 | The student is trained from scratch on real ∪ synthetic each ablation. | **Hard** (§1.4 invariant for this thesis) | No — but for general research, false. |
| C8 | Evaluation is a single-number accuracy on a static benchmark. | **Soft** | Yes — could be a curve, a distribution, a per-question-type breakdown. |
| C9 | The unit of synthesis is `(image, question, answer)`. | **Hidden** | Yes — could synthesize `(image, dialog turn 1, turn 2, …)` — multi-turn. |
| C10 | "Quality" is judged in expectation over the pool. | **Hidden** | Yes — could be judged per-sample given a downstream skill profile. |
| C11 | The pipeline runs once → student deployed. | **Soft** | Yes — student could keep adapting at test time (TTT). |
| C12 | Pseudo-QA must be in the same language as the labeled data. | **Soft** | Yes — multilingual augmentation could broaden coverage. |
| C13 | The image domain is "natural photos". | **Hidden** | Yes — synthetic/cartoon/medical/satellite could be mixed. |
| C14 | "More data is better" — student training scales with `|D|`. | **Hidden** | Probably false above a noise threshold (Tab.4 shows degradation at 4×). |
| C15 | The teacher is a single model. | **Hidden** | Yes — could be an *ensemble* of teachers from different finetuning runs. |
| C16 | The student's gradient signal is uniform across pseudo-samples. | **Hidden** | Yes — could be a per-sample weight (= soft filtering). |

**Highest-leverage hidden constraints to drop**: C2, C4, C5, C6, C9, C10, C14, C15.

### F7 — Adjacent Possible for SelTDA (what is newly feasible since Khan 2023?)

| Enabler emerged 2024-2026 | What it unlocks for SelTDA |
|---|---|
| **Open-weight MLLMs 7-13B with LoRA-finetuning fitting on 24GB** (Qwen2-VL, InternVL2, LLaVA-NeXT) | Teacher is no longer trapped at BLIP-base scale. Drop C1. |
| **GPT-4V / Claude 3.5 / Gemini 1.5 Pro as scored judge** | Replace SBERT semantic match with strong-VLM judgment. Drop weak-proxy constraint. |
| **SDXL + Flux for realistic counterfactual image edits** | Generate (I', Q, A) pairs at scale. Drop "image is given" constraint. |
| **DPO / GRPO / RLAIF infrastructure (TRL, OpenRLHF)** | Filter score becomes a reward signal training the teacher in-loop. Drop C5. |
| **SAM-2 + DINOv2 region tokens + Florence-2 grounding** | Visually scaffold the teacher's generation. Drop ungrounded-generation constraint. |
| **Inference-time reasoning (o1-style, DeepSeek-R1)** | Pseudo-QA can include explicit reasoning chains. Drop C9 (Q-A only). |
| **Multimodal long-context (1M+ tokens)** | Retrieve K biomedical contexts cheaply for RAG. |
| **EU AI Act + clinical AI standards (2024-2026)** | Provenance and per-sample lineage become regulatory requirements. Creates a publishing niche. |
| **Embedding-based deduplication at scale** (FAISS, ScaNN) | Coreset selection on millions of pseudo-samples is now O(seconds). |

**Convergent adjacent possibles** (intersection of multiple enablers): 
- **MLLM teacher + GPT-4V judge + DPO loop** = a fundamentally different SelTDA where the teacher self-improves via reward signals from a stronger judge. This is the "GPT-4V-distilled-self-training" idea — a paradigm shift, not an extension.

---

## Phase 2 — Generate Disruptions

### F5 — Negate Hidden Constraints

| Negation | What emerges |
|---|---|
| **NEG(C2)**: "Q and A are NOT generated jointly." | **Pivot-VQG**: Stage 1 = teacher generates *question* from image alone; Stage 2 = a separate (possibly different) model generates *answer* from (image, question). Decouples question creativity from answer correctness — each can be filtered independently. |
| **NEG(C4)**: "The teacher is NOT a fixed snapshot." | **Streaming-SelTDA**: teacher's weights update continuously as the student's skill profile evolves. Sample-by-sample online learning. |
| **NEG(C5)**: "Filtering is NOT post-hoc — it shapes the teacher." | **Reward-shaped VQG**: filter score is the reward for DPO/GRPO training the teacher. Teacher learns to produce pseudo-QA that the filter cannot drop. (This is also a danger: filter-gaming. Mitigation: hold-out judge.) |
| **NEG(C6)**: "Pseudo-A is NOT a single answer." | **Distributional-VQG**: teacher emits `(Q, p(A))` — a distribution over plausible answers (e.g., top-5 candidates with calibrated probs). Student trains with soft labels. *[§1.4-breaker for soft training, but duplicate-sampling proxy works.]* |
| **NEG(C9)**: "The unit is NOT (image, Q, A)." | **Dialogue-SelTDA**: synthesize multi-turn `(image, Q1, A1, Q2, A2, …)` where each turn conditions on the previous. Coverage of compositional reasoning rises. |
| **NEG(C10)**: "Quality is NOT pool-average." | **Skill-profile-aware filter**: kept pseudo-QA optimizes the *student's specific weakness profile*. Filter is conditioned on a per-student feedback signal. (Connects to DataEnvGym.) |
| **NEG(C13)**: "Image domain is NOT given — it's chosen." | **Curated-SelTDA**: a budget for which unlabeled images to even generate from. Use clustering + active learning to pick images that maximize diversity gain per dollar. |
| **NEG(C14)**: "More data is NOT better." | **Optimal-stop SelTDA**: empirically find the saturation point per dataset, deploy with exactly that size. Cuts $$$ without accuracy loss. (Companion finding: where saturation point is.) |
| **NEG(C15)**: "Teacher is NOT a single model." | **Teacher-Council**: ensemble 3-5 teachers from different seeds / different question-type curricula. Each pseudo-QA scored by consensus. |
| **NEG(C16)**: "Gradient is NOT uniform per sample." | **Per-sample-weighted SelTDA**: pseudo-QA contribution to loss is proportional to filter confidence. *[§1.4-breaker, but feasible via duplicate-sampling proxy.]* |

**The most transformational among these**: NEG(C5) — Reward-shaped VQG. Replaces "filter-as-discard-step" with "filter-as-training-signal". A fundamentally different teacher training procedure.

### F1 — Bisociation (cross-product with distant fields)

Picking three structurally rich domains: **statistical physics (energy-based models)**, **immunology (self/non-self discrimination)**, **economics (mechanism design)**.

#### A. Statistical Physics × SelTDA

| | Pseudo-QA generation | Filtering | Self-training |
|---|---|---|---|
| **Energy landscape** | Each pseudo-QA has an energy `E(I,Q,A)` — low = plausible, high = noise. Teacher samples from `exp(-E/T)`. | Filter = threshold on `E`. Replace heuristic gates with one learned energy function. | Student loss = pull "low-E" samples down further → trains an energy-based model implicitly. |
| **Phase transitions** | Above noise threshold (Tab.4 ratio 4×), accuracy drops sharply — that's a phase transition. Locate it precisely. | Filter strictness parameter = order parameter; sweep to find critical point. | Train across the phase boundary; characterize collapse modes. |
| **Renormalization group** | Coarse-grain pseudo-QA into types/clusters; apply filter at multiple scales. | Per-scale filter → preserved across scales = invariant features. | Self-training as RG flow toward fixed point of "best generalizing model". |

**Bisociative idea**: **Energy-Based SelTDA** — train a single energy function `E_φ(I,Q,A)` that replaces the 3-gate cascade entirely. Score = `-E_φ`. Filtering reduces to thresholding. Theoretical link: filter precision/recall mapped to free energy. Strong novelty.

#### B. Immunology × SelTDA

| Immunology concept | SelTDA mapping |
|---|---|
| **Self/non-self discrimination** (T-cell negative selection) | Filter learns to recognize "self" (training-distribution-correct pseudo-QA) and reject "non-self" (off-distribution noise). |
| **Clonal selection (positive feedback)** | Pseudo-QA that successfully helps the student get duplicated more in next round. Like B-cell expansion. |
| **Memory cells** | After convergence, save a "memory pool" of highest-impact pseudo-QA for later fast adaptation. |
| **Affinity maturation** | Teacher's outputs progressively refine over rounds via small mutations + selection. |
| **Autoimmune disease** (model attacks itself) | Bias amplification = self-attack. Diagnostic: where does the student misclassify in ways the teacher would also misclassify? |

**Bisociative idea**: **Clonal-Selection SelTDA** — each pseudo-QA's "fitness" = downstream student accuracy contribution. Reproduce high-fitness ones, mutate (paraphrase), discard low-fitness. Population-genetics algorithmic frame.

#### C. Economics (Mechanism Design) × SelTDA

| Economic concept | SelTDA mapping |
|---|---|
| **Auction theory** | Pseudo-QA "bid" for inclusion in training pool via score. Filter is the auctioneer. Second-price (Vickrey) auction = truthful score elicitation. |
| **Incentive alignment** | Teacher's reward (from RL) must align with student's accuracy, not just filter-pass rate. Watch for filter-gaming Goodhart effect. |
| **Mechanism design (revelation principle)** | Design the teacher's training objective such that *honest* generation is the optimal strategy. |
| **Public goods game** | Pseudo-QA pool is a public good — overuse degrades quality. Optimal contribution = limited. |

**Bisociative idea**: **Vickrey-Filter SelTDA** — adapt second-price auction logic to set per-sample weights, where the marginal contribution of each pseudo-QA (computed via influence functions) sets its inclusion weight.

### F2 — Problem Reformulation

Current problem statement: *"Filter noisy pseudo-QA from a VLM teacher to improve student VQA accuracy."*

Each reformulation changes a specific dimension:

| Dimension changed | New formulation | What it enables |
|---|---|---|
| **Change the objective** | "Maximize the *information gain* from each generated pseudo-QA, not the *count* of kept samples." | Active selection (AL-1) becomes a corollary. |
| **Change the formalism** | "Frame self-training as an MCMC sampler over a posterior `p(student_params \| unlabeled_data)`. Filter = acceptance ratio." | Bayesian self-training; principled stopping criterion (chain convergence). |
| **Change the granularity** | "Filter at the token level, not the sample level — each pseudo-A token gets a confidence." | Selective masking during student loss. Matches CTC literature. |
| **Change the agent** | "Don't ask 'how should the model learn?' Ask 'how should the data teach?' — let pseudo-QA samples self-organize into a curriculum." | Self-organizing curriculum learning. |
| **Change the timescale** | "Don't generate-then-filter — generate, score, train, evaluate in one online loop." | True online SelTDA. |
| **Invert direction** | "Don't filter to remove bad pseudo-QA — *generate* counterfactual *good* pseudo-QA targeted at the student's gaps." | Targeted synthesis, not blind filtering. |
| **Change the cost model** | "Account for the *opportunity cost* of generating one more pseudo-QA vs. fine-tuning teacher better." | Joint allocation of compute between teacher-improvement and data-generation. |

**Strongest reformulation**: **"Frame self-training as MCMC over the student-parameter posterior."** This recasts the entire problem in Bayesian terms. Filter precision = acceptance rate. Iteration count = chain length. Convergence diagnostics apply directly. Could yield a clean theoretical thesis chapter.

---

## Phase 3 — Deepen Promising Leads

### F3 — Structure-Mapping for the Top Disruptions

Pick the most promising 3 and validate the analogy depth.

#### Energy-Based SelTDA — structure-mapping to Boltzmann machines

| Boltzmann machine concept | EB-SelTDA equivalent | Mechanism preserved? |
|---|---|---|
| Energy function `E(v,h)` | `E_φ(I, Q, A)` | ✓ |
| Free energy `F = -log Z` | Filter calibration constant | ✓ |
| Temperature `T` | Filter strictness (1-quantile) | ✓ |
| Contrastive divergence training | Positive samples = real `(I,Q,A)` from train.json; negative samples = teacher-generated pseudo | ✓ |
| Mode collapse | Filter retains only one question type | ✓ (predictive!) |

**Validation**: Yes — the mechanism transfers. Predicts: at low temperature (`keep_top=0.5`), filter risks mode collapse to easiest question type. This is an empirically testable claim. **Strong analogy.**

#### Reward-Shaped VQG (NEG-C5) — structure-mapping to RLHF

| RLHF concept | Reward-Shaped VQG equivalent | Mechanism preserved? |
|---|---|---|
| SFT base model | Teacher fine-tuned on train.json (the existing VQG_IC) | ✓ |
| Reward model | The 3-gate filter score (or learned reward from human labels) | ✓ |
| KL constraint to base | Penalty against teacher drifting too far from VQG_IC (preserve coverage) | ✓ |
| PPO/DPO update | Apply to teacher conditional on pseudo-QA samples | ✓ |
| Reward hacking | Teacher generates QA that passes filter but is wrong | ✓ (must mitigate with held-out judge) |

**Validation**: Yes — RLHF infrastructure directly applies. Predicts reward-hacking will emerge unless judge is held out. **Strong analogy.**

#### MCMC-SelTDA — structure-mapping to Metropolis-Hastings

| MH concept | MCMC-SelTDA equivalent | Mechanism preserved? |
|---|---|---|
| Proposal distribution `q(θ' \| θ)` | Teacher generates pseudo-QA given current student | ✓ |
| Acceptance probability | Filter probability of keeping the sample | ✓ |
| Stationary distribution | Optimal student-parameter posterior | ✓ (in principle) |
| Burn-in period | First N rounds of iterative SelTDA before measuring | ✓ |
| Detailed balance | Filter must satisfy a calibration condition for the chain to converge | Predictive — testable. |
| Effective sample size | Real informative pseudo-QA after autocorrelation | ✓ |

**Validation**: Partial — detailed balance is non-trivial to enforce. But the framing yields *new metrics* (ESS, R-hat) that have never been applied to self-training. **Moderate analogy with strong narrative power.**

### F6 — Abstraction Laddering for Top 3 Ideas

**Energy-Based SelTDA**:
- UP: "All self-training filters are implicitly energy functions; we make this explicit and learnable."
- DOWN: "Specifically for yes/no VQA, the energy reduces to a sigmoid of CLIP-ITM — simpler closed form."
- SIDEWAYS: "Energy-based models also unify constrastive self-supervised pretraining (SimCLR, MoCo)."

**Reward-Shaped VQG**:
- UP: "Any post-hoc data filter can be lifted to an in-loop reward — a general procedure."
- DOWN: "Specifically for clinical NLP, this is how MedLM teachers can be aligned with clinician preferences."
- SIDEWAYS: "Code-execution feedback in code LLMs is the same idea (filter = passing tests)."

**MCMC-SelTDA**:
- UP: "All iterative self-training is implicitly MCMC; convergence diagnostics apply."
- DOWN: "On A-OKVQA specifically, chain R-hat across seeds predicts final accuracy."
- SIDEWAYS: "MCMC analysis of diffusion model training is a parallel research thread."

### F8 — Janusian Synthesis of Apparent Contradictions

| Contradiction | Naive resolution | Janusian synthesis |
|---|---|---|
| Filter strictly (clean) **and** retain quantity (large) | Compromise: medium strictness | **Multi-scale filter**: strict on highly-informative samples (kept few), lenient on redundant samples (kept many) — exploits coreset-selection insight (CS-1 from divergent brainstorm). |
| Teacher fluent **and** visually grounded | Compromise: smaller more grounded teacher | **Dual-path teacher**: language-only draft → visual revision pass. Each pass optimizes one criterion. |
| Diversity **and** quality | Compromise: tolerate some noise for diversity | **Stratified-quality filter** (TS-1): per-question-type quantile + per-image-cluster cap. Quality enforced *within* each diversity stratum, not globally. |
| Generality **and** specialization | Choose one or train both | **Mixture-of-experts teacher**: shared backbone + per-domain expert heads activated by domain token. One model, many specializations. |
| Pseudo-QA hard label **and** uncertainty awareness | Compromise: soft label (breaks §1.4) | **Curriculum + multi-rate sampling**: easy samples shown often early, hard samples shown often late — encodes uncertainty via training-time exposure pattern without changing the loss. |

**Strongest Janusian move**: **Multi-scale / coreset-stratified filter** — the strictness-vs-quantity tension is resolved by recognizing that "quantity that adds information" ≠ "raw count". Drop the framing entirely.

---

## Phase 4 — Two-Sentence Tests for the 5 New Transformational Candidates

Beyond the 14 candidates in `2026-05-26-divergent-brainstorm.md`, the creative-thinking phase yielded 5 fundamentally new (not just combinational) candidates. Each must pass the two-sentence test.

| ID | Pitch |
|---|---|
| **EB-1 Energy-Based-SelTDA** | Existing filters use three independent heuristic gates whose interactions are unprincipled and whose combined behavior is hard to calibrate. We replace the cascade with a single learned energy function `E_φ(I,Q,A)` trained contrastively against real labeled data, unifying the gates and giving us principled calibration and convergence guarantees from energy-based modeling literature. |
| **RW-1 Reward-Shaped-VQG** | Filters are currently post-hoc discard steps — most generated pseudo-QA is thrown away. We instead use the filter score as a DPO reward signal training the teacher in-loop, so the teacher learns to produce only filter-passing pseudo-QA upfront, cutting wasted generation 5× while exposing reward-hacking failure modes that drive a follow-up bias-mitigation contribution. |
| **MC-1 MCMC-SelTDA** | Iterative self-training is empirically validated but theoretically opaque — we don't know if more rounds converge to anything or just diverge. We reframe iterative SelTDA as Metropolis-Hastings sampling over the student-parameter posterior, importing convergence diagnostics (R-hat, ESS) that yield the first principled stopping criterion for self-training. |
| **CS-X Coreset-Stratified-Filter** (extension of CS-1 via Janusian) | The quantity-vs-quality trade-off in pseudo-label filtering is a false dichotomy: most of "quantity" is redundant. We combine submodular coreset selection with per-stratum quantile filtering to maximize informational coverage per kept sample, resolving the trade-off in favor of more *diverse* signal at the same total budget. |
| **PV-1 Pivot-VQG** (extension via NEG-C2) | Joint `(Q,A)` generation entangles question creativity with answer correctness — a noisy answer poisons an otherwise good question. We decouple them: stage-1 model generates only the question from the image; stage-2 model (possibly stronger, possibly an MLLM judge) answers it. Filtering each stage independently improves both quality and diversity. |

---

## Top-12 Consolidated Candidate List (Divergent + Creative-Thinking)

For the upcoming Converge phase. Removed two weak candidates (PROV-1, MULTI-1 — out of thesis scope) and added the 5 new from creative-thinking.

| Rank-pending | ID | Family | Novelty | 1-GPU Feasibility | §1.4 OK? |
|---|---|---|---|---|---|
| Phase 1 (defensive) | CF-1 Cascade-Filter | Filtering | Low (incremental) | ✓ | ✓ |
| Phase 1 (defensive) | TS-1 Type-Stratified-Filter | Filtering | Med | ✓ | ✓ |
| Phase 2 (mid-novelty) | IT-1 Iterative-SelTDA | Iteration | Med-High | ✓ | ✓ |
| Phase 2 | CS-X Coreset-Stratified | Filtering | Med-High | ✓ | ✓ |
| Phase 3 (high-novelty) | RG-1 RAG-SelTDA-Medical | Teacher quality | High | ✓ | ✓ |
| Phase 3 | GROUND-1 SAM-Grounded-VQG | Teacher quality | High | △ (SAM heavy) | ✓ |
| Phase 3 | CT-1 / DIFF-1 Counterfactual | Robustness | High | △ (Diffusion heavy) | ✓ |
| Phase 4 (transformational) | **EB-1 Energy-Based-SelTDA** | Filtering theory | **Very high** | ✓ | ✓ (filter only) |
| Phase 4 | **RW-1 Reward-Shaped-VQG** | Teacher training | **Very high** | △ (RL fits 1 GPU at small scale) | ✓ (teacher only) |
| Phase 4 | **MC-1 MCMC-SelTDA** | Theory | **Very high** | ✓ (analysis, not training) | ✓ |
| Phase 4 | **PV-1 Pivot-VQG** | Generation | High | ✓ | ✓ |
| Phase 5 (long-tail) | JUDGE-1 GPT-Judge-Filter | Filtering | Med (API cost) | $ | ✓ |

---

## Reflection

The brainstorming skill (operational) surfaced 14 combinational/exploratory candidates. The creative-thinking skill (cognitive) pushed beyond into 5 transformational candidates — ideas that change the framing of the problem (energy-based, RL-shaped, MCMC, decoupled) rather than tune the parameters.

**Key insight from holding the tensions (F8)**: the filter-strictness vs. data-quantity trade-off is a *false* dichotomy. Once you decompose "quantity" into "informational coverage" (via coreset / submodularity), more diverse signal at the same budget is possible. This is the single most underexplored leverage point — and is technically simple enough to implement in Phase 2 of the thesis.

**Most important new bisociation**: Energy-Based SelTDA. The 3-gate cascade is *implicitly* energy-shaped — we just don't say so. Making it explicit unlocks decades of statistical-mechanics theory (calibration, phase transitions, RG) for principled use. This is a defensible thesis chapter on its own.

**Most important reformulation**: MCMC-SelTDA. Iterative self-training is *implicitly* MCMC — but no one has applied chain-convergence diagnostics to it. The result is iterative SelTDA papers that just "do N rounds" with no stopping criterion. R-hat across seeds would give the first principled answer to "how many rounds is enough?"

**Most important negation**: NEG(C5) — filtering as reward, not discard. Drops a hidden assumption baked into all post-hoc filtering papers.

These three (EB-1, MC-1, RW-1) form a coherent transformational triad: **theory (EB) + analysis (MC) + algorithm (RW)**. Together they could anchor an ambitious thesis. Defensible fallback: Phase 1 (CF-1 + TS-1 + CS-X) for a safer paper.
