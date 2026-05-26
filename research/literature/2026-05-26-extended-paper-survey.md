# Extended Paper Survey — SelTDA & Adjacent Fields (2020–2026)

- **Date**: 2026-05-26 (session 2)
- **Scope**: 45+ verified papers (arXiv IDs via Hugging Face index)
- **Purpose**: Expand literature base beyond Phase 2 synthesis; organize by mechanism for idea generation

> **Note**: xcons judge leakage đã được user sửa trên server — không còn là blocker.

---

## A. SelTDA baseline & direct successors

| ID | Paper | arXiv | Key takeaway for thesis |
|----|-------|-------|-------------------------|
| A1 | Khan et al. — SelTDA (CVPR 2023) | (PDF in repo) | 30% pseudo-QA noisy; peak synth:real=2:1; PathVQA zero-shot +1.67% |
| A2 | Khan et al. — DataEnvGym | 2410.06215 | Iterative student-feedback data generation; same author; VQA supported |
| A3 | Schwenk et al. — A-OKVQA benchmark | 2206.01718 | EK/VR questions need world knowledge — aligns with filter-by-type |

---

## B. Pseudo-label selection & SSL (mechanisms transferable to VQA filtering)

| ID | Paper | arXiv | Mechanism | SelTDA mapping |
|----|-------|-------|-----------|----------------|
| B1 | Cascante-Bonilla — Curriculum Labeling | 2001.06001 | Curriculum + **restart weights** each cycle | IT-1 anti confirmation-bias |
| B2 | Arazo et al. — Confirmation Bias in SSL | 1908.02983 | Naive pseudo-label overfits wrong labels | Motivation for multi-gate filter |
| B3 | Li et al. — SemiReward | 2310.03013 | Learned reward model for pseudo-label quality | Alternative to rule gates; mini-ablation |
| B4 | Zou & Caragea — JointMatch | 2310.14583 | **Classwise adaptive thresholds** + cross-labeling | Direct analog to TS-1 per question-type |
| B5 | Yang et al. — ShrinkMatch | 2308.06777 | Shrink class space to rescue uncertain samples | Recover quantity after strict filter |
| B6 | Du et al. — Future Self-Training (FST) | 2209.06993 | Teacher from **virtual future student** | Reduces confirmation bias in pseudo-labels |
| B7 | Ali et al. — Prototype-Guided (PICS+NALR) | 2507.22075 | Prototype + neighborhood consistency for CLIP pseudo-labels | Extend ITM gate with neighborhood graph |
| B8 | Ali et al. — DPA dual prototypes | 2408.08855 | Dual visual/text prototypes for pseudo-label ranking | Rank pseudo-QA by prototype alignment |
| B9 | Kahn et al. — Self-training ASR | 1909.09116 | **Ensemble + label filtering** for pseudo-label diversity | Teacher-council-lite at generation |

---

## C. Data diversity, coreset & instruction-data selection

| ID | Paper | arXiv | Mechanism | SelTDA mapping |
|----|-------|-------|-----------|----------------|
| C1 | Griffin et al. — ZCore | 2411.15349 | Zero-shot submodular coreset on FM embeddings | CS-X core |
| C2 | Zheng et al. — CCS coverage-centric | 2210.15809 | Coverage at high prune rates | High keep_top regime |
| C3 | Zhang et al. — TAGCOS | 2407.15235 | Gradient-clustered coreset | Expensive; use ZCore instead |
| C4 | Wang et al. — DPP diversity (2402.02318) | 2402.02318 | Determinantal point processes for diversity | Alternative to ZCore |
| C5 | Bukharin & Zhao — QDIT | 2311.14736 | Quality-diversity tradeoff explicit | TS-1 + CF-1 framing |
| C6 | Yang et al. — NovelSum diversity metric | 2502.17184 | Sample-level novelty correlates 0.97 with perf | Metric for filter report |
| C7 | Chen et al. — On diversity of synthetic data | 2410.15226 | Diversity metric for synthetic LLM data | Khan Fig.6 sunburst problem |
| C8 | Zhu et al. — BARE base+instruct refine | 2502.01697 | Base model diversity + instruct quality | Two-stage VQG: diverse Q, refined A |
| C9 | Yu et al. — Diversify and Conquer | 2409.11378 | k-means + iterative cluster resampling | Per-cluster filter thresholds |
| C10 | Chen et al. — MIG information gain | 2504.13835 | Maximize semantic information gain | Active pseudo-QA selection |
| C11 | Zhang et al. — D3 diversity/difficulty/dependability | 2503.11441 | Weighted coreset on 3 axes | Composite filter score redesign |

---

## D. Vision-language self-training & quality control

| ID | Paper | arXiv | Mechanism | SelTDA mapping |
|----|-------|-------|-----------|----------------|
| D1 | Deng et al. — STIC | 2405.19716 | Preferred/dispreferred via corruption | H3 multi-view filter |
| D2 | Luo et al. — ViLP | 2501.00569 | Good-bad image pairs; language prior probe | Filter drops text-only solvable QA |
| D3 | Sun et al. — SQ-LLaVA | 2403.11299 | Self-questioning for alignment | Teacher generates self-check questions |
| D4 | Hu et al. — Socratic Questioning (SQ) | 2501.02964 | Multi-round self-guide reasoning | Multi-turn pseudo-dialogue |
| D5 | Wang et al. — Q&A Prompts | 2401.10712 | Mine QA prompts as visual clues | Enrich teacher prompt with tags |
| D6 | Su et al. — SK-VQA | 2406.19593 | 2M synthetic knowledge-required QA | External knowledge synthetic scale |
| D7 | He et al. — Self-Correction Learning (SCL) | 2410.04055 | DPO on self-correction pairs | RW-1 variant |
| D8 | Kulkarni — VideoSAVi | 2412.00624 | Self-generate Q, score answers, DPO | Full self-alignment loop on video |
| D9 | Zhou et al. — VisionFoundry | 2604.09531 | Task-keyword synthetic VQA at scale | Target weak perception skills |

---

## E. Cycle consistency & cross-modal agreement

| ID | Paper | arXiv | Mechanism | SelTDA mapping |
|----|-------|-------|-----------|----------------|
| E1 | Bahng et al. — CycleReward | 2506.02095 | Image→text→image cycle as reward; 866K pairs | Replace/augment xcons with cycle score |
| E2 | Krestenitis — CycleCap | 2603.18282 | GRPO with cycle consistency reward | Stronger xcons narrative |
| E3 | Ray et al. — ConVQA entailed Q | 1909.04696 | Entailed question for consistency | Reverse Q-cycle gate |
| E4 | Chen et al. — Cross Pseudo Supervision | 2106.01226 | Two networks cross-supervise | Dual-student xcons agreement |
| E5 | Han et al. — UniCorn / UniCycle | 2601.03193 | Text→image→text cycle benchmark | Eval metric for pseudo-QA quality |
| E6 | Chung et al. — ACON cyclic consistency | 2505.24211 | Formal cyclic consistency criteria | Theoretical framing for xcons |

---

## F. Knowledge, RAG & external evidence (A-OKVQA EK / PathVQA)

| ID | Paper | arXiv | Mechanism | SelTDA mapping |
|----|-------|-------|-----------|----------------|
| F1 | Xia et al. — MMed-RAG | 2410.13085 | Domain RAG + adaptive context + DPO | RG-1 PathVQA teacher |
| F2 | Zhang et al. — PMC-VQA | 2305.10415 | 227k generative Med-VQA pairs | Scale reference for medical synth |
| F3 | Chen et al. — MISS medical SSL | 2401.05163 | Multi-task self-supervised Med-VQA | Medical pretrain before SelTDA |
| F4 | Liu et al. — GEMeX | 2411.16778 | Groundable explainable Med-VQA | Eval + grounding filter |
| F5 | Gai et al. — MedThink rationales | 2404.12372 | Decision-making rationales on PathVQA | R-Path benchmark for filter eval |
| F6 | Wang et al. — InViC cue tokens | 2603.16372 | Intent-aware visual cues; bottleneck | Reduce shortcut answers in pseudo-QA |
| F7 | Fu et al. — LiveVQA | 2504.05288 | Latest visual knowledge from web | Dynamic KB for EK questions |
| F8 | Shbita et al. — WikiVQABench | 2605.21479 | Wikipedia+Wikidata grounded VQA | KB construction template |
| F9 | Zhang et al. — Patho-AgenticRAG | 2508.02258 | Textbook page multimodal RAG + RL | PathVQA advanced RAG |

---

## G. Counterfactual, robustness & bias

| ID | Paper | arXiv | Mechanism | SelTDA mapping |
|----|-------|-------|-----------|----------------|
| G1 | Xu et al. — DeFacto | 2509.20912 | Counterfactual + random-mask + GRPO | CT-1 branch |
| G2 | Chen et al. — CounterVQA | 2511.19923 | Counterfactual VQA benchmark | Eval for CT-1 |
| G3 | van Sprang — REST cross-modal inconsistency | 2512.08923 | Same content, different modality answers | Detect language-prior pseudo-QA |
| G4 | Wang et al. — Evidence VQA (2020) | 2002.10215 | Reasoning eval vs coincidence | Filter metric design |

---

## H. OVD / detection — VLM pseudo-label purification (analogy)

| ID | Paper | arXiv | Mechanism | SelTDA mapping |
|----|-------|-------|-----------|----------------|
| H1 | Wang et al. — MarvelOVD | 2407.21465 | Detector + VLM co-guidance; online mining | Teacher+student co-filter |
| H2 | Jung et al. — Maieutic Prompting | 2205.11822 | Tree of explanations; SAT over noisy gens | Filter via explanation consistency |
| H3 | Liu et al. — Right this way (2411.00394) | 2411.00394 | VLMs indicate insufficient visual info | Drop pseudo-QA when image insufficient |

---

## I. Diversity of synthetic generation (teacher side)

| ID | Paper | arXiv | Mechanism | SelTDA mapping |
|----|-------|-------|-----------|----------------|
| I1 | Mostafazadeh et al. — VQG task (2016) | 1603.06059 | Commonsense/event-centric questions | Diversity target for teacher |
| I2 | Alampalle et al. — Weakly supervised VQG | 2306.06622 | Procedural Q from caption+visual | Cheap Q diversity without full VQG train |
| I3 | Wu et al. — Visual Haystacks / MIRAGE | 2407.13766 | Multi-image retrieval QA | Extend to multi-image unlabeled sets |

---

## Cross-theme synthesis (literature gaps)

1. **No paper combines SelTDA-style BLIP self-training with ZCore + type-stratified thresholds** — thesis Contribution 1 gap confirmed.
2. **CycleReward (E1) applied to pseudo-label filtering for VQA** — not found; novel if implemented.
3. **JointMatch classwise thresholds (B4) for VQA question types** — not found; TS-1 is novel application.
4. **ViLP good-bad pairs (D2) as filter without full STIC training** — underexplored for BLIP-scale models.
5. **BARE two-stage gen (C8) for VQG** — decouple diverse-Q from accurate-A; aligns with Pivot-VQG.
6. **Calibration/ECE on VQA pseudo-label scores** — still no direct prior in this survey.

---

## Reading priority (full text)

1. CycleReward 2506.02095 — closest to upgrading xcons
2. JointMatch 2310.14583 — TS-1 theoretical backing
3. ViLP 2501.00569 — language-prior filter
4. BARE 2502.01697 — two-stage teacher
5. SK-VQA 2406.19593 — knowledge-heavy synthetic at scale
6. MarvelOVD 2407.21465 — co-guidance filter pattern
