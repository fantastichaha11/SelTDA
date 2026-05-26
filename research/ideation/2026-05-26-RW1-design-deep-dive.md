# RW-1 Design Deep-Dive — Reward-Shaped VQG with Hacking Mitigation

- **Date**: 2026-05-26 ~03:15 local
- **Focus**: Transform RW-1 (DPO of teacher VQG with filter-as-reward) from a sketch into a defendable design, given the published reward-hacking evidence (Rafailov 2024 arXiv 2406.02900) and adjacent DPO literature.
- **Skills applied**: `brainstorming-research-ideas` Frameworks 3/4/6/8; `creative-thinking-for-research` Frameworks 4/5/8.

---

## The Core Tension (F3 + F8 = Janusian)

| Pole | What pulls this direction |
|---|---|
| **Pole A — Maximize reward** | Filter score is the only training signal; teacher should produce QA that passes filter perfectly. |
| **Pole B — Stay faithful** | If teacher overfits filter, it learns to game spurious filter features (high CLIP-ITM but visually wrong, etc.) — reward hacking. |
| **Naive resolution** | KL constraint to baseline VQG_IC (standard DPO move). Limits drift. |
| **Janusian synthesis** | Filter as reward AND filter as discriminator of hacking. Two filters: one drives policy, one held out as audit. *The held-out judge becomes a measurement instrument, not a regularizer.* |

This synthesis makes the failure mode (hacking) **measurable** — exactly what Rafailov 2024 says is missing in current DPO literature.

---

## F4 (Brainstorm) — Cross-Pollination for Reward Hacking Mitigation

Other fields that have grappled with "model gets too good at its proxy metric":

| Source field | Mitigation pattern | RW-1 import |
|---|---|---|
| **Classical RL** | Reward shaping with potential-based bonuses (Ng et al. 1999) — bonuses must be irrotational. | Decompose filter reward into invariant components (visual grounding) vs. surface features (CLIP-ITM). Use only invariant for reward. |
| **Goodhart-aware metrics literature** | Multi-metric aggregation; track ratio of "easy" to "hard" reward acquisition. | Track per-filter-gate reward acquisition rates. If teacher's reward gain comes entirely from `s_itm` (cheap to game), flag it. |
| **Adversarial ML** | Adversarial training: explicitly generate worst-case examples that fool the reward. | Adversarial-VQG: teacher trained on `s_filter - α · s_held_out_judge` instead of `s_filter` alone. |
| **Mechanism design / RLHF** | Reference-policy KL budget (PPO/DPO). | Standard DPO mitigation. Report KL trajectory. |
| **Statistics (proper scoring rules)** | Use only proper scoring rules so optimization can't be gamed without true accuracy. | Replace `s_conf` (sequence log-prob, not proper for label) with a Brier or log score over discrete answer set. |
| **Behavioral econ (commitment devices)** | Pre-commit to evaluation set before training; cannot adjust ex-post. | Lock the 200-sample held-out human-judge set BEFORE DPO starts. No look. |
| **Cryptographic auditing** | Commitment + reveal protocol. | Hash-commit teacher checkpoint at start; release-with-hash at end. Auditor verifies. |
| **Multi-agent (mediators)** | Third-party arbiter unaffected by either party's incentive. | GPT-4V as the held-out judge — not used as reward, only as monitor. |

**Most actionable**: Decompose filter reward into per-gate components and track per-gate reward acquisition. If RW-1's gains come from gaming `s_itm` (cheap, gameable) rather than `s_xcons` (expensive, hard to game), the failure mode is identified *in vivo*, not post-hoc.

---

## F6 (Brainstorm) — Boundary Probes for RW-1

Where might RW-1 break? Each probe → potential paper finding.

| Probe | Hypothesis | Outcome if confirmed |
|---|---|---|
| **Filter-gate ablation under DPO** | Train RW-1 with each filter gate as sole reward in turn. Measure: which gates can be gamed quickest? | Identifies hackable gates → recommends which gates to swap out for harder ones. |
| **KL budget sweep** | Train RW-1 with KL coefficient β ∈ {0.01, 0.1, 1.0, 10.0}. Measure: at what β does accuracy peak vs. reward peak diverge? | Locates Goodhart point. Defines safe operating regime. |
| **Held-out judge stability** | Use 3 different held-out judges (BLIP-ITM, CLIP-Large, GPT-4V). Do they agree on whether teacher is hacking? | Convergent judgment → trustworthy. Divergent → instructive failure of judge choice. |
| **Long-run drift** | Train RW-1 for 1, 3, 10, 30 epochs. Does pseudo-QA quality degrade after a point even as filter score rises? | Direct empirical confirmation of Rafailov 2024 in VQA domain. |
| **Counter-example generation under hacking** | When teacher hacks filter, what specific QA patterns emerge? Manually inspect. | Qualitative taxonomy of hacking modes. |
| **Reward-stop early termination** | Stop DPO when held-out judge score peaks. Measure: does early-stop RW-1 beat unbounded RW-1? | If yes → "minimal DPO" is the prescription. |

**Boundary insight**: most published DPO papers report final model only. RW-1's contribution can be the **trajectory-level** analysis: how filter-score, held-out-judge-score, and downstream-student-accuracy *diverge over training*. No prior VQA self-training paper has this view.

---

## F5 (Creative Thinking) — Negate RW-1's Own Assumptions

| Assumption | Negation | What emerges |
|---|---|---|
| The reward is the filter score | Reward is *delta filter score from teacher rev1 to rev2* — i.e., gradient of improvement | **Gradient-DPO**: teacher rewarded for producing QA that improves student more, not just passes filter. |
| DPO updates the teacher | DPO updates the **decoding strategy** (top-p, temperature, length penalty) only | **Decoder-DPO**: cheap, no weight update, just sampling-policy tuning. Could be a footnote experiment. |
| Filter is the judge | Filter is the *defender*; teacher is *attacker*; they alternate | **Adversarial co-training**: filter and teacher train against each other → arms race → bias toward harder QA. |
| Reward is scalar | Reward is per-token (sequence-level RL) | **Token-DPO**: penalize specific bad tokens (hallucinated colors, wrong counts) rather than whole-sample. Higher resolution. |
| RW-1 trains one teacher | RW-1 trains an ensemble of N teachers each with different filter weights | **Population-RW**: diversity emerges from population, not from sampling. |
| Reward signal is from filter only | Reward includes student loss-reduction signal | **Student-feedback-RW**: closes the loop directly (DataEnvGym territory). |
| Reward optimized at training | Reward optimized at inference (Best-of-N from frozen teacher) | **BoN-only RW**: skip training entirely; just sample N=8 and keep best. Surprisingly effective baseline. |

**Strongest negation**: **BoN-only RW** as a baseline. If sampling N=8 from frozen VQG_IC + keeping the highest-filter-score one matches DPO-trained RW-1, then DPO is unnecessary. This is the Simplicity Test (F7) applied to RW-1. **Must include this baseline.**

---

## F4 (Creative Thinking) — Hidden Constraints in RW-1

| Hidden constraint | Negotiable? | If dropped: |
|---|---|---|
| Filter score is *static* throughout DPO training | Yes | Filter could be updated mid-DPO — but that gives the teacher a moving target → instability. Risk. |
| Reward dataset is *fixed* preference pairs | Yes | Iterative DPO: regenerate preferences after each epoch → tracks distribution shift. |
| Only the teacher's QA-generation is updated | Yes | Could also update teacher's *image encoder* via DPO → grounding improves. But violates §1.4? VQG_IC is the teacher = generate_questions.py uses BLIP_Decoder; image encoder weights are inside. **Confirm whether updating these requires train_vqg.py changes.** |
| Same KL coefficient throughout training | Yes | Anneal KL: start tight, loosen → mimics warmup + exploration. |
| Reward is scalar combination of gates | Yes | Per-gate Pareto front: track teacher progress in 3D reward space; identify gates that lag. |

---

## F8 (Janusian Final Synthesis)

The strongest Janusian framing for RW-1:

> **Filter is simultaneously a reward (for the teacher) and an adversary (for the auditor).** The contribution is not eliminating reward hacking — it's *making hacking measurable, characterizable, and controllable* via held-out auditing.

This reframes RW-1 from "we propose RL for SelTDA" to "we propose a *measurement framework* for what RL does to SelTDA, including its failure modes". The latter is more defensible because the paper still has a contribution even if the headline accuracy gain is modest — the *characterization* is the contribution.

---

## RW-1 Final Design (after deepening)

```
Algorithm: Reward-Shaped VQG (RW-1)
  Input: VQG_IC teacher (frozen pretrained), 51k unlabeled images, 3-gate filter,
         held-out judge (GPT-4V or independent BLIP-ITM-Large)

  Step 1 (preference dataset construction, West-of-N style):
    for each unlabeled image I:
      sample N=8 candidate (Q,A) from VQG_IC at top-p=0.92
      score each with 3-gate filter → s_total
      chosen = argmax s_total, rejected = argmin s_total
    output: 51k DPO triples (I, chosen, rejected)

  Step 2 (DPO of teacher):
    train teacher with DPO loss + KL penalty β=0.1 to frozen VQG_IC ref
    log every epoch:
      - per-gate filter score (conf, itm, xcons) acquisition rate
      - held-out judge score
      - KL(teacher || VQG_IC)
      - student-acc-after-1-epoch-finetune on a 1k subsample
    early-stop when held-out-judge score plateaus or decreases
    early-stop also when KL exceeds budget (β-tuned)

  Step 3 (downstream A-OKVQA evaluation):
    generate fresh 51k pseudo from DPO-trained teacher
    apply same 3-gate filter at keep_top=0.75
    train student per spec on real + filtered synth
    measure A-OKVQA val accuracy

  Headline outcomes (any one is a publishable contribution):
    (a) A-OKVQA val ≥ +1.0 over CF-1 → "DPO compresses filtering into teacher" win
    (b) Held-out judge diverges from filter score during DPO → "characterized reward hacking in VQA self-training" win
    (c) Per-gate acquisition uneven → "which filter gates are gameable" diagnostic win

Required baselines (do NOT skip):
  - BoN-only (no DPO, sample N=8, keep best, generate from frozen teacher) → tests if DPO is needed
  - Filtered-without-DPO (CF-1) → tests if DPO adds anything
  - Random-subsample-matched-size → tests if filter is doing real work
  - DPO without KL penalty → drift baseline (failure case for science)
```

Compute estimate: Step 1 = 8× generation = ~24h A5000. Step 2 DPO = 8-12h. Step 3 train+eval = 12h. **Total ~48h ≈ 2 days A5000 ≈ $24.** Cheap enough for an ablation chapter.

---

## Updated top candidate ranking (after Cycle 2)

Same top-12 list as `2026-05-26-EB-RW-CS-deep-dive.md` but with RW-1's risk reduced from HIGH to MEDIUM thanks to the Janusian reframe (measurement framework, not just method) and explicit hacking-mitigation design.

**New view**: RW-1 is *not* a single-experiment paper; it's a **measurement instrument**. This makes it a strong candidate for thesis Chapter 4 (after Phase 1 filtering, Phase 2 iterative, before paper-writing).
