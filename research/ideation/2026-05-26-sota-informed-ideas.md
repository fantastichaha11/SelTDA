# Ideas Catalog Bổ Sung — Informed by SOTA Landscape (Firecrawl)

- **Date**: 2026-05-26
- **Input**: `research/literature/2026-05-26-sota-landscape-firecrawl.md`
- **Bổ sung cho**: `2026-05-26-extended-ideas-catalog.md` (IDEA-01 → IDEA-28)
- **Ngôn ngữ**: Tiếng Việt

---

## IDEA-29: Metric-Calibrated Reporting (MCR-1)

**Vấn đề**: PathVQA leaderboard top (VILA-M3 92.7%) dùng **Overall accuracy** gộp yes/no dễ + free-form khó. SelTDA báo **26.76% VQA soft acc** — so sánh trực tiếp sẽ bị reviewer bác.

**Cơ chế**: Chuẩn hóa báo cáo theo BESTMVQA: Closed (yes/no), Open (free-form), Overall, Recall — trên cùng checkpoint và cùng eval script (`pathvqa_eval.py`).

**Cách làm trong SelTDA**:
1. Post-process `vqa_result.json` → stratify theo answer type (regex / PathVQA metadata).
2. Mọi ablation filter báo **bảng 4 cột**, không chỉ một số.
3. Caption rõ: "BLIP-base data-scarce setting; not comparable to VILA-M3 92.7% overall."

**Dự đoán**: Không tăng accuracy — tăng **credibility** và tránh misinterpretation. Free-form có thể là metric nhạy hơn cho medical branch (+2–3 điểm khi filter hoạt động).

**Rủi ro**: Free-form sample count nhỏ → variance cao.

**Liên quan SOTA**: Wizwand PathVQA (MMQ free-form 13.4% vs yes/no 84%); BESTMVQA ECML PKDD 2024.

---

## IDEA-30: Knowledge-Consistency Fourth Gate (KC-1)

**Vấn đề**: Top A-OKVQA MC (HinD-CoT-Know 87.2%, Prophet 76.4%) đều **retrieve external knowledge**. SelTDA pseudo-QA trên EK questions có ~30% sai (Khan Tab.3) — nhiều khả năng do **factual hallucination**, không phải visual mismatch.

**Cơ chế**: Sau gate `itm`, thêm gate `kcons`:
1. Retrieve top-k passages (Wikipedia API / ConceptNet — giống Prophet nhẹ).
2. Prompt LLM nhỏ (hoặc NLI model): "Given passages P, is answer A supported for question Q?"
3. Score = entailment probability; quantile filter như `conf`.

**Cách làm**: Module mới `filtering/scorers.score_knowledge_consistency` + config gate `kcons` trong `filter_pseudo.yaml`. **Không** sửa train loop.

**Dự đoán**: EK stratum accuracy +4–8 điểm; tổng A-OKVQA val +1–2 nếu EK chiếm ~40% câu.

**Rủi ro**: Latency + API cost; retrieval noise. Ablation: kcons alone vs itm alone on EK subset.

**Papers/SOTA**: Prophet (OK-VQA 61.11%); QACap (A-OKVQA DA 73.4% val); HinD-CoT-Know (MC 87.2%).

---

## IDEA-31: SemIAug ⊕ SelTDA Composition (SS-1)

**Vấn đề**: SemIAug (MDPI 2024) cải thiện BLIP **+1.4% A-OKVQA** bằng cross-image question reassignment trên **labeled** data — không cần unlabeled. SelTDA thêm synthetic từ unlabeled. Hai trục **độc lập**.

**Cơ chế**:
```
real_aug = SemIAug(train.json)          # implicit augmentation
synthetic = filter(generate(unlabeled)) # SelTDA pipeline
train_files = [real_aug, synthetic]
```

**Cách làm**: Script preprocessing SemIAug (ngoài §1.4) tạo `train_semiaug.json`; giữ nguyên `train_vqa.py`.

**Dự đoán**: Additive gain ~+1.0 đến +2.5 trên 62.1% SelTDA filtered (nếu không saturation sớm).

**Rủi ro**: Double augmentation → overfit nếu không giữ synth:real cap.

**Papers**: SemIAug 10.3390/cmsf2024009003; Khan 2023 peak 2:1 ratio.

---

## IDEA-32: BLIP-Ceiling Framing + Teacher-Scale Ablation (TC-1)

**Vấn đề**: BLIP-2 đạt **80.2% A-OKVQA MC test** zero-shot fine-tune khác — trong khi SelTDA BLIP-base ~62% DA val. Reviewer sẽ hỏi: "Sao không dùng BLIP-2 làm teacher?"

**Cơ chế**: Ablation **chỉ ở generation** (§1.4 cho phép): so sánh pseudo-label quality khi teacher = BLIP-VQG (hiện tại) vs checkpoint lớn hơn (BLIP-2 decoder nếu compatible, hoặc caption model khác).

**Metric**: Không phải final student accuracy — mà **human-judge answer-correctness** trên 200 pseudo-QA (Khan Tab.3 protocol).

**Dự đoán**: Teacher lớn hơn → +10–15% pseudo-label precision → filter có ít việc hơn nhưng ceiling student cao hơn.

**Rủi ro**: BLIP-2 VQG pipeline khác architecture — integration cost.

**SOTA ref**: Wizwand BLIP-2 80.2% MC; SelTDA GitHub vẫn BLIP-only.

---

## IDEA-33: DataEnvGym Skill-Tree Wrapper (DE-1)

**Vấn đề**: DataEnvGym (cùng Zaid Khan) chứng minh iterative agent + skill feedback → +5.58 trên NaturalBench. SelTDA hiện **single-round**.

**Cơ chế**:
```
Round r:
  state_r = per-skill val errors on A-OKVQA (EK / VR / VI buckets)
  generate extra pseudo-QA targeting weak skills
  filter → synthetic_r.json
  train student_r → checkpoint_r
  if gain < ε: stop
```

**Cách làm**: Orchestration script (bash/python) gọi lại `generate_questions.py` + `filter_pseudo.py` + `train_vqa.py` — không sửa train internals. Skill buckets = A-OKVQA question categories.

**Dự đoán**: Round 2 +1–2 điểm val; round 3 diminishing returns (giống DataEnvGym Fig.4).

**Rủi ro**: Confirmation bias tích lũy — cần held-out judge và restart-from-pretrained mỗi 2 rounds (Cascante-Bonilla 2020).

**SOTA ref**: dataenvgym.github.io Skill-Tree +5.58; IT-1 trong catalog gốc.

---

## IDEA-34: Hallucination-Aware Medical Gate (HAL-1)

**Vấn đề**: PathVQA SOTA cao (92.7%) có thể **ẩn hallucination** — HALT-MedVQA benchmark chỉ ra model trả lời đúng format nhưng sai fact. SelTDA medical pseudo-labels dùng từ general-domain teacher → risk cao.

**Cơ chế** (inspired HALT-MedVQA):
1. **Image perturbation test**: crop/shuffle patch → nếu student answer không đổi trên câu visual-dependent → drop.
2. **Lexical gate**: answer phải intersect pathology lexicon (UMLS/SNOMED subset) với score > τ.
3. Optional: MMed-RAG retrieve → check answer in context.

**Cách làm**: `filtering/scorers.score_medical_plausibility` — chỉ bật khi `dataset=pathvqa`.

**Dự đoán**: Overall có thể giảm nhẹ; **free-form accuracy + recall** tăng 3–5 điểm (metric quan trng hơn clinically).

**Rủi ro**: Lexicon quá strict → drop hết sample.

**SOTA ref**: HALT-MedVQA arXiv 2401.05827; PathVQA free-form 13.4% (MMQ).

---

## IDEA-35: Yes/No vs Free-Form Dual-Track Curriculum (YF-1)

**Vấn đề**: Wizwand PathVQA — yes/no **84%** vs free-form **13.4%** (MMQ). SelTDA gain +1.67 overall masked failure mode trên open-ended medical answers.

**Cơ chế**: Tách synthetic pool:
- **Track A (YN)**: strict filter (q=0.9), train sớm (epoch 1–5)
- **Track B (FF)**: lenient filter + RAG teacher (MMed-RAG), train muộn (epoch 6–10) — curriculum

**Cách làm**: `filter_pseudo.py` output two JSONs; orchestration merge theo epoch schedule (duplicate-sampling weights).

**Dự đoán**: Free-form +5–8 điểm; yes/no không regression.

**Rủi ro**: §1.4 — epoch schedule có thể cần two-phase train runs thay vì sửa train_vqa.

**Liên quan**: IDEA TS-1 (type-stratified); J-1 staged pools; MCR-1 metrics.

---

## IDEA-36: Competitive Baseline Bundle (CB-1)

**Vấn đề**: Thesis dễ bị attack "so với SOTA 87%" nếu chỉ báo SelTDA 62%.

**Cơ chế**: Bắt buộc **bảng baseline đa tầng** trong mọi experiment table:

| Tier | Method | Expected A-OKVQA val DA |
|------|--------|-------------------------|
| T0 | BLIP finetune only | ~57% |
| T1 | SelTDA unfiltered | ~60% |
| T2 | SelTDA + cascade filter (ours) | target >62% |
| T3 | SemIAug + SelTDA (SS-1) | target >63% |
| T4 | Literature (QACap, not reproduced) | ~73% (reference only) |

**Cách làm**: Documentation + eval protocol — không code mới.

**Dự đoán**: Làm rõ contribution boundary → reviewer acceptance.

---

## IDEA-37: NaturalBench Transfer Probe (NB-1)

**Vấn đề**: DataEnvGym dùng NaturalBench cho multimodal VQA iterative teaching — benchmark đo **robustness** (không shortcut). SelTDA filter có thể overfit A-OKVQA distribution.

**Cơ chế**: Sau mỗi student checkpoint, eval zero-shot trên NaturalBench subset (nếu có adapter) — metric phụ **generalization**.

**Dự đoán**: Filtered student → higher NaturalBench than unfiltered at matched train size (hypothesis: less noise → less shortcut).

**Rủi ro**: BLIP-base có thể không có NaturalBench numbers public — cần custom eval script.

**SOTA ref**: DataEnvGym NaturalBench table (+3.72 to +5.58).

---

## IDEA-38: Saturation-Aware Synth Ratio Search (SSR-1)

**Vấn đề**: Khan peak **2:1** synthetic:real; với filter chất lượng cao hơn, optimal ratio có thể **4:1 hoặc 8:1** (SAT-1 trong catalog). SOTA models dùng **hàng triệu** synthetic instruction pairs — khác scale.

**Cơ chế**: Grid search `synth:real ∈ {1:1, 2:1, 4:1, 8:1}` × `{unfiltered, filtered}` — plot accuracy vs N synthetic (saturation curve).

**Dự đoán**: Filtered curve plateau muộn hơn → optimal ratio shift right.

**Cách làm**: `--overrides truncate_train_dataset_to=...` + varying synthetic JSON size.

**Liên quan**: Khan Tab.4; ZCore CS-X (quality over quantity).

---

## Ma trận ưu tiên (bổ sung)

| ID | Tên | Effort | Impact | Novelty | §1.4 OK |
|----|-----|--------|--------|---------|---------|
| MCR-1 | Metric reporting | Thấp | Credibility | Thấp | ✓ |
| KC-1 | Knowledge gate | Trung bình | Cao (EK) | Cao | ✓ |
| SS-1 | SemIAug compose | Thấp | Trung bình | Trung bình | ✓ |
| DE-1 | Skill-Tree loop | Cao | Cao | Cao | ✓ |
| HAL-1 | Medical hallucination gate | Trung bình | Cao (PathVQA FF) | Cao | ✓ |
| YF-1 | Dual-track curriculum | Trung bình | Cao (PathVQA) | Trung bình | ✓ (orchestration) |
| CB-1 | Baseline bundle | Thấp | Review | — | ✓ |
| SSR-1 | Saturation curves | Trung bình | Medium | Medium | ✓ |

**Đề xuất chạy trước GPU**: MCR-1 + CB-1 (free) → H1 cascade → SSR-1 → SS-1 → KC-1 ablation on EK subset.
