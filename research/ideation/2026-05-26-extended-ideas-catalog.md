# Extended Ideas Catalog — Chi tiết 24 hướng cải tiến SelTDA

- **Date**: 2026-05-26
- **Ngôn ngữ**: Tiếng Việt (thuật ngữ kỹ thuật giữ tiếng Anh khi cần)
- **Ràng buộc**: Tuân thủ §1.4 — chỉ sửa generation, filtering, orchestration (user đã fix xcons trên server)
- **Literature**: `research/literature/2026-05-26-extended-paper-survey.md`

Mỗi ý tưởng gồm: **Vấn đề → Cơ chế → Cách làm trong SelTDA → Dự đoán → Rủi ro → Paper liên quan**

---

## Nhóm I — Lọc pseudo-label (Filtering)

### IDEA-01: Cascade Filter nâng cao (CF-1+) — Baseline đã có spec

**Vấn đề**: Khan 2023 Tab.3 — ~30% pseudo-QA sai; Tab.4 — thêm synthetic quá 2:1 làm accuracy giảm.

**Cơ chế**: Ba tín hiệu độc lập:
- `conf`: teacher tin vào chuỗi decode (log-prob)
- `itm`: ảnh khớp text "Q? A." (CLIP cosine)
- `xcons`: student zero-shot trả lại A từ (I,Q)

Cascade loại dần — sample phải vượt cả ba ngưỡng quantile.

**Cách làm**: `filter_pseudo.py` + `filtering/gates.py` (đã implement). Ablation 8 variant trong H1 protocol.

**Dự đoán**: +1.5 đến +3 điểm A-OKVQA val so với 60.01% unfiltered.

**Rủi ro**: Nếu không beat random-subsample cùng size N → filter vô nghĩa.

**Papers**: Khan 2023; SemiReward 2310.03013; Arazo 1908.02983.

---

### IDEA-02: Type-Stratified Filter (TS-1)

**Vấn đề**: Một ngưỡng global `keep_top=0.75` bất công — câu yes/no cần strict, câu open-ended EK cần lenient (vì SBERT match khó).

**Cơ chế**: Lấy cảm hứng **JointMatch** (2310.14583) — *classwise adaptive thresholds*. Phân loại pseudo-QA theo question type (how-many / yes-no / color / EK / VR), tính quantile **riêng từng stratum**.

**Cách làm**:
1. Gán type bằng rule-based parser trên question (hoặc A-OKVQA metadata nếu có).
2. Trong `filter_pseudo.py`, group scores → `thresholds_from_quantile` per group.
3. Giữ **tổng số sample** cố định bằng cách cap mỗi stratum theo tỷ lệ trong val set.

**Dự đoán**: EK/VR accuracy +3–5 điểm mà không regression tổng thể.

**Rủi ro**: Type classifier sai → ngưỡng lệch. Cần ablation "oracle types" vs heuristic.

**Papers**: JointMatch 2310.14583; Khan 2023 Fig.6 sunburst.

---

### IDEA-03: Coreset-Stratified Filter (CS-X)

**Vấn đề**: Sau filter confidence, nhiều sample **gần trùng** (cùng ảnh, câu hỏi paraphrase) — lãng phí budget synthetic.

**Cơ chế**: **ZCore** (2411.15349) chọn subset đại diện trên embedding CLIP/DINOv2 của (I, "Q? A."), tối đa hóa coverage, giảm redundancy. Kết hợp TS-1: coreset **trong từng question type**.

**Cách làm**:
```
score_quality = f(conf, itm, xcons)
candidates = top 90% by quality per stratum
selected = ZCore(candidates, budget=N_stratum)
```
Port code từ github.com/voxel51/zcore.

**Dự đoán**: Ở cùng N kept samples, CS-X > filter-only ≥1 điểm val.

**Rủi ro**: Embedding không capture semantic answer equivalence.

**Papers**: ZCore 2411.15349; CCS 2210.15809; NovelSum 2502.17184.

---

### IDEA-04: Random-Subsample Control (bắt buộc)

**Vấn đề**: Mọi claim "filter giúp" đều có thể do **chỉ cần ít data sạch hơn**, không phải do gate thông minh.

**Cơ chế**: Chọn ngẫu nhiên N sample từ raw pool (N = |filtered set|). Nếu random ≈ filter → filter fail.

**Cách làm**: Script nhỏ post-filter; 3 seeds; cùng `train_vqa.py` config.

**Dự đoán**: Filter beat random ≥0.5 điểm nếu gates có signal thật.

**Papers**: Best practice từ brainstorm F7; LLM Unlearning coreset effect 2504.10185 (cảnh báo subset nhỏ vẫn mạnh).

---

### IDEA-05: ViLP Language-Prior Gate (LP-1)

**Vấn đề**: ViLP (2501.00569) chứng minh VLM trả lời đúng **chỉ từ text prior** trên một số câu — pseudo-QA đó không dạy visual grounding.

**Cơ chế**: Với mỗi (I,Q,A): tạo **text-only baseline** — crop ảnh thành noise hoặc dùng caption-only prompt. Nếu student vẫn trả A → **drop** (language shortcut).

**Cách làm**: Thêm gate thứ 4 `lp`: score = 1 nếu answer thay đổi khi ảnh bị corrupt mạnh; 0 nếu không. Không cần train code mới — chỉ thêm forward student.

**Dự đoán**: AdVQA / VQA-CE robustness +2–4 điểm; A-OKVQA val +0.5–1.

**Rủi ro**: 3× student forward; có thể drop quá nhiều counting questions.

**Papers**: ViLP 2501.00569; STIC 2405.19716; Evidence VQA 2002.10215.

---

### IDEA-06: CycleReward Gate (CR-1) — nâng cấp xcons

**Vấn đề**: xcons chỉ check A' ≈ A. Không verify **image-text alignment đầy đủ**.

**Cơ chế**: **CycleReward** (2506.02095): I → (Q,A) → reconstruct I' qua text-to-image (hoặc caption→CLIP image embedding). Score = similarity(I, I').

**Cách làm**: Lightweight: dùng CLIP text encoder + image encoder — không cần full T2I. Hoặc: teacher sinh caption từ (Q,A), so CLIP(I, caption).

**Dự đoán**: Filter precision +10% trên human judge 100 mẫu.

**Rủi ro**: CLIP cycle yếu trên fine-grained attributes; compute cost.

**Papers**: CycleReward 2506.02095; CycleCap 2603.18282; ConVQA 1909.04696.

---

### IDEA-07: Reverse Question Cycle (RQ-1)

**Vấn đề**: xcons một chiều — không bắt được **Q sai nghĩa nhưng A tình cờ đúng**.

**Cơ chế**: ConVQA (1909.04696) — sinh Q' từ (I, A), so SBERT(Q, Q').

**Cách làm**: Template teacher: `"Question: <?>. Answer: {A}."` — decode Q'. Gate `rq` = sbert_match(Q, Q').

**Dự đoán**: Đặc biệt giúp VR/EK — nơi Q dài và paraphrase nhiều.

**Rủi ro**: +30% teacher forward at generation/filter time.

**Papers**: ConVQA 1909.04696; Cycle consistency literature (E1–E6).

---

### IDEA-08: ShrinkMatch-style Rescue (SM-1)

**Vấn đề**: Strict filter vứt nhiều sample **gần ngưỡng** — mất diversity.

**Cơ chế**: ShrinkMatch (2308.06777): với sample uncertain, **thu hẹp không gian câu trả lời** (top-k answers của student) rồi re-score confidence.

**Cách làm**: Pseudo-QA có xcons ∈ [τ-0.1, τ]: lấy top-5 student answers; nếu A nằm trong top-5 và margin đủ → **rescue** thay vì drop.

**Dự đoán**: Giữ thêm 15% pool với precision không giảm >5%.

**Papers**: ShrinkMatch 2308.06777; FlexMatch family.

---

### IDEA-09: Prototype-Guided Filter (PG-1)

**Vấn đề**: Rule gates không học được structure của **correct vs incorrect** pseudo-QA trong embedding space.

**Cơ chế**: PICS/NALR (2507.22075): xây prototype per question type từ labeled train; pseudo-QA gần prototype "correct cluster" → keep.

**Cách làm**:
1. Embed train (I,Q,A) bằng BLIP hoặc CLIP.
2. K-means per type trên labeled data.
3. Score pseudo = distance to nearest "positive" prototype − distance to "negative".

**Dự đoán**: Beat 3-gate rule-based trên filter P/R nếu có đủ labeled signal.

**Rủi ro**: Overfit prototype trên 17k A-OKVQA train.

**Papers**: PICS 2507.22075; DPA 2408.08855.

---

### IDEA-10: SemiReward Mini-Ablation (SR-1)

**Vấn đề**: Rule gates có thể suboptimal — learned reward có thể tốt hơn.

**Cơ chế**: SemiReward (2310.03013) — MLP nhỏ predict quality từ (conf, itm, xcons, type one-hot).

**Cách làm**: Label 500 pseudo-QA bằng human judge → train reward → rank thay vì cascade. **Appendix only** — so sánh với CF-1.

**Dự đoán**: +0–1 điểm so rule gates; mainly for paper completeness.

**Papers**: SemiReward 2310.03013.

---

## Nhóm II — Sinh dữ liệu (Generation)

### IDEA-11: BARE Two-Stage VQG (PV-1 / BARE-VQG)

**Vấn đề**: Teacher sinh (Q,A) jointly — lỗi Q và lỗi A entangle; Khan Tab.3 không tách được.

**Cơ chế**: **BARE** (2502.01697): base model sinh **đa dạng** Q; instruct model sinh A từ (I,Q) — tách diversity và quality.

**Cách làm**:
1. Stage 1: nucleus sample Q từ image-only prompt (BLIP decoder).
2. Stage 2: conditional `"Question: {Q} Answer:"` → A.
3. Filter Q và A **độc lập** — drop nếu Q ITM thấp hoặc A xcons thấp.

**Dự đoán**: Answer-correctness manual eval +5–10% absolute.

**Rủi ro**: 2× decode latency.

**Papers**: BARE 2502.01697; Pivot-VQG từ creative session 1.

---

### IDEA-12: Q&A Prompts Visual Scaffolding (QP-1)

**Vấn đề**: Teacher thiếu "visual clues" — đặc biệt A-OKVQA EK.

**Cơ chế**: Q&A Prompts (2401.10712): image tagging → sinh nhiều (Q,A) mini từ tags → prompt vào MLLM.

**Cách làm trên BLIP**:
1. Florence-2 / RAM tags trên unlabeled image (offline).
2. Prepend tags vào teacher prompt: `"Tags: dog, frisbee, park. Question:"`.
3. Không đổi student.

**Dự đoán**: EK accuracy +3–5 trên A-OKVQA.

**Rủi ro**: Tag noise; dependency thêm model.

**Papers**: Q&A Prompts 2401.10712; VisionFoundry 2604.09531.

---

### IDEA-13: Type-Conditioned Generation (TC-1 / H7)

**Vấn đề**: Sunburst Khan Fig.6 — quá nhiều "how many", thiếu EK/VR.

**Cơ chế**: Prompt teacher với type token: `"[TYPE=external_knowledge] Question:"`. Oversample underrepresented types until distribution match val.

**Cách làm**: `generate_questions.py` — thêm `--question_type` schedule; 2 questions/image với type rotation.

**Dự đoán**: Per-type +5 điểm EK/VR; tổng +1–2.

**Papers**: QDIT 2311.14736; diversity synthetic 2410.15226.

---

### IDEA-14: SK-VQA-style Knowledge Context (KC-1)

**Vấn đề**: EK questions cần fact ngoài ảnh — teacher BLIP không có.

**Cơ chế**: SK-VQA (2406.19593) — synthetic QA **bắt buộc** external knowledge; retrieve Wikipedia snippet theo image tags.

**Cách làm**: Giống RG-1 nhưng cho A-OKVQA: retrieve 1–2 câu Wiki → prepend prompt teacher. Filter thêm gate: retrieved text phải ITM-match A.

**Dự đoán**: A-OKVQA EK slice +5–8 điểm.

**Papers**: SK-VQA 2406.19593; WikiVQABench 2605.21479; LiveVQA 2504.05288.

---

### IDEA-15: Teacher Council (TC-2)

**Vấn đề**: Một teacher một seed — variance cao.

**Cơ chế**: Kahn ASR (1909.09116) — ensemble pseudo-labels; giữ khi **2/3 teachers agree** on A (SBERT).

**Cách làm**: Train 3 VQG_IC khác seed; generate 3×; merge by agreement. Cost 3× generation — one-time.

**Dự đoán**: Precision +8–12% human judge; recall −10% (tradeoff).

**Papers**: Self-training ASR 1909.09116; MarvelOVD co-guidance 2407.21465.

---

### IDEA-16: SQ-LLaVA Self-Check Questions (SQ-1)

**Vấn đề**: Pseudo-QA không có cơ chế **tự kiểm tra** trước khi vào pool.

**Cơ chế**: SQ-LLaVA (2403.11299): model sinh câu hỏi phụ để verify hiểu ảnh.

**Cách làm**: Sau khi sinh (Q,A), teacher sinh Q2: "What in the image supports that {A}?" — nếu Q2 không ITM-match image regions → drop.

**Dự đoán**: Giảm hallucination rate 10–15% trên manual eval.

**Papers**: SQ-LLaVA 2403.11299; Socratic Questioning 2501.02964.

---

## Nhóm III — Vòng lặp & curriculum (Iteration)

### IDEA-17: Iterative SelTDA / DataEnvGym-style (IT-1)

**Vấn đề**: SelTDA 1 vòng — student yếu ở type X nhưng teacher không biết.

**Cơ chế**: DataEnvGym (2410.06215): sau mỗi vòng, báo **skill-gap** (error rate per type) → re-fine-tune VQG trên weak types → regenerate → filter → student v2.

**Cách làm**:
- Round 1: standard SelTDA + filter.
- Eval val per type → top-3 weak types.
- Round 2: `train_vqg.py` chỉ trên subset train có types đó (+ curriculum easy→hard).
- **Restart** student weights mỗi round (Cascante-Bonilla 2001.06001).

**Dự đoán**: Round 3 ≥ Round 1 + 2.0 val (H6).

**Rủi ro**: 3× training cost; confirmation bias nếu không restart.

**Papers**: DataEnvGym 2410.06215; Curriculum Labeling 2001.06001; FST 2209.06993.

---

### IDEA-18: Staged Pool Curriculum (J-1)

**Vấn đề**: Tension strict filter ↓ quantity vs curriculum cần easy→hard.

**Cơ chế**: Hai file synthetic:
- `synthetic_easy.json`: keep_top=0.9 (lenient)
- `synthetic_hard.json`: full C+I+X @ 0.75

**Cách làm**: `train_files=[train, synthetic_easy, synthetic_hard]` với duplicate-sampling: easy ×1, hard ×3 (proxy weight). §1.4 safe.

**Dự đoán**: Convergence nhanh hơn; final acc +0.5–1 vs single pool.

**Papers**: Curriculum Labeling; J-1 Janusian synthesis.

---

### IDEA-19: Future Self-Training Filter (FST-1)

**Vấn đề**: Pseudo-label từ **student hiện tại** — confirmation bias.

**Cơ chế**: FST (2209.06993): virtual gradient step → teacher tương lai → pseudo-label chất lượng hơn.

**Cách làm** (approximation): Train student 1 epoch on current pseudo → dùng **checkpoint đó** làm xcons judge cho round tiếp (khác pretrained-only — user đã kiểm soát leakage bằng fresh A-OKVQA-only student).

**Dự đoán**: Iterative gain +0.5 so IT-1 không FST.

**Papers**: FST 2209.06993; Arazo 1908.02983.

---

### IDEA-20: Active Image Selection (AL-1)

**Vấn đề**: SelTDA generate đều trên 17k unlabeled — không ưu tiên ảnh "student cần".

**Cơ chế**: BADGE/MIG-style: chọn ảnh có **gradient diversity cao** hoặc entropy student cao trên val.

**Cách làm**:
1. Student v0 trên real only.
2. Score mỗi unlabeled image = mean entropy câu hỏi template.
3. Chỉ generate từ top 50% images.

**Dự đoán**: Cùng số pseudo-QA, accuracy +1–2 vs uniform.

**Rủi ro**: TAGCOS cần backward — dùng entropy-only proxy.

**Papers**: MIG 2504.13835; DiverseEvol 2311.08182.

---

## Nhóm IV — Medical / PathVQA

### IDEA-21: PathVQA Direct SelTDA (H4)

**Vấn đề**: Khan chỉ zero-shot PathVQA (+1.67%) — không công bằng.

**Cơ chế**: Train teacher VQG trên PathVQA train → generate từ unlabeled pathology → filter → train student.

**Dự đoán**: Test ≥ 35–40% (vs 26.76% published).

**Papers**: Khan 2023; PMC-VQA 2305.10415; MISS 2401.05163.

---

### IDEA-22: MMed-RAG Teacher (RG-1 / H5)

**Vấn đề**: Limitation #2 — vocabulary chuyên ngành.

**Cơ chế**: MMed-RAG (2410.13085): retrieve domain-aware snippets → adaptive selection → generate.

**Cách làm**: KB nhỏ (PubMed abstracts + pathology glossary) → top-k retrieve by image embedding → prepend teacher prompt. Gate R: drop nếu retrieval confidence thấp (J-4).

**Dự đoán**: Pseudo answer-correctness +10–15% human judge; test +4 vs filtered-only.

**Papers**: MMed-RAG 2410.13085; Patho-AgenticRAG 2508.02258; GEMeX 2411.16778.

---

### IDEA-23: MedThink Rationale Filter (MT-1)

**Vấn đề**: PathVQA cần **grounded** answers — filter semantic không đủ.

**Cơ chế**: MedThink (2404.12372): sinh rationale trước A — filter drop nếu rationale không mention visual findings trong tags.

**Cách làm**: Teacher prompt: `"Rationale: ... Answer: ..."` — parse rationale; check keyword overlap với CheXpert tags.

**Dự đoán**: Giảm false positive trên closed-ended PathVQA.

**Papers**: MedThink 2404.12372; InViC 2603.16372.

---

## Nhóm V — Robustness & học teacher

### IDEA-24: Counterfactual Filter (CT-1)

**Vấn đề**: Limitation #3 — bias amplification; AdVQA chỉ +6.37.

**Cơ chế**: DeFacto (2509.20912): sinh I' (remove object / swap attribute). Drop nếu student vẫn trả A trên I'.

**Cách làm**: Lightweight inpainting (LaMa) hoặc simple crop-remove — không cần full GRPO.

**Dự đoán**: AdVQA +3–5; VQA-CE +5–10 nếu augment cả positive/negative pairs.

**Papers**: DeFacto 2509.20912; CounterVQA 2511.19923; ViLP 2501.00569.

---

### IDEA-25: Maieutic Consistency Filter (MA-1)

**Vấn đề**: Một pseudo-QA có thể **self-contradictory** khi suy luận sâu.

**Cơ chế**: Maieutic Prompting (2205.11822): sinh cây giải thích abductive → SAT check consistency.

**Cách làm** (lightweight): GPT-4 mini sinh 3 supporting facts cho (Q,A) — drop nếu facts mâu thuẫn nhau (NLI model). **Mini-ablation 200 sample** — không scale full pool.

**Papers**: Maieutic 2205.11822; Khan 2023 Section 5 future work.

---

### IDEA-26: Reward-Shaped VQG / DPO (RW-1)

**Vấn đề**: Filter chỉ **loại** — không **dạy** teacher sinh tốt hơn.

**Cơ chế**: West-of-N + Filtered DPO: mỗi ảnh sinh N pairs → filter score → best=chosen, worst=rejected → DPO teacher.

**Mitigation**: KL budget + held-out judge (Rafailov 2406.02900 reward hacking).

**Dự đoán**: Generation pass rate +20%; student +1–2 hoặc **negative result publishable**.

**Papers**: Filtered DPO 2404.13846; West-of-N 2401.12086; Rafailov 2406.02900.

---

### IDEA-27: Saturation Curve Characterization (SAT-1)

**Vấn đề**: Khan Tab.4 — mechanistic hiểu biết yếu về **noise saturation**.

**Cơ chế**: Sweep synth:real ∈ {0.5, 1, 2, 4, 8} × {unfiltered, filtered, CS-X} → plot accuracy vs ratio.

**Dự đoán**: Filtered peak shifts right (optimal ≥4:1) — **contribution lý thuyết** dù absolute gain nhỏ.

**Papers**: Khan 2023 Tab.4; DATED diversity guidelines 2305.09018.

---

### IDEA-28: NovelSum Diversity Report (NS-1)

**Vấn đề**: Không metric chuẩn cho "diversity" pseudo pool.

**Cơ chế**: NovelSum (2502.17184) trên embedding pool — correlate với student accuracy.

**Cách làm**: `scoring_only` mode → compute NovelSum before/after filter → report trong paper.

**Papers**: NovelSum 2502.17184; QDIT 2311.14736.

---

## Ma trận ưu tiên (impact × feasibility × novelty)

| Tier | Ideas | Lý do |
|------|-------|-------|
| **P0 — chạy ngay** | 01, 02, 03, 04, 27 | Core thesis + baseline bắt buộc |
| **P1 — chapter 2** | 17, 18, 21, 22, 13 | Iteration + PathVQA + diversity |
| **P2 — differentiation** | 05, 06, 11, 24, 07 | Novel gates / generation |
| **P3 — appendix** | 08, 09, 10, 25, 26, 28 | Ablation / negative results |

---

## Kết hợp đề xuất cho luận văn 4 chương thí nghiệm

1. **Chương 4.1**: CF-1 + TS-1 + CS-X + SAT-1 + random control (IDEA 01–04, 27)
2. **Chương 4.2**: LP-1 hoặc CR-1 gate mở rộng (IDEA 05–06)
3. **Chương 4.3**: IT-1 + J-1 curriculum (IDEA 17–18)
4. **Chương 4.4**: PathVQA direct + RG-1 + MT-1 (IDEA 21–23)
5. **Appendix**: RW-1, SemiReward, Maieutic, NovelSum (IDEA 10, 25, 26, 28)

---

## AI disclosure

Catalog assisted by AI; all arXiv IDs verified via Hugging Face paper index before inclusion.
