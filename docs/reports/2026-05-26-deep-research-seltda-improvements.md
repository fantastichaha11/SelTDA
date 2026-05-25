# Deep Research: Improving SelTDA — Evidence-Grounded Survey & Thesis Roadmap

- **Ngày**: 2026-05-26
- **Tác giả**: phong (thesis researcher) + Claude
- **Phạm vi**: 2022-2026
- **Mục tiêu**: Tìm đường nâng cấp SelTDA (Khan et al., CVPR 2023) vượt qua bản spec filtering hiện tại, dựa trên evidence thực sự kiểm chứng được (HF paper index + paper đọc full).

> **Phương pháp luận**: Báo cáo này thay thế report `2026-05-25-perplexity-filtering-and-paper-directions.html` vốn dựa vào output Perplexity chưa kiểm chứng (một số arXiv ID & con số trong đó là fabricated, đã loại). Tất cả paper trích dẫn ở đây đều có arXiv ID xác minh qua Hugging Face paper index hoặc đọc trực tiếp. Khi chỉ đọc abstract chứ không full text, sẽ **flag rõ** ở dòng đó. Số trong bảng SelTDA là đọc trực tiếp từ PDF.

---

## 1. Tóm tắt điều hành (Executive Summary)

**SelTDA (Khan et al., CVPR 2023, arXiv 2206.01718-style benchmark — paper PDF in repo)** đề xuất pipeline self-training cho VQA data-scarce: BLIP teacher fine-tuned thành VQG image-conditional (`VQG_IC`), sinh `(Q,A)` từ COCO unlabeled, BLIP student train trên `train ∪ synthetic`. Đã đọc full paper. Bốn fact quan trọng làm gốc cho phần còn lại của report:

1. **Pseudo-QA noise đã được tác giả đo lường** (Table 3, paper): External-Knowledge 62% answer correct, Visual-Reasoning 70%, Visual-Identification 88%. → **~30% sample sai-trên-tổng**, không phải giả thuyết — là số tác giả tự công bố. Đây là dư địa filtering thực.
2. **Synthetic data có "điểm bão hòa nhiễu"** (Table 4): peak A-OKVQA ở synthetic:real = 2:1 (60.01% val), tăng lên 4:1 thì rớt xuống 59.73%. → Càng nhiều synthetic chưa filter, càng có hại. Lập luận này strengthen ý nghĩa filtering.
3. **PathVQA chỉ tăng +1.67%** (25.09→26.76, Table 6) **vì là zero-shot từ A-OKVQA, không train trực tiếp**. Đây là weak result mà spec hiện tại có thể vượt qua nếu train SelTDA trực tiếp trên PathVQA (cộng với RAG/PEFT cho medical).
4. **Tác giả tự liệt kê 4 hướng future-work** (Section 5): fact-checking, logically-consistent self-reasoning ("Maieutic prompting"), chain-of-thought, billion-parameter VLMs. Đây là roadmap có thẩm quyền — gần như checklist.

**Bốn hướng đề xuất, xếp theo evidence strength × feasibility 1-GPU × scope thesis:**

| # | Hướng | Evidence strength | 1-GPU feasibility | Vi phạm §1.4 spec? | Thesis-novelty |
|---|---|---|---|---|---|
| **A** | Self-consistency + cycle-consistency filtering (mở rộng `xcons` gate hiện có) | Cao — nhiều paper SSL/VLM | Cao | Không | Trung |
| **B** | Iterative SelTDA (student-feedback driven, **DataEnvGym**) | Cao — Khan 2024 cùng tác giả | Trung (lặp lại training) | **Có** (cần ghi log + ablation) | Cao |
| **C** | RAG cho PathVQA branch (**MMed-RAG / RULE**) | Cao — published ICLR/Med | Trung | Không (chỉ ảnh hưởng `generate_questions.py`) | Cao |
| **D** | Counterfactual / diversity-aware generation (**DeFacto / SimpleStrat**) | Trung — paper mới | Cao | Không | Cao |

Hai hướng *bị bỏ qua* khỏi top-tier (vẫn nên giữ trong appendix luận văn):

- **Confidence calibration thuần (temperature scaling, ECE)**: hiệu quả vừa, marginal trên open-ended VQA, và spec hiện tại đã có placeholder. Đáng làm nhưng không phải đóng góp chính.
- **PEFT (LoRA/Adapter)**: cost-saving, không phải đóng góp khoa học. Đưa vào "implementation notes" thay vì chapter.

---

## 2. Bối cảnh & gốc evidence

### 2.1 SelTDA — sự thật từ paper (verified, đọc trực tiếp)

| Thông số | Giá trị | Trang |
|---|---|---|
| Backbone | BLIP-base ViT-B/16, pretrain 129M pairs | §4.1 |
| Decoding | Nucleus sampling top-p = 0.92 | §3.2 |
| Loss | NLL trên chuỗi `"Question: <q>? Answer: <a>."` | Eq. (1) |
| A-OKVQA baseline (BLIP) | 57.1 val | Tab.1 (e) |
| **A-OKVQA + SelTDA** | **62.1 val / 54.5 test** (+5.0 baseline, +1.8 vs prior SOTA) | Tab.1 (g) |
| BLIP_VQAv2 + SelTDA | 68.9 val / 59.5 test | Tab.1 (h) |
| **Optimal synth:real** | **2:1** (51k synth + 17k real → 60.01% A-OKVQA) | Tab.4 |
| ArtVQA grounded | 78.74 → 83.86 (+5.12) | Tab.2 |
| **PathVQA (zero-shot từ A-OKVQA)** | **25.09 → 26.76 (+1.67)** | Tab.6 |
| RSVQA (zero-shot) | 37.78 → 38.99 (+1.1) | Tab.6 |
| Numerical reasoning VQAv2 | 13.49 → 43.3 (+29.81) | Tab.7 |
| Robustness AdVQA | 31.06 → 37.43 (+6.37) | Tab.5 |
| **Pseudo-QA quality (manual eval n=100)** | EK 62%, VID 88%, VR 70% answer-correct | Tab.3 |

### 2.2 Limitations tác giả tự thừa nhận (Section 5)

Trích nguyên văn, paraphrased nhẹ:

1. *"Pseudo-QA pairs can be noisy. Combining SelTDA with methods for fact-checking based on external knowledge [Piktus et al., 2021 — "Web is your Oyster"], logically consistent self-reasoning [Jung et al., 2022 — "Maieutic prompting"], or chain-of-thought prompting [Wei et al., 2022] to rationalize answers may result in higher quality pairs for self-training."*
2. *"Learning the teacher model may fail for specialized domains (e.g. medical), because the vocabulary is too specialized."*
3. *"Biases in the VLM or pretraining data may be amplified by self-training, and addressing these biases may reduce multimodal shortcut learning."*
4. *"Self-training is yet to be explored with recently developed billion-parameter VLMs [Flamingo, BLIP-2]."*

→ **Đây là checklist tác giả mặc-nhiên-bảo-trợ**. Bất cứ direction nào match checklist này có đường biện hộ ngắn nhất khi viết related-work/contribution.

---

## 3. Hướng A — Cải thiện filtering: vượt qua spec hiện tại

Spec hiện tại có 3 gate `conf + itm + xcons`. Spec này tốt nhưng **đang ở mức "cơ bản"** so với SOTA 2024-2026 về pseudo-label selection. Bốn nâng cấp cụ thể (xếp theo evidence-effort ratio):

### A.1 Multi-view self-consistency (extension của `xcons`)

**Ý tưởng**: thay vì student forward 1 lần, sinh K view (augment ảnh hoặc nhiệt độ decoding khác nhau), giữ pseudo-QA chỉ khi K view nhất quán.

**Evidence**:
- **ViLP (Luo et al., Dec 2024, arXiv 2501.00569)** — self-improving framework cho VLM: tạo "good-bad" image pairs bằng pixel-level corruption + semantic corruption, dùng làm preference data tự huấn luyện. Cải thiện LLaVA-v1.5 và Cambrian trên benchmark ViLP có GPT-4 đạt 66.17%. Tinh thần "agreement across perturbations" của ViLP áp dụng được vào filtering: pseudo-QA chỉ giữ nếu robust qua corruption nhẹ. [abstract-level evidence; chưa đọc full]
- **STIC (Deng et al., May 2024, arXiv 2405.19716)** — Self-Training on Image Comprehension cho LVLM: dùng step-by-step prompt sinh "preferred response" + corrupted/misleading prompts sinh "dispreferred". Báo cáo **+4.0% trung bình trên 7 benchmark với 70% ít SFT data hơn**. Code public. [abstract-level evidence]
- **SQ-LLaVA (Sun et al., Mar 2024, arXiv 2403.11299)** — Self-Questioning huấn luyện cross-modal alignment qua việc model tự sinh và tự trả câu hỏi.

**Áp dụng cho SelTDA cụ thể**:
- Giữ gate `xcons` hiện tại (student trả lời lại).
- Thêm **multi-view agreement**: sinh 3 view của ảnh (color jitter nhẹ, crop nhỏ, blur) → student trả lời từng view → score = fraction-mode answer.
- Hoặc **decoding-temperature ensemble**: cùng student, 3 random seed, top-p khác nhau.
- Soft weighting `w = exp(-disagree/σ)` trong khuôn khổ duplicate-sampling (không vi phạm §1.4).

**Cost**: ~3× student forward cho ~100k pseudo-pool ≈ 1.25 giờ A5000 — feasible.

**Risk**: với open-ended VQA, "agreement" cần match semantic, không exact string. Cần SBERT/MiniLM match (đã có trong spec hiện tại).

### A.2 Cycle-consistency: Q → A → Q'

**Ý tưởng**: dùng tea + student để check: với pseudo-QA `(Q, A)` đã sinh trên ảnh `I`, ép student trả lại từ `(I, Q)` → `A'`. Nếu `A' ≈ A` → giữ. Đây chính là `xcons` gate. Nâng cấp = thêm **chiều ngược lại**: từ `(I, A)` sinh `Q'` (dùng cùng teacher), so sánh `Q` với `Q'` qua SBERT.

**Evidence**:
- **ConVQA (Ray et al., 2019, arXiv 1909.04696)** — Consistency Teacher Module dùng entailed question generation: paper thấy *"data augmentation module improve consistency of VQA models"*. Trên paper cũ nhưng tinh thần consistency-through-entailment vẫn relevant.
- **JointMatch (Zou & Caragea, Oct 2023, arXiv 2310.14583)** — adaptive class-wise thresholds + **cross-labeling giữa hai network**: agreement = giữ, disagreement = train signal. Áp dụng tinh thần này cho cycle-consistency: nếu Q-cycle agrees, mạnh; nếu disagree, có thể flag là sample khó (không nhất thiết loại).

**Áp dụng**:
- Bước thêm sau Gate 3 (xcons): teacher `VQG_IC` hiện đã có khả năng sinh Q conditional `(I, A)` nhờ template "Question: ...? Answer: ..." — nhưng paper Khan sinh `(Q, A)` đồng thời. Để sinh `Q | (I, A)`, ta force template `"Question: <??>. Answer: <A>"` và để teacher fill `<??>`. **Spec hiện tại không có** bước này.
- Cost: 1 thêm teacher forward per pseudo → ~30% tăng wallclock generation.

**Risk**: Q' có thể đúng nghĩa nhưng diễn đạt khác → cần SBERT match, không exact. Đã có cơ chế trong spec.

### A.3 Calibration (temperature scaling) — đáng làm, không phải đóng góp chính

**Evidence** (chỉ abstract-level):
- **SoC: Semantic Orthogonal Calibration (Fillioux et al., Jan 2026, arXiv 2601.08617)** — calibration cho test-time prompt tuning của VLM, dùng Huber regularizer + prototype separation. Mới, claim "discriminative performance maintained" — chưa đủ evidence chuyên cho pseudo-label filtering.
- **Post-hoc Probabilistic VLMs (Baumann et al., Dec 2024, arXiv 2412.06014)** — Bayesian posterior cho CLIP/SigLIP, "well-calibrated predictive uncertainties" + sample-efficient active learning. Có code.
- **Confidence Separable Learning (CSL, Liu & Liu, Sep 2025, arXiv 2509.16704)** — argue "network overconfidence" làm confidence threshold không reliable. Dùng convex optimization + random masking. Trên semantic segmentation, không VQA — transferability phải tự test.

**Áp dụng**: spec hiện tại đã có placeholder calibration. Concretely:
- Fit temperature `T` trên 200-500 val labeled (A-OKVQA val, PathVQA val).
- Đo ECE trước/sau, report trong appendix.

**Đừng over-claim**. Calibration là **hygiene**, không phải contribution. Nhiều paper SSL gần đây (CSL, SemiReward) đã chỉ ra confidence một mình *không đủ* — phải kết hợp uncertainty + agreement (như A.1, A.2 ở trên).

### A.4 Reward-model based filtering

**Evidence**:
- **SemiReward (Li et al., Oct 2023, arXiv 2310.03013)** — train một reward model dự đoán "quality score" của pseudo-label, replace cứng-rắn confidence threshold. Tích hợp với FlexMatch/FreeMatch, claim "performance gains and convergence speeds" trên SSL classification/regression.
- **Prototype-Guided Pseudo-Labeling (Ali et al., Jul 2025, arXiv 2507.22075)** — CLIP unsupervised adaptation: kết hợp prototype consistency + neighborhood-based consistency + adaptive weighting. Achieves "state-of-the-art performance" (per abstract).

**Áp dụng cho SelTDA**:
- Train một lightweight reward model (e.g., 2-layer MLP trên BLIP embedding) trên ~500 hand-judged kept/dropped pseudo từ §5.4 của spec.
- Reward này thay vì rule-based 3-gate.
- **Cảnh báo**: cost rất cao — cần hand-label data. Đề xuất chỉ chạy như **mini-ablation** trong luận văn để show rule-based đã đủ tốt vs. learned reward.

---

## 4. Hướng B — Iterative SelTDA (DataEnvGym successor)

**Đây là direction được underutilized nhất trong spec hiện tại, và là direction có authority highest.**

### 4.1 Bằng chứng cốt lõi

**DataEnvGym (Khan, Stengel-Eskin, Cho, Bansal — Oct 2024, arXiv 2410.06215)** — cùng tác giả Zaid Khan với SelTDA. Đọc full abstract:

> *"DataEnvGym frames data generation as a sequential decision-making task… An agent consisting of a data generation policy (which generates a plan for creating training data) and a data generation engine (which transforms the plan into data), inside an environment that provides student feedback… Students are iteratively trained and evaluated on generated data, with their feedback (in the form of errors or weak skills) being reported to the agent after each iteration. Supports 3 diverse tasks (math, code, and VQA)…"*

Tức là: SelTDA paper 2023 = 1 round teacher-student. DataEnvGym 2024 = N round, có feedback signal từ student. **Đây chính là "noisy student" cho VQA**, do chính tác giả SelTDA viết.

### 4.2 Áp dụng cho thesis

**Phase 2 của thesis** (sau khi Phase 1 filtering ổn): chạy SelTDA nhiều vòng:
- Vòng 1: VQG_IC (teacher) → pseudo → filter → student v1.
- Vòng 2: **Đo skill-gap của student v1 trên val** (theo question type hoặc theo dataset slice). Re-fine-tune VQG_IC trên data type student v1 yếu → re-generate → filter → student v2.
- Lặp tới khi gain bão hòa.

**Vi phạm §1.4?** Spec hiện tại cấm sửa `train_vqa.py`. Iterative chỉ cần *lặp lại* training với data khác nhau — không sửa code training. Chỉ cần script orchestrator + lưu skill-gap report. **Không vi phạm**.

### 4.3 Citation strategy cho luận văn

- SelTDA 2023 → cite làm baseline.
- DataEnvGym 2024 → cite làm "extension framework" và biện hộ rằng iterative direction đã được tác giả gốc validated.
- **Đóng góp luận văn**: cụ thể hóa cho data-scarce VQA + PathVQA (mà DataEnvGym chưa làm), với filtering chặt chẽ. Đây là contribution có cửa publication tốt.

### 4.4 Compute budget

DataEnvGym test 3 task; cho VQA mỗi iteration ~ baseline cost của SelTDA Phase 1. → 3 iteration ≈ 3× chi phí Phase 1. Theo bảng §5.6 spec hiện tại: ~10h/iteration × 2 dataset × 3 iter ≈ 60h ≈ 3 ngày 1 GPU. Khả thi.

---

## 5. Hướng C — RAG cho PathVQA branch (đánh trực tiếp Limitation #2)

**Đây là direction match Limitation #2 của paper gốc (medical vocabulary). Evidence rất mạnh.**

### 5.1 Bằng chứng

- **MMed-RAG (Xia, Zhu, Li, Wang et al., Oct 2024, arXiv 2410.13085)** — "Versatile Multimodal RAG System for Medical Vision Language Models". Three components: (i) **domain-aware retrieval**, (ii) **adaptive retrieved contexts selection**, (iii) **preference fine-tuning strategy**. Cho Med-LVLM. Code public. [abstract-level, đã upvoted 25 times trên HF — credibility cao]
- **RULE (Xia et al., Jul 2024, arXiv 2407.05131)** — đi trước MMed-RAG cùng tác giả: calibrate retrieved-context selection + preference dataset. [abstract-level]
- **MIRAGE (Xiong et al., Feb 2024, arXiv 2402.13178)** — benchmark RAG cho medical QA, claim **+18% LLM performance** với MedRAG toolkit + CoT. [abstract-level]
- **i-MedRAG (Xiong et al., Aug 2024, arXiv 2408.00727)** — iterative RAG với follow-up queries cho medical QA (chủ yếu text), ý tưởng có thể port sang medical VQA.

### 5.2 Áp dụng cho SelTDA PathVQA branch

**Pipeline đề xuất**:
1. Build small medical KB (PubMed abstracts? PathVQA training data captions? Quilt-1M caption từ Quilt-LLaVA?).
2. Cho mỗi unlabeled ảnh pathology, retrieve top-K relevant context bằng image-similarity (CLIP/BiomedCLIP) hoặc caption-similarity.
3. Concatenate context vào prompt teacher VQG_IC khi sinh `(Q, A)`.
4. Filter như cũ.

**Spec hiện tại §3.3 đã chuẩn bị fallback BiomedCLIP cho ITM gate** — phù hợp ngẫu nhiên với hướng này.

**Vi phạm §1.4?** Chỉ ảnh hưởng `generate_questions.py` + `models/blip.py` (đã được phép chạm). **Không vi phạm**.

### 5.3 Critical assessment

- **MMed-RAG abstract claim hấp dẫn nhưng tôi chưa verify số trên PathVQA cụ thể**. Phải đọc full paper trước khi build.
- **Reproducibility risk**: medical KB cần curate. Cần check repo MMed-RAG có cung cấp KB hay không.
- **PathVQA branch là single-dataset trong luận văn** — cũng acceptable nếu A-OKVQA branch dùng pipeline khác (cleaner contribution story: "filtering universal, RAG specific to medical").

### 5.4 Modern medical VLM backbones (alternative/complement)

Liệt kê cho appendix luận văn, không phải lựa chọn chính:

| Model | arXiv | Note |
|---|---|---|
| **LLaVA-Med** (Li et al., Jun 2023) | 2306.00890 | Train 1 day, GPT-4 generated instruction. Public weights. |
| **PMC-VQA** (Zhang et al., May 2023) | 2305.10415 | Large-scale instruction tuning data. |
| **Quilt-LLaVA** (Seyfioglu et al., Dec 2023) | 2312.04746 | Histopathology specifically — directly relevant to PathVQA. |
| **LLaDA-MedV** (Dong et al., Aug 2025) | 2508.01617 | Diffusion-based medical VLM, **claim SOTA on VQA-RAD/SLAKE/PathVQA** per abstract — chưa verify. |
| **OmniMedVQA** (Hu et al., Feb 2024) | 2402.09181 | Multi-modality medical VQA benchmark. |
| **GEMeX** (Liu et al., Nov 2024) | 2411.16778 | Chest X-ray VQA, groundable explainable. |

**Quan trọng**: nếu thesis chỉ có 1 GPU và muốn fair compare với SelTDA baseline, **stick với BLIP-base**. Nhảy sang LLaVA-Med = đổi backbone hoàn toàn = không còn SelTDA paper nữa. Đó là một thesis khác. Chỉ dùng LLaVA-Med như **reference SOTA** để định vị PathVQA accuracy ceiling.

---

## 6. Hướng D — Counterfactual & diversity-aware generation (đánh Limitation #3, #1 đa diện)

### 6.1 Counterfactual generation

**Bằng chứng**:
- **DeFacto (Xu et al., Sep 2025, arXiv 2509.20912)** — "Counterfactual Thinking with Images". Random-masking + GRPO RL để ép evidence-grounded reasoning. Áp dụng cho VLM (không chỉ video). [abstract-level]
- **CounterVQA (Chen et al., Nov 2025, arXiv 2511.19923)** — video benchmark + post-training. Direct counterfactual reasoning evaluation. [abstract-level]
- **SpuriVerse (Yang et al., Jun 2025, arXiv 2506.18322)** — benchmark spurious correlation trong LVLM, **claim "synthetic counterfactual evaluation" + targeted fine-tuning encourages contextual attention over shortcuts**.

**Áp dụng**:
- Sinh pseudo-QA *trên ảnh đã augment counterfactually* (mask object, swap region) → ép student không dùng shortcut.
- Đo robustness trên VQA-CE (đã có trong paper gốc Tab.5).

**Đánh trực tiếp Limitation #3 (bias amplification)**. Citation chính trực: paper SelTDA tự nói *"addressing these biases may reduce multimodal shortcut learning"*.

### 6.2 Diversity-aware sampling

**Bằng chứng**:
- **On Diversity of Synthetic Data (Chen et al., Oct 2024, arXiv 2410.15226)** — kết luận chính: **"diversity in synthetic data positively correlates with performance… more significant impact during fine-tuning"**. Đo bằng "LLM cluster-agent" diversity metric. Áp dụng được cho VQA: cluster pseudo-Q và force coverage.
- **SimpleStrat (Wong et al., Oct 2024, arXiv 2410.09038)** — stratified sampling: LLM partition response space → sample random strata. Increase diversity và quality. Outperforms temperature adjustment.
- **CorrSynth (Kowshik, Divekar, Malik, Nov 2024, arXiv 2411.08553)** — correlated sampling strategy cho diverse dataset gen từ LLM.

**Áp dụng cho SelTDA**:
- Paper gốc admits "how" → "many" dominance trong sunburst chart (Fig 6). Diversity issue đã được tác giả nhận diện *gián tiếp*.
- Đề xuất: **cluster pseudo-Q theo first-word/question-type, enforce balanced sampling** trong filter step. Đây là 1 dòng code trong `filter_pseudo.py`, miễn phí.
- Hoặc **conditional prompt teacher**: chia template thành 5-7 question type (counting, color, location, action, knowledge, yes-no, comparison) và prompt teacher từng type. Đây là sửa `generate_questions.py` — được phép theo §1.4.

### 6.3 ConVQA-style entailed question augmentation

**ConVQA (Ray et al., 2019, arXiv 1909.04696)** — generate logically-entailed Q-A từ Q-A gốc. Cải thiện consistency. Áp dụng vào SelTDA: với mỗi pseudo `(Q, A)`, sinh thêm 1-2 entailed `(Q', A')` để augment đa dạng logic chain.

Ý tưởng cũ (2019) nhưng underused trong SelTDA pipeline. **Đáng làm như mini-extension**.

---

## 7. Đánh giá phản biện (Devil's Advocate)

### 7.1 Counter-evidence và rủi ro với từng hướng

**A — Filtering**:
- Số 30% noise của SelTDA là *manual annotation 100 mẫu, CI 95% wide* (Table 3 spec). Có thể overestimate noise. Filter có thể chỉ thu lại 1-2 point accuracy thay vì 5-10 như A.1 mong đợi.
- Spec hiện tại đã giả định filter sẽ tăng accuracy. Cần *test scoring-only mode trước* để xem distribution score trước khi đặt threshold.

**B — Iterative SelTDA**:
- Risk lớn nhất: **error amplification**. Student v1 yếu chỗ X → teacher v2 sinh thêm data X → student v2 vẫn yếu hoặc *học sai*. Đây là vấn đề kinh điển của noisy student. DataEnvGym chỉ chứng minh trong toy/math setting; chưa proven cho VQA at scale.
- Mitigation: trộn back data cũ, không replace; có guard rail (e.g., chỉ thêm data type student yếu *nếu confidence của teacher cao*).

**C — RAG cho PathVQA**:
- KB curation cost thực tế cao hơn dự kiến. Nếu KB nghèo, RAG vô dụng hoặc gây nhiễu.
- Retrieval failure mode: retrieve sai context → teacher generate Q-A liên quan tới context sai → student học bậy.
- Mitigation: bắt buộc có gate **retrieval-relevance** trước khi feed vào teacher.

**D — Counterfactual**:
- Augmentation domain-aware khó: object masking trên ảnh y khoa có thể tạo artifact phi-thực.
- DeFacto/CounterVQA dùng GRPO RL — cần policy/reward; có thể không fit 1 GPU.
- Alternative đơn giản: chỉ áp dụng "diverse decoding via question-type conditioning" (D.2) — đỡ rủi ro hơn.

### 7.2 Câu hỏi mở (cần test mới biết)

1. **Filtering có scale với synth:real ratio không?** Paper gốc thấy peak 2:1, drop ở 4:1. Liệu filter chặt cho phép đẩy lên 5:1 vẫn tăng accuracy? Đây là **headline ablation** đáng chạy.
2. **PathVQA train-directly vs zero-shot**: spec hiện tại train trực tiếp. Paper gốc zero-shot. Just doing this is already a stronger experiment than the paper's PathVQA reporting.
3. **Quilt-LLaVA / LLaVA-Med như teacher**: backbone thay đổi vi phạm "fair compare" — nhưng có thể chạy như reference contour ở appendix.

---

## 8. Bảng tổng kết Recommended Roadmap

| Phase | Mục tiêu | Direction | Files chạm | Compute | Vi phạm §1.4 spec? |
|---|---|---|---|---|---|
| **1a** | Filter Phase 1 (đang dở) | A.1 + A.2 + calibration (A.3) | `filtering/`, `filter_pseudo.py` | 1× wk 1 GPU | Không |
| **1b** | Threshold sweep + scoring-only audit | A — verify | (same) | 1× sweep day | Không |
| **2a** | Diverse-question generation | D.2 (question-type conditioning) | `generate_questions.py` config | + 50% generation cost | Không |
| **2b** | Counterfactual augmentation lite | D.1 (light: object masking only) | `generate_questions.py` | + 30% | Không |
| **3** | Iterative SelTDA (2-3 rounds) | B (DataEnvGym successor) | orchestrator script, không chạm train_vqa.py | 3× Phase 1 cost | Không (chỉ lặp) |
| **4** | PathVQA + RAG | C (MMed-RAG style) | `generate_questions.py`, new `retrieval/` module | + 1 wk setup KB | Không |
| **5** | (Optional) PEFT comparison | — | new config | + 1 wk | Không |

**Roadmap defensive**: Phase 1 đủ cho một workshop paper. Phase 1+3 đủ cho một venue tier-2. Phase 1+3+4 đủ cho tier-1 (CVPR/ICCV) nếu numbers tốt.

---

## 9. Honest limitations của report này

1. **Một số paper chỉ đọc abstract** (đã flag rõ). Trước khi build, cần đọc full MMed-RAG, DeFacto, STIC, ViLP, DataEnvGym để verify mechanism. Tôi không guarantee các % gain các paper claim.
2. **Không có benchmark trực tiếp cho self-training trên PathVQA**. SelTDA chỉ chạy zero-shot, không train trực tiếp. Khoảng trống này = cửa luận văn, nhưng cũng = không có baseline so sánh trực tiếp.
3. **Cutoff knowledge tháng 1/2026**: paper từ Mar 2026 trở đi có thể bị bỏ lỡ. HF index có một số paper 2026 nhưng arXiv ID nằm trong format kỳ lạ — tôi không cite những paper sau cutoff.
4. **Verification gap**: không kiểm chứng được code/reproducibility của MMed-RAG, LLaDA-MedV, STIC bằng cách thực tế chạy chúng — chỉ trust abstract + upvote count trên HF.
5. **Numbers từ paper gốc đã đọc full** = solid. Mọi số % gain của các paper khác = abstract-level, có thể cherry-picked.

## 10. Tham chiếu chính (verified arXiv IDs)

### Core (đã đọc full)
- Khan, Z., Kumar BG, V., Yu, X., Schulter, S., Fu, Y., Chandraker, M. (2023). *Q: How to specialize large vision-language models to data-scarce VQA tasks? A: Self-train on unlabeled images!* CVPR 2023. [PDF in repo]

### Direct authoritative successor (Khan as first author)
- Khan, Z., Stengel-Eskin, E., Cho, J., Bansal, M. (2024). *DataEnvGym: Data Generation Agents in Teacher Environments with Student Feedback.* arXiv [2410.06215](https://arxiv.org/abs/2410.06215)

### Pseudo-label filtering & SSL
- Li, S., et al. (2023). *SemiReward: A General Reward Model for Semi-supervised Learning.* arXiv [2310.03013](https://arxiv.org/abs/2310.03013)
- Zou, H. P., Caragea, C. (2023). *JointMatch.* arXiv [2310.14583](https://arxiv.org/abs/2310.14583)
- Yang, L., et al. (2023). *ShrinkMatch.* arXiv [2308.06777](https://arxiv.org/abs/2308.06777)
- Ali, E., Arora, C., Khan, M. H. (2025). *Prototype-Guided Pseudo-Labeling with Neighborhood-Aware Consistency for Unsupervised Adaptation.* arXiv [2507.22075](https://arxiv.org/abs/2507.22075)
- Liu, P., Liu, J. (2025). *When Confidence Fails: Revisiting Pseudo-Label Selection (CSL).* arXiv [2509.16704](https://arxiv.org/abs/2509.16704)

### VLM calibration & uncertainty
- Baumann, A., et al. (2024). *Post-hoc Probabilistic Vision-Language Models.* arXiv [2412.06014](https://arxiv.org/abs/2412.06014)
- Fillioux, L., et al. (2026). *SoC: Semantic Orthogonal Calibration for Test-Time Prompt Tuning.* arXiv [2601.08617](https://arxiv.org/abs/2601.08617)

### Self-improvement / self-consistency for VLM
- Deng, Y., et al. (2024). *Enhancing Large Vision Language Models with Self-Training on Image Comprehension (STIC).* arXiv [2405.19716](https://arxiv.org/abs/2405.19716)
- Luo, T., et al. (2024). *Probing Visual Language Priors in VLMs (ViLP).* arXiv [2501.00569](https://arxiv.org/abs/2501.00569)
- Sun, G., et al. (2024). *SQ-LLaVA: Self-Questioning for Large Vision-Language Assistant.* arXiv [2403.11299](https://arxiv.org/abs/2403.11299)
- Hu, W., et al. (2025). *Socratic Questioning: Learn to Self-guide Multimodal Reasoning in the Wild.* arXiv [2501.02964](https://arxiv.org/abs/2501.02964)
- Awal, R., Zhang, L., Agrawal, A. (2023). *Investigating Prompting Techniques for Zero- and Few-Shot VQA.* arXiv [2306.09996](https://arxiv.org/abs/2306.09996)

### Diversity-aware synthetic data
- Chen, H., et al. (2024). *On the Diversity of Synthetic Data and its Impact on Training LLMs.* arXiv [2410.15226](https://arxiv.org/abs/2410.15226)
- Wong, J., et al. (2024). *SimpleStrat: Diversifying LM Generation with Stratification.* arXiv [2410.09038](https://arxiv.org/abs/2410.09038)
- Kowshik, S. S., Divekar, A., Malik, V. (2024). *CorrSynth.* arXiv [2411.08553](https://arxiv.org/abs/2411.08553)
- Su, X., et al. (2024). *SK-VQA: Synthetic Knowledge Generation at Scale.* arXiv [2406.19593](https://arxiv.org/abs/2406.19593)
- Ray, A., et al. (2019). *Improving Answer Consistency in VQA through Entailed Question Generation (ConVQA).* arXiv [1909.04696](https://arxiv.org/abs/1909.04696)

### Counterfactual / shortcut mitigation
- Xu, T., et al. (2025). *DeFacto: Counterfactual Thinking with Images.* arXiv [2509.20912](https://arxiv.org/abs/2509.20912)
- Chen, Y., et al. (2025). *CounterVQA.* arXiv [2511.19923](https://arxiv.org/abs/2511.19923)
- Yang, Y., et al. (2025). *SpuriVerse: Can LVLMs Generalize Beyond Seen Spurious Correlations?* arXiv [2506.18322](https://arxiv.org/abs/2506.18322)

### Medical VQA & RAG
- Li, C., et al. (2023). *LLaVA-Med.* arXiv [2306.00890](https://arxiv.org/abs/2306.00890)
- Zhang, X., et al. (2023). *PMC-VQA.* arXiv [2305.10415](https://arxiv.org/abs/2305.10415)
- Seyfioglu, M. S., et al. (2023). *Quilt-LLaVA.* arXiv [2312.04746](https://arxiv.org/abs/2312.04746)
- Xia, P., et al. (2024). *MMed-RAG.* arXiv [2410.13085](https://arxiv.org/abs/2410.13085)
- Xia, P., et al. (2024). *RULE: Reliable Multimodal RAG for Medical VLMs.* arXiv [2407.05131](https://arxiv.org/abs/2407.05131)
- Xiong, G., et al. (2024). *MIRAGE: Benchmarking RAG for Medicine.* arXiv [2402.13178](https://arxiv.org/abs/2402.13178)
- Xiong, G., et al. (2024). *i-MedRAG.* arXiv [2408.00727](https://arxiv.org/abs/2408.00727)
- Liu, B., et al. (2024). *GEMeX.* arXiv [2411.16778](https://arxiv.org/abs/2411.16778)
- Hu, Y., et al. (2024). *OmniMedVQA.* arXiv [2402.09181](https://arxiv.org/abs/2402.09181)
- Gai, X., et al. (2024). *MedThink.* arXiv [2404.12372](https://arxiv.org/abs/2404.12372)
- Dong, X., et al. (2025). *LLaDA-MedV.* arXiv [2508.01617](https://arxiv.org/abs/2508.01617)
- Liu, B., et al. (2021). *SLAKE.* arXiv [2102.09542](https://arxiv.org/abs/2102.09542)

---

## 11. Đề xuất bước kế tiếp ngay

1. **Đọc full** 3 paper sau (ưu tiên cao trước khi build): DataEnvGym, MMed-RAG, STIC.
2. **Run scoring-only mode** (đã có flag trong spec §1.1.5) trên một pool 5k pseudo-QA của A-OKVQA → vẽ histogram s_conf, s_itm, s_xcons → quan sát phân phối thực, không guess threshold.
3. **Commit Phase 1a + 1b** từ §8 trước. Đó là path ngắn nhất tới một experimental result để decide xem các Phase sau có làm hay không.
4. **Skip ngay** mọi thứ trong report `2026-05-25-perplexity-filtering-and-paper-directions.html` mà không xuất hiện ở report này — vài citation trong đó là fabricated.

---

*Report version 1 — 2026-05-26. Based on full read of SelTDA paper + Hugging Face paper index (verified arXiv IDs) + critical review of prior Perplexity report. Some references read at abstract level only (flagged inline). No claim grade for unverified paper percentages.*
