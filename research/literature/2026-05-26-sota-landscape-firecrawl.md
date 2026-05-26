# SOTA Landscape — Firecrawl Survey (ngoài arXiv)

- **Date**: 2026-05-26
- **Nguồn**: Firecrawl search + scrape (Wizwand, OpenCodePapers, CodeSOTA, OK-VQA official, DataEnvGym site, SemIAug MDPI, SelTDA GitHub)
- **Mục đích**: Đặt SelTDA vào bối cảnh benchmark thực tế — không chỉ paper — để calibrate kỳ vọng thesis và ưu tiên ý tưởng

---

## 1. Tóm tắt điều hành

| Benchmark | SelTDA (BLIP-base, DA) | Frontier gần nhất | Khoảng cách | Ghi chú metric |
|-----------|------------------------|-------------------|-------------|----------------|
| A-OKVQA val (DA) | 57.1 → **62.1** (+5.0) | QACap **73.4%** val DA (CVPR 2025) | ~11 điểm | Cùng open-ended DA; khác backbone (LLM+KB) |
| A-OKVQA test (MC) | — (SelTDA dùng DA) | HinD-CoT-Know **87.2%** (7B, Nov 2025) | — | MC ≠ DA; BLIP-2 MC test **80.2%** |
| OK-VQA test | 31.65 → **38.03** (+6.38) | Prophet **61.11%** (official leaderboard) | ~23 điểm | OK-VQA leaderboard chính thức |
| PathVQA test | 25.09 → **26.76** zs (+1.67) | VILA-M3 **92.7%** overall | ~66 điểm | **Metric khác nhau** — xem §4 |
| VQA v2 test-dev | BLIP CapFilt-L ~78.3% | Qwen2-VL 72B **87.6%** | ~9 điểm | General VQA đã bão hòa |

**Insight chính**: SelTDA cải thiện có ý nghĩa trong **niche data-scarce + BLIP-base + không cần KB pipeline**, nhưng absolute SOTA trên A-OKVQA/PathVQA đã do các hệ **LLM + retrieval + CoT + medical pretrain quy mô lớn** nắm giữ. Thesis nên frame contribution là **chất lượng pseudo-label / data efficiency**, không phải đuổi HinD-CoT-Know 87.2%.

---

## 2. A-OKVQA

### 2.1 Leaderboard Multi-Choice (test)

Nguồn: [Wizwand — A-OKVQA MC test](https://www.wizwand.com/sota/visual-question-answering-multi-choice-on-a-okvqa-test) (cập nhật ~2025-12)

| Rank | Method | Accuracy | Ghi chú |
|------|--------|----------|---------|
| 1 | HinD-CoT-Know | **87.2%** | 7B, knowledge + CoT |
| 2 | HinD-Know | 85.9% | 7B |
| 3 | ReAuSE | 85.0% | |
| 4 | BLIP-2 (FlanT5-XXL) | 80.2% | Cùng họ BLIP |
| 5 | LLaVA-1.5 7B | 77.1% | |
| 6 | InstructBLIP | 76.7% | |
| 7 | Prophet | 76.4% | KB-VQA classic |
| 8 | PromptCap | 73.2% | |
| — | GPV-2 (2022) | 53.7% | Baseline cũ trên OpenCodePapers |

Nguồn bổ sung: [OpenCodePapers A-OKVQA](https://opencodepapers-b7572d.gitlab.io/benchmarks/visual-question-answering-on-a-okvqa.html) — PaLI-X specialist MC **83.75%**, DA VQA score **70.55** (Dec 2023).

### 2.2 Direct Answer (open-ended) — liên quan trực tiếp SelTDA

| Method | Split | Metric | Score | Nguồn |
|--------|-------|--------|-------|-------|
| SelTDA (BLIP-base) | val | DA accuracy | **62.1%** | Khan CVPR 2023 Tab.4 |
| SelTDA baseline | val | DA accuracy | 57.1% | Khan CVPR 2023 |
| QACap | A-OKVQA val | DA | **73.4%** | CVPR 2025 poster 33258 |
| QACap | OK-VQA val | — | 68.2% | CVPR 2025 |
| SemIAug + BLIP | A-OKVQA | DA | ~55.8% (+1.4 vs BLIP) | MDPI SemIAug 2024 |
| SemIAug + BLIP | OK-VQA | DA | ~56.5% (+1.2) | MDPI SemIAug 2024 |
| GPV-2 | test | DA VQA score | 40.7% | A-OKVQA paper |

Wizwand dataset page (meta): A-OKVQA val Accuracy **79.5%**, test **89.17%** — có thể gộp nhiều task variant; **không dùng trực tiếp** mà không đối chiếu paper gốc.

### 2.3 Xu hướng SOTA (2023→2026)

1. **Knowledge retrieval + LLM reasoning** (Prophet → PromptCap → HinD-CoT-Know) chiếm top MC.
2. **BLIP-2 / InstructBLIP** vượt BLIP-base ~15–20 điểm MC mà **không** self-training — implication: teacher lớn hơn có thể quan trọng hơn filter nếu §1.4 cho phép đổi teacher checkpoint (generation only).
3. **SemIAug** (Feb 2024) là competitor trực tiếp cùng niche data-scarce + BLIP, gain nhỏ (+1.4%) nhưng **không cần unlabeled images** — có thể compose với SelTDA.

---

## 3. OK-VQA (benchmark anh em)

Nguồn chính thức: [okvqa.allenai.org/leaderboard](https://okvqa.allenai.org/leaderboard.html)

| Rank | Model | Overall Accuracy |
|------|-------|------------------|
| 1 | Prophet | **61.11%** |
| 2 | PromptCap | 60.4% |
| 3 | REVIVE | 58.0% |
| 4 | KAT | 54.41% |

SelTDA trên OK-VQA (BLIP-base, open-ended): **31.65 → 38.03** (+6.38) — gain lớn theo tỷ lệ % nhưng absolute vẫn thấp vì không dùng external KB.

CodeSOTA: trang OK-VQA **chưa index kết quả** (2026) — Papers with Code offline ([blog TIB 2025](https://blog.tib.eu/2025/10/02/papers-with-code-went-offline-the-knowledge-doesnt-have-to/)); Wizwand/OpenCodePapers thay thế một phần.

---

## 4. PathVQA (medical) — metric trap quan trọng

Nguồn: [Wizwand PathVQA test](https://www.wizwand.com/sota/visual-question-answering-on-pathvqa-test)

| Method | Overall Acc | Free-form Acc | Yes/No Acc | Recall |
|--------|-------------|---------------|------------|--------|
| VILA-M3 40B | **92.7%** | — | — | — |
| VILA-M3 8B | 91.0% | — | — | — |
| Med-Gemini 1.5T | 83.3% | — | — | — |
| Quilt-LLaVA 7B | 58.7% | — | — | 15.3% |
| LLaVA-Med 7B | 56.2% | — | — | 12.0% |
| MMQ (BAN) | 48.8% | **13.4%** | **84.0%** | — |
| SelTDA BLIP zs | **26.76%** | (VQA acc metric) | — | — |

**Cảnh báo phương pháp luận**:

- Leaderboard **Overall** gộp yes/no (dễ) và free-form (cực khó).
- SelTDA paper dùng **VQA soft accuracy** trên open-ended — **không so sánh trực tiếp** với 92.7% VILA-M3.
- Free-form accuracy của các model cổ điển chỉ **1.6–13.4%** — đúng pain point Khan 2023 nêu (medical vocabulary).
- Thesis PathVQA branch **bắt buộc** báo cáo stratified: yes/no vs free-form vs overall + recall (theo BESTMVQA convention).

Nguồn bổ sung: BESTMVQA (ECML PKDD 2024) — unified eval closed/open/overall trên PathVQA, VQA-RAD, SLAKE.

---

## 5. VQA v2 — bối cảnh VLM chung

Nguồn: [CodeSOTA VQA v2](https://www.codesota.com/benchmark/vqa-v2) (2026)

| Model | test-dev accuracy |
|-------|-------------------|
| Qwen2-VL 72B | **87.6%** |
| InternVL2-76B | 87.2% |
| Gemini 1.5 Pro | 86.5% |
| BLIP-2 FlanT5-XXL | 82.19% |
| BLIP CapFilt-L | 78.32% |

CodeSOTA ghi nhận VQAv2 **saturated** (>85%); leaderboard attention chuyển sang MMMU, OK-VQA, A-OKVQA.

---

## 6. DataEnvGym — cùng tác giả SelTDA

Nguồn: [dataenvgym.github.io](https://dataenvgym.github.io/) (ICLR 2025 Spotlight)

- VQA task trong DataEnvGym = **GQA + NaturalBench**, không phải A-OKVQA.
- Multimodal leaderboard (NaturalBench): Skill-Tree + GPT-4o → **+5.58** student improvement; Open-Ended + GPT-4o → +3.72.
- Skill-Tree + GPT-4o-mini → +4.65.
- **Implication**: iterative feedback loop đã validate trên VQA-like tasks; chưa có public số trên A-OKVQA/PathVQA.

Repo: [github.com/codezakh/DataEnvGym](https://github.com/codezakh/DataEnvGym) — cùng maintainer với [SelTDA](https://github.com/codezakh/SelTDA) (17 stars, last update ~2024).

---

## 7. Competitor & adjacent work (không phải SOTA table)

| Work | Loại | Relevance |
|------|------|-----------|
| SemIAug (MDPI 2024) | Cross-image Q shuffle trên labeled data | +1.2–1.4% A-OKVQA/OK-VQA, BLIP — compose với SelTDA |
| HALT-MedVQA (arXiv 2401.05827) | Hallucination benchmark | PathVQA subset — filter nên đo hallucination rate, không chỉ accuracy |
| HEAL-MedVQA (IJCAI 2025) | Grounded Med-VQA 67K QA | Localization-aware eval — future work |
| MedXpertQA / PMC-VQA | New hard benchmarks | PathVQA có thể quá dễ ở yes/no; free-form vẫn hard |

---

## 8. Gap analysis → hướng research (map sang ideas)

| Gap quan sát được | Cơ hội cho SelTDA+filter |
|-------------------|--------------------------|
| KB-SOTA (Prophet/QACap/HinD) >> BLIP SelTDA | Gate thứ 4: knowledge consistency (retrieve → check answer) |
| SemIAug orthogonal (+1.4% không unlabeled) | Pipeline: SemIAug trên real + SelTDA synthetic filtered |
| PathVQA yes/no vs free-form split | Type-stratified filter + RAG teacher cho free-form only |
| DataEnvGym iterative +5.58 on NaturalBench | IT-1 wrap filter trong Skill-Tree loop |
| HALT hallucination >> accuracy | Medical gate: consistency under image perturbation |
| Metric không comparable | MCR-1: báo cáo stratified + document protocol |

Chi tiết ý tưởng: `research/ideation/2026-05-26-sota-informed-ideas.md`

---

## 9. Nguồn tham chiếu nhanh

- Wizwand A-OKVQA MC: https://www.wizwand.com/sota/visual-question-answering-multi-choice-on-a-okvqa-test
- Wizwand PathVQA: https://www.wizwand.com/sota/visual-question-answering-on-pathvqa-test
- OpenCodePapers A-OKVQA: https://opencodepapers-b7572d.gitlab.io/benchmarks/visual-question-answering-on-a-okvqa.html
- OK-VQA official: https://okvqa.allenai.org/leaderboard.html
- CodeSOTA VQA v2: https://www.codesota.com/benchmark/vqa-v2
- DataEnvGym: https://dataenvgym.github.io/
- SemIAug: https://www.mdpi.com/2813-0324/9/1/3
- SelTDA GitHub: https://github.com/codezakh/SelTDA
- QACap CVPR 2025: https://cvpr.thecvf.com/virtual/2025/poster/33258
