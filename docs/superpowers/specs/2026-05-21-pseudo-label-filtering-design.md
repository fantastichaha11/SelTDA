# Thiết kế: Lọc chất lượng pseudo-label cho SelTDA (Improvement #1)

- **Ngày**: 2026-05-21
- **Chủ đề luận văn**: Synthetic Data cho VQA trên nền SelTDA
- **Phạm vi spec**: Improvement #1 của báo cáo định hướng — *Nâng cao chất lượng Nhãn giả (Pseudo-label Quality & Filtering)*
- **Trạng thái**: Draft chờ review

---

## 0. Bối cảnh & vấn đề

SelTDA (Khan et al., CVPR 2023) sinh pseudo-QA từ ảnh không nhãn bằng Teacher VLM (BLIP), rồi cho Student tự huấn luyện. Trong codebase hiện tại của repo `SelTDA`:

- `generate_questions.py` chỉ **parse regex** đầu ra teacher → bỏ những output không khớp template, **không** đánh giá chất lượng nội dung sample đã parse được.
- `train_vqa.py` đọc thẳng `synthetic_data.json` + `train.json` → toàn bộ pseudo-sample được dùng đều nhau.

Hệ quả: pseudo-QA noisy hoặc hallucination đi thẳng vào training, là **nguyên nhân chính** mà chính tác giả thừa nhận trong Limitations.

**Mục tiêu thiết kế**: chèn một bước lọc **giữa** hai script trên — tách bạch khỏi `train_vqa.py` — để có thể bật/tắt từng cổng lọc cho ablation và không phá vỡ tính tái lập của baseline.

## 1. Yêu cầu & ràng buộc

### 1.1 Yêu cầu chức năng

1. Tính ba score per-sample: `s_conf` (confidence teacher), `s_itm` (CLIP image–text matching), `s_xcons` (cross-consistency với student zero-shot).
2. Áp threshold độc lập từng cổng, loại sample fail bất kỳ cổng nào (Approach A — cascade).
3. Ghi `synthetic_data_filtered.json` tương thích hoàn toàn với reader hiện tại của `train_vqa.py`.
4. Sinh `filter_report.json` chứa: tổng số input, số pass/fail mỗi cổng, histogram score, danh sách 50 sample bị loại có lý do (cho qualitative analysis trong luận văn).
5. Hỗ trợ chế độ **scoring-only** (tính score không lọc) để vẽ ROC/sweep threshold offline.

### 1.2 Yêu cầu phi chức năng

- Compute budget: **1 GPU 24–48GB trên Vast.ai**. Bước nặng nhất (Gate 3 — Student forward) phải fit cùng batch 384×384.
- Tái lập: cố định seed cho mọi sampling/nucleus; lưu lại config thực tế vào report.
- Tối thiểu xâm lấn lên repo SelTDA: chỉ thêm 1 module mới (`filter_pseudo.py` + thư mục `filtering/`) + chỉnh nhỏ `generate_questions.py`. Danh sách đầy đủ files chạm vào nằm ở §1.4 và §4.

### 1.3 Out of scope (phòng scope creep)

- Iterative self-training (Improvement #3).
- Chain-of-Thought / RAG trong teacher (Improvement #2).
- Thử LLaVA / Qwen-VL backbone (Improvement #4).
- Counterfactual generation (Improvement #5).
- Soft-loss / sample weight trong training — được tách thành **Phase 2** ở §6, chỉ stub interface, không implement trong spec này.

### 1.4 Bất biến — KHÔNG SỬA CODE TRAINING (hard constraint)

> **Quy tắc bất di bất dịch của spec này**: chỉ được phép thay đổi phần **sinh dữ liệu** (generation) và thêm bước **lọc** (filtering). **Tuyệt đối không** chỉnh sửa logic huấn luyện.

**Files cấm chạm:**

- `train_vqa.py` — code train Student.
- `train_vqg.py` — code train Teacher.
- `data/vqa_dataset.py`, `data/vqg_dataset.py` — reader cho training loop.
- `models/` — **trừ** `models/blip.py` (xem mục dưới).
- `transform/`, `utils.py` ở phần liên quan tới training.
- Tất cả file dưới `vqa_eval_tools/` và các script `*_eval.py` — không đụng cho fair comparison.
- `examples/self_train_synthetic.sh`, `examples/evaluate.sh`, `examples/train_teacher.sh` — giữ nguyên.
- `configs/aokvqa.yaml`, `configs/pathvqa.yaml`, và mọi config train/eval khác — giữ nguyên.

**Files được phép chạm** (giới hạn rõ):

- `generate_questions.py` — (a) thêm 2 field optional vào `attrs` class `VQARecord` (`gen_logprob: Optional[float] = None`, `scores: Optional[Dict[str, float]] = None`); (b) gọi `model.generate(..., return_logprob=True)` để lấy log-prob và gán vào record. **Không đổi** logic generate/parse hiện tại.
- `models/blip.py` — **scope hẹp**: chỉ sửa `BLIP_Decoder.generate()`, thêm tham số `return_logprob=False` (default backward-compatible). Khi `True`, cache `image_embeds` từ lần encode duy nhất và dùng lại cho teacher-force log-prob inline — loại bỏ visual encoder forward dư thừa (~25–35% speedup) mà **không đổi** công thức raw `log p_θ(T|I)` (khớp báo cáo định hướng §1.1). **Không** sửa `BLIP_Decoder.forward()`, `blip_vqa.py`, `med.py`, `vit.py`, hay bất kỳ file nào khác dưới `models/`.
- `configs/generate_questions_*.yaml` — không bắt buộc đụng; nếu cần thêm flag log-prob, chỉ thêm key mới có default backward-compatible.

**Lưu ý**: `schemas.py` **không cần đụng**. `VQARecord` không nằm trong `schemas.py` (đó là pydantic models cho dataset annotation: `TrainingRecord`, `TestingRecord`, v.v.) — `VQARecord` là attrs class nội bộ của `generate_questions.py`. Reader `data/vqa_dataset.py` đọc JSON bằng `json.load(...)` rồi index dict trực tiếp, không validate schema → thêm field vào dict không ảnh hưởng training.

**Files mới hoàn toàn** (không ràng buộc với code cũ):

- `filter_pseudo.py`, `filtering/**`, `configs/filter_pseudo.yaml`, `examples/filter_synthetic.sh`, `tests/test_*`.

**Hệ quả thiết kế quan trọng (phải tuân thủ ở mọi quyết định kỹ thuật):**

1. Output filter **bắt buộc** dùng đúng tên file và schema mà `train_vqa.py` hiện đang đọc (`synthetic_data.json` với cấu trúc list of dict `{question_id, question, answer[], image, dataset}`). Đây là lý do trong §4.3 ta đổi tên raw thành `synthetic_data_raw.json`, không phải tên file đã lọc.
2. Mọi cơ chế cần "soft weight per sample" (Phase 2, Improvement #3) **không** được implement bằng cách sửa loss trong `train_vqa.py` ở giai đoạn này — chỉ được phép biểu diễn qua **duplicate sampling** hoặc **lọc rời rạc**. Nếu sau này phải đụng `train_vqa.py`, đó là một spec mới, không thuộc spec này.
3. So sánh "filtered vs SelTDA-original" phải fair: cùng `train_vqa.py`, cùng config, cùng seed, cùng số bước. Khác duy nhất là `synthetic_data.json`.

**Kiểm tra trước khi merge** (bắt buộc):

```bash
# diff không được phép có dòng nào dưới đây thay đổi (models/blip.py được phép)
git diff --name-only HEAD origin/main | grep -E \
  '^(train_vqa\.py|train_vqg\.py|data/|models/(?!blip\.py)|vqa_eval_tools/|.*_eval\.py|examples/self_train_synthetic\.sh|examples/evaluate\.sh|examples/train_teacher\.sh|configs/(aokvqa|pathvqa|okvqa|advqa|artvqa|rsvqa|vqa)\.yaml)' \
  && echo "VI PHẠM BẤT BIẾN" || echo "OK"
```

## 2. Quyết định kiến trúc

### 2.1 Approach đã chọn: **Sequential Cascade (A)**

Ba cổng chạy nối tiếp; sample fail bất kỳ cổng nào → loại. Lý do chọn:

- Khớp success criteria “ablation từng tầng” — bật/tắt mỗi gate độc lập tạo \(2^3 = 8\) biến thể đo đếm được.
- Compute-friendly: Gate 3 (forward Student, đắt nhất) chỉ chạy trên sample còn lại sau Gate 1+2 → giảm 2–5× số forward so với chạy song song.
- Cấu trúc unit hóa rõ; có thể thay tầng kết hợp bằng Approach B (Phase 2) mà không sửa các unit score.

### 2.2 Phase 2 extension — **Soft Score Fusion (B)**

Được spec ở **§6**, chỉ thiết kế interface để Phase 1 không phải refactor về sau:

\[
s = w_1 s_{conf} + w_2 s_{itm} + w_3 s_{xcons}, \quad \text{keep nếu } s \ge \tau_{soft}
\]

Trong khuôn khổ ràng buộc của §1.4, soft weighting **không** được hiện thực bằng cách sửa loss trong `train_vqa.py`. Thay vào đó: emit sample với tần suất tỉ lệ với `s` (duplicate sampling) hoặc lọc bằng `s ≥ τ_soft`. Việc dùng `s` như sample-weight thực sự trong loss thuộc về một spec sau (đặt nền cho Improvement #3), không thuộc spec này.

### 2.3 Backbone

- **Teacher**: BLIP-base (đúng paper SelTDA) — dùng checkpoint VQG đã train sẵn theo `examples/train_teacher.sh`.
- **Student**: BLIP-base — dùng cho cả Gate 3 zero-shot scoring và mục tiêu cuối self-train.
- **ITM**: CLIP ViT-B/32 (HuggingFace `openai/clip-vit-base-patch32`) — nhẹ, fit chung GPU với BLIP-base.

Không sử dụng LLaVA / Qwen-VL — thuộc Improvement #4, vượt scope.

### 2.4 Datasets

- **A-OKVQA** (general, data-scarce): so sánh trực tiếp với số liệu SelTDA paper.
- **PathVQA** (medical specialized): kiểm chứng đóng góp filtering trên đúng điểm yếu mà SelTDA paper thừa nhận.

Mỗi dataset chạy độc lập (không cross-train) — báo cáo riêng.

## 3. Thiết kế chi tiết các cổng

### 3.1 Gate 1 — Confidence (decoder log-likelihood)

**Tín hiệu**: mean log-probability per token mà teacher gán cho chính chuỗi `T = "Question: <q>? Answer: <a>."` mà nó vừa sample ra.

**Cách lấy** (sửa `models/blip.py` + `generate_questions.py` theo §1.4):

- Sửa `BLIP_Decoder.generate()` thêm tham số `return_logprob=False` (default backward-compat). Khi `True`, hàm cache `image_embeds` từ `visual_encoder(image)` (encode duy nhất cho cả generate lẫn scoring) và dùng lại cho 1 teacher-force forward qua `text_decoder` ngay trong cùng call. Kết quả: `(captions, mean_logprobs)`.

\[
s_{conf} = \frac{1}{|T|} \sum_{t=1}^{|T|} \log p_\theta(T_t \mid T_{<t}, I)
\]

- Cụ thể: trong `generate_questions.py`, gọi `captions, logprobs = model.generate(..., return_logprob=True)` và gán `record.gen_logprob = logprobs[i]`. Logic teacher-force nằm trong `BLIP_Decoder.generate()` — không re-encode ảnh lần 2.

- **Implementation note**: phiên bản đầu (Task 12) gọi `_compute_mean_logprob` riêng → visual encoder chạy 2 lần/batch. Phiên bản hiện tại cache `image_embeds` → tiết kiệm ~25–35% wallclock; công thức raw log-likelihood **không đổi** (khớp báo cáo định hướng: "log-likelihood của decoder").

**Chuẩn hoá**: \(s_{conf}\) là log-prob âm; chuẩn hoá min-max trên toàn bộ pool sinh được:

\[
\tilde s_{conf} = \frac{s_{conf} - \min}{\max - \min} \in [0, 1]
\]

**Threshold**: dùng **quantile-based** (giữ top-\(q_1\) phần trăm) thay vì giá trị tuyệt đối — robust với phân phối log-prob khác nhau giữa dataset.

- Khoảng quét trong ablation: \(q_1 \in \{1.0, 0.9, 0.75, 0.5\}\) (1.0 = không lọc).

**Test unit**: trong `tests/test_scorers.py` — mock 1 record có log-prob biết trước, assert `score_confidence(...)` trả đúng giá trị.

### 3.2 Gate 2 — CLIP image-text matching

**Tín hiệu**: cosine similarity giữa embedding ảnh CLIP và embedding chuỗi `f"{question} {answer}"`.

\[
s_{itm} = \cos\!\big(\text{CLIP}_v(I),\ \text{CLIP}_t(\text{"Q? A."})\big) \in [-1, 1]
\]

**Lý do format `"Q? A."`** (thay vì chỉ A): CLIP train trên caption tự nhiên; gắn câu hỏi vào giúp phân biệt được trường hợp answer chung chung (`"yes"`) đúng nội dung ảnh nhưng không trả lời đúng câu hỏi cụ thể.

**Chuẩn hoá**: rescale về \([0,1]\) bằng \((s+1)/2\).

**Threshold**: quantile \(q_2 \in \{1.0, 0.9, 0.75, 0.5\}\).

**Hiệu năng**: CLIP ViT-B/32 batch 64 ảnh trên RTX A5000 ~ 50 ms / batch; tổng cost cho ~100k pseudo-sample < 5 phút → không phải nút thắt.

**Edge case**: yes/no question — CLIP không phân biệt được polarity tốt. Mitigate: log riêng metric cho yes/no và non-yes/no trong `filter_report.json` để xét sau.

### 3.3 Gate 3 — Cross-consistency (Student zero-shot)

**Tín hiệu**: cho Student (BLIP pretrained, **chưa fine-tune trên synthetic**) trả lời câu hỏi pseudo, đo độ khớp với answer pseudo.

\[
A' = \text{Student}_{zs}(I, Q), \quad s_{xcons} = \text{match}(A', A)
\]

**Hàm match**:

- Yes/no: exact match sau lowercase.
- Open-ended: dùng **VQA soft accuracy** chuẩn (chia điểm theo số human-annotator trùng) **không áp dụng được** vì chỉ có 1 answer pseudo → fallback:
  - Lemmatize + lowercase + strip punct → exact match (score 1.0 / 0.0); **hoặc**
  - Cosine similarity của Sentence-BERT embedding `all-MiniLM-L6-v2` giữa `A'` và `A` (chấp nhận paraphrase: "a dog" ≈ "dog").

Spec mặc định **dùng cả hai**: \(s_{xcons} = \max(\text{exact}, \text{sbert\_cos})\). Lý do: exact bắt được case dễ; SBERT bắt paraphrase; lấy `max` tránh phạt oan.

**Lưu ý quan trọng** (đặt thẳng vào spec để tránh sai lầm khi implement): Student dùng ở Gate 3 phải là checkpoint **pretrained gốc**, **không phải** student đã train trên `synthetic_data.json` — nếu không sẽ rò rỉ thông tin (Student đã thấy data → luôn match cao → cổng vô hiệu).

**Threshold**: \(q_3 \in \{1.0, 0.9, 0.75, 0.5\}\). Quét riêng cho yes/no và open-ended vì phân phối score khác hẳn.

**Hiệu năng**: forward 1 lần BLIP-base 384² ≈ 25 ms / sample trên A5000. Trên ~100k pseudo còn lại sau G1+G2 (giả sử 60% pass) → ~25 phút. Acceptable.

### 3.4 Logic kết hợp (Cascade)

```python
def apply_gates(record, thresholds) -> tuple[bool, str]:
    if record.s_conf < thresholds.tau_conf:
        return False, "conf"
    if record.s_itm < thresholds.tau_itm:
        return False, "itm"
    if record.s_xcons < thresholds.tau_xcons:
        return False, "xcons"
    return True, "kept"
```

Mỗi gate có thể tắt bằng cách đặt threshold = \(-\infty\) (hoặc quantile 1.0). 8 biến thể ablation:

| Biến thể | Conf | ITM | X-cons |
|---|---|---|---|
| `none` (= SelTDA baseline) | off | off | off |
| `C` | on | off | off |
| `I` | off | on | off |
| `X` | off | off | on |
| `C+I` | on | on | off |
| `C+X` | on | off | on |
| `I+X` | off | on | on |
| `C+I+X` (full) | on | on | on |

## 4. Cấu trúc code & integration

### 4.1 Files mới

```
SelTDA/
├── filter_pseudo.py              # mới — entrypoint
├── configs/
│   └── filter_pseudo.yaml        # mới — config thresholds, paths
├── filtering/                    # mới — module gốc
│   ├── __init__.py
│   ├── scorers.py                # score_confidence, score_clip_itm, score_xcons
│   ├── matchers.py               # exact_match, sbert_match
│   ├── gates.py                  # apply_gates, apply_soft_fusion (stub Phase 2)
│   └── report.py                 # build_filter_report
├── examples/
│   └── filter_synthetic.sh       # mới — kết nối generate → filter → train
└── tests/
    ├── test_scorers.py           # mới
    ├── test_matchers.py          # mới
    └── test_gates.py             # mới
```

### 4.2 Files chỉnh nhỏ

- `generate_questions.py`:
  - Thêm 2 trường optional vào `attrs` class `VQARecord` (line ~72):
    ```python
    gen_logprob: Optional[float] = None
    scores: Optional[Dict[str, float]] = None  # {"conf":.., "itm":.., "xcons":..}
    ```
  - Gọi `captions, logprobs = model.generate(..., return_logprob=True)` và gán `record.gen_logprob`.
  - Không động vào regex parse, dataset loading, hay control flow.

- `models/blip.py` (scope hẹp — xem §1.4):
  - Thêm `return_logprob=False` vào `BLIP_Decoder.generate()`.
  - Cache `image_embeds` và teacher-force inline khi `return_logprob=True`.

→ Toàn bộ thay đổi backward-compatible: file JSON mới có thêm field, `data/vqa_dataset.py` (cấm chạm) đọc bằng `json.load` + dict key access → field thừa bị bỏ qua.

### 4.3 Pipeline shell

```bash
# bước 1 (cũ, sửa nhỏ)
bash examples/generate_synthetic_data.sh
# → datasets/aokvqa/synthetic_data_raw.json

# bước 2 (mới)
bash examples/filter_synthetic.sh aokvqa  # hoặc pathvqa
# đọc raw → ghi datasets/aokvqa/synthetic_data.json (sau filter)
# ghi datasets/aokvqa/filter_report.json

# bước 3 (cũ, không sửa)
bash examples/self_train_synthetic.sh
```

Lý do giữ tên `synthetic_data.json` cho output đã lọc: `train_vqa.py` đọc đúng tên này theo config — không cần đụng reader. Raw đổi tên thành `synthetic_data_raw.json`.

### 4.4 Config schema (`configs/filter_pseudo.yaml`)

```yaml
input: datasets/aokvqa/synthetic_data_raw.json
output: datasets/aokvqa/synthetic_data.json
report: datasets/aokvqa/filter_report.json

gates:
  conf:
    enabled: true
    mode: quantile        # quantile | absolute
    keep_top: 0.75
  itm:
    enabled: true
    clip_model: "openai/clip-vit-base-patch32"
    keep_top: 0.75
  xcons:
    enabled: true
    student_ckpt: "cache/blip_pretrained.pth"  # KHÔNG dùng student đã train
    matcher: max_exact_sbert
    sbert_model: "sentence-transformers/all-MiniLM-L6-v2"
    keep_top: 0.75
    image_size: 384

scoring_only: false       # nếu true: chỉ tính score, không loại
seed: 42
```

## 5. Thiết kế thí nghiệm

### 5.1 Baseline

1. **No-synthetic**: train Student chỉ trên `train.json` thật. Sàn dưới.
2. **SelTDA-original**: reproduce con số paper trên A-OKVQA và PathVQA — train trên `train.json` ∪ `synthetic_data_raw.json` (không lọc).
3. **Filtered (chúng ta)**: train trên `train.json` ∪ `synthetic_data_filtered.json` cho từng biến thể ablation §3.4.

### 5.2 Ablation chính

| # | Biến thể | A-OKVQA acc | PathVQA acc | # pseudo giữ lại |
|---|---|---|---|---|
| 1 | No-synthetic | | | 0 |
| 2 | `none` (SelTDA) | | | N |
| 3 | `C` | | | |
| 4 | `I` | | | |
| 5 | `X` | | | |
| 6 | `C+I` | | | |
| 7 | `C+X` | | | |
| 8 | `I+X` | | | |
| 9 | `C+I+X` (full) | | | |

Ở mỗi biến thể "on", dùng `keep_top = 0.75` (giữ 75% top theo gate đó).

### 5.3 Threshold sweep (chỉ cho biến thể full)

Sweep \(\{0.5, 0.75, 0.9, 1.0\}\) cho **một** trong 3 gate cùng lúc, hai gate còn lại giữ 0.75. Mục đích: phát hiện gate nào nhạy với threshold nhất → đóng góp chính.

### 5.4 Phân tích chất lượng pseudo (qualitative)

Lấy ngẫu nhiên 100 sample bị loại / 100 sample giữ lại cho mỗi dataset. Human-judge (chính bạn) gán nhãn `correct/incorrect/ambiguous`. Tính precision/recall của filter so với judgment.

### 5.5 Metrics

- **VQA accuracy** theo công thức chuẩn của từng dataset (sẵn trong `aokvqa_mc_eval.ipynb`, `pathvqa_eval.py`).
- **Số pseudo giữ lại** (absolute & %).
- **Filter precision/recall** so với human judgment (§5.4).
- **Histogram score** từng gate (đưa vào appendix luận văn).

### 5.6 Compute budget ước lượng (Vast.ai 1×A5000 48GB)

| Bước | Thời gian | Ghi chú |
|---|---|---|
| Reproduce baseline SelTDA (A-OKVQA) | ~12h | đã có config sẵn |
| Generate raw synthetic | ~3h | có sẵn trong paper hint |
| Filter (toàn pipeline 3 gate) | ~45 phút | 1 lần / dataset |
| Student train (full filter) | ~10h / dataset | giống baseline |
| Toàn bộ ablation (8 biến thể × 2 dataset) | ~7 ngày | có thể song song hóa nếu thuê 2 instance |

→ **Feasibility**: nằm trong khả năng 1–2 tuần thuê Vast.ai.

## 6. Phase 2 — Soft Score Fusion (interface stub)

Không implement trong spec này, chỉ chừa chỗ.

```python
# filtering/gates.py
def apply_soft_fusion(record, weights, tau) -> tuple[bool, float]:
    """Phase 2 — chưa implement, chỉ stub."""
    raise NotImplementedError
```

Khi implement Phase 2, **không** thay đổi 3 hàm score; chỉ thay tầng `apply_gates`. **Theo §1.4, không được sửa `train_vqa.py`** — nếu muốn thực sự đưa `s` vào loss như sample-weight (Improvement #3), đó phải là một spec riêng được phê duyệt sau, không thuộc phạm vi Phase 2 ở đây. Phase 2 chỉ cho phép biến `s` thành (a) ngưỡng cut mềm hoặc (b) duplicate sampling trên cùng `synthetic_data.json` mà thôi.

## 7. Rủi ro & câu hỏi mở

1. **BLIP decoder có export log-prob trực tiếp không?**
   - **Đã giải quyết**: `BLIP_Decoder.generate(return_logprob=True)` cache `image_embeds` và trả `(captions, mean_logprobs)` — raw teacher-force log-likelihood, không cần visual encoder forward thứ 2.
2. **CLIP ViT-B/32 quá yếu cho PathVQA (ảnh y khoa)?**
   - Mitigation: chuẩn bị fallback PubMedCLIP / BiomedCLIP. Ghi nhận trong limitations nếu B/32 fail.
3. **Cross-consistency mâu thuẫn với mục tiêu sinh dữ liệu mới**:
   - Nếu Student zero-shot đã biết câu trả lời, sample đó **dễ** → filter có thể vô tình loại sample khó-nhưng-đúng. Mitigate: kiểm tra trong qualitative analysis (§5.4); cân nhắc dùng \(s_{xcons}\) như **soft weight** trong Phase 2 thay vì hard cut.
4. **Threshold quantile không robust nếu phân phối score lệch nặng**:
   - Mitigation: scoring-only mode (§1.1.5) chạy 1 lần trước để vẽ phân phối; chọn quantile sau khi quan sát.
5. **Số seed nhỏ**:
   - Mitigation: chạy mỗi cấu hình với 3 seed nếu budget cho phép; ít nhất biến thể `none` và `C+I+X` phải có 3 seed.

## 8. Tiêu chí hoàn thành (Definition of Done) cho Phase 1

- [ ] `filter_pseudo.py` chạy được end-to-end trên A-OKVQA, sinh ra `synthetic_data.json` tương thích `train_vqa.py`.
- [ ] Bảng ablation §5.2 có 9 dòng filled trên cả A-OKVQA và PathVQA.
- [ ] `filter_report.json` đầy đủ field (counts, histograms, sample loại).
- [ ] Unit tests xanh cho `scorers`, `matchers`, `gates`.
- [ ] Section trong luận văn viết được dựa trên kết quả: methodology + 1 bảng chính + ablation + qualitative samples.

## 9. Sau Phase 1

Khi Phase 1 cho kết quả khả quan, hai hướng mở rộng kế tiếp (theo báo cáo định hướng):

- **Phase 2** — Soft Fusion (§6). Lưu ý: vẫn tuân thủ bất biến §1.4, không đụng `train_vqa.py`.
- **Phase 3** — RAG + CoT cho teacher trên PathVQA (Improvement #2) — tận dụng đúng pipeline filter đã có để đo hiệu quả CoT.
- **Soft loss / sample-weight trong training (Improvement #3)** — nếu muốn thực sự sửa loss, bắt buộc phải là spec **riêng**, đi qua review riêng, vì sẽ phá vỡ bất biến hiện tại.

---

*Spec này khớp với báo cáo định hướng nghiên cứu (mục Improvement #1) và codebase SelTDA hiện tại trong workspace.*
