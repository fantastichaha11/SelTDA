# Compute Efficiency — Filter & Student Training

- **Date**: 2026-05-26
- **Mục đích**: Ghi nhận vì sao pipeline tốn GPU lâu + chiến lược giảm số lần train full student (§1.4-safe)

---

## Phần A — Vì sao `filter_pseudo.py` chậm?

### Tóm tắt

| Gate | Chi phí | ~51k records (ước lượng A5000) |
|------|---------|--------------------------------|
| `conf` | Đọc `gen_logprob` trong JSON | **&lt; 1 phút** |
| `itm` | CLIP ViT-B/32, **batch=1**, mở ảnh từ disk | **~30–90 phút** |
| `xcons` | BLIP-VQA load + **beam-3 generate** mỗi sample + SBERT | **~2–5 giờ** (phần lớn tổng thời gian) |

**Tổng filter (3 gate bật)**: thường **~3–6 giờ** cho 51k pseudo-QA, không phải ~45 phút như H1 protocol cũ ước tính.

### Nguyên nhân trong code

1. **`xcons` = inference VQA đầy đủ** — `BlipStudentAdapter.answer_question()` → `blip_vqa.generate(num_beams=3)` mỗi record (`filtering/adapters.py`, `models/blip_vqa.py`).
2. **`itm` = 2 forward CLIP/sample**, không batch (`filter_pseudo.py` vòng `for r in tqdm(records)`).
3. **Ảnh đọc 2 lần** — ITM và xcons mỗi gate mở lại `Image.open` (không cache embedding theo `image` path).
4. **3 pass tuần tự** — conf → itm → xcons; CLIP + BLIP có thể cùng chiếm VRAM.
5. **Scale tuyến tính** — 51k QA từ ~17k ảnh unlabeled × `questions_per_image`; gấp đôi synthetic → gấp đôi filter time.

### Cách giảm thời gian filter (không vi phạm §1.4)

| Chiến lược | Hiệu quả | Ghi chú |
|------------|----------|---------|
| **`scoring_only=true` một lần** | Tránh chạy lại xcons cho mỗi ablation threshold | Sau đó sweep `keep_top` offline bằng `apply_gates` |
| **Tắt xcons cho pilot** | −70–90% filter time | `gates.xcons.enabled=false` |
| **Chỉ C+I** (bỏ xcons) | Nhanh, vẫn có signal visual | Ablation row 4 trong H1 |
| **Batch CLIP + batch BLIP** (sửa `filter_pseudo.py`) | ~5–20× | Cải tiến code, vẫn trong scope filtering |
| **Cache image embedding** theo path | Giảm I/O + encode trùng ảnh | Nhiều QA / 1 ảnh COCO |
| **Subset pilot** (5k–10k) | Tỷ lệ tuyến tính | Chỉ để tune threshold, không báo cáo chính thức |

---

## Phần B — Vì sao train student chậm?

### Tóm tắt

Mỗi lần thử nghiệm = chạy lại **`train_vqa.py` từ đầu** (trừ `--resume`):

| Tham số (mặc định `configs/aokvqa.yaml`) | Ảnh hưởng |
|------------------------------------------|-----------|
| `max_epoch: 10` | Luôn train **đủ 10 epoch**, không early-stop theo val |
| `train_files: [train, synthetic_data]` | ~17k real + ~51k synth ≈ **68k steps/epoch** (nếu không truncate) |
| `truncate_train_dataset_to: 34000` | SelTDA paper/README — cắt còn 34k/epoch |
| `batch_size_train: 8` | GPU utilization có thể thấp |
| `image_size: 480` | ViT input lớn → chậm hơn 384 |
| Eval cuối | `k_test: 128` ranking inference trên full val |

**Ước lượng H1**: ~**10h × variant × seed** trên A5000 → matrix 7 variant × 3 seed ≈ **210h** chỉ riêng train (chiếm ~80% budget Phase 1a).

### Vì sao mỗi filter variant phải train lại?

§1.4: **không sửa** `train_vqa.py`. Khác biệt duy nhất hợp lệ là file `synthetic_data.json` sau filter → mỗi cascade variant = **một run train độc lập**.

---

## Phần C — Ideas giảm thời gian thử nghiệm (train)

### Tier 0 — Làm ngay, không cần code mới

| ID | Idea | Tiết kiệm ước tính | Cách làm |
|----|------|-------------------|----------|
| **CE-01** | **Score-once, train-many** | Filter: 8× → 1× | `scoring_only=true` → lưu JSON có đủ `scores` → script nhỏ (hoặc notebook) apply nhiều `GateThresholds` |
| **CE-02** | **Successive halving (H1)** | Train: ~70% | Seed 1 + **max_epoch=3** cho 7 variant → chỉ top-2 chạy full 10 epoch × 3 seeds |
| **CE-03** | **Dùng baseline published** | −1 run | Không retrain unfiltered nếu tin Khan Tab.4 **60.01%** — chỉ train filtered variants |
| **CE-04** | **Reuse checkpoint có sẵn** | −N runs | `cache/student_weights/checkpoint_09.pth` = unfiltered SelTDA — chỉ eval lại nếu cần so sánh công bằng |
| **CE-05** | **Tắt wandb** | ~5–10% wall | `--overrides wandb=false` |
| **CE-06** | **save_last_only** | I/O nhẹ | `--overrides save_last_only=true` — chỉ cần checkpoint cuối |

### Tier 1 — Config overrides (§1.4 OK)

| ID | Idea | Mô tả | Rủi ro |
|----|------|-------|--------|
| **CE-07** | **Proxy training** | `--overrides max_epoch=3` + `truncate_train_dataset_to=10000` để **xếp hạng** filter variant; full budget chỉ cho winner | Thứ tự variant có thể đảo |
| **CE-08** | **Epoch ladder** | Winner: train epoch 3 → 6 → 10, dừng sớm nếu val không tăng (manual) | Cần eval giữa chừng bằng `--evaluate` + checkpoint |
| **CE-09** | **Matched-N without retrain** | So sánh filtered **34k** vs random subsample **34k** từ cùng raw pool — công bằng về size | Bắt buộc cho paper; giảm số variant cần full train |
| **CE-10** | **Filter-only gate** | Chỉ train full cho **C+I+X** + **random subsample** + **unfiltered** (3 runs × 3 seeds) | Bỏ 5 ablation row — đủ cho story nếu có ablation filter scores |
| **CE-11** | **Tăng batch_size** | `batch_size_train=16` nếu VRAM đủ → ~1.3–1.5× nhanh/epoch | OOM trên 24GB |

### Tier 2 — Phương pháp luận (ít train hơn)

| ID | Idea | Mô tả |
|----|------|-------|
| **CE-12** | **Filter precision ↔ student gain** | Human-judge 200 sample: nếu correlation cao → có thể **chọn threshold** mà không train 7 lần |
| **CE-13** | **Learning curve từ 1 run** | Train 1 variant, lưu checkpoint mỗi epoch (`save_last_only=false` tạm thời) → extrapolate — **không có val trong loop** nên chỉ dùng train loss |
| **CE-14** | **Synthetic size sweep không train** | Chỉ thay `truncate` trên synthetic file size (CS-X / quantile) → 1 train với curriculum duplicate-sampling |
| **CE-15** | **Multi-GPU** | `torch.distributed.run --nproc_per_node=2` — near-linear speedup nếu đã setup |

### Tier 3 — Cải tiến code (vẫn §1.4 nếu chỉ orchestration/filter)

| ID | Idea | File |
|----|------|------|
| **CE-16** | Orchestration script: `apply_thresholds.py` đọc scored JSON → nhiều `synthetic_*.json` | `filtering/` hoặc script mới |
| **CE-17** | Batch inference trong filter | `filter_pseudo.py` |
| **CE-18** | **Eval-only mode** hàng loạt: loop `--evaluate` trên checkpoints đã có | bash orchestration |

---

## Phần D — Budget H1 đề xuất (revised)

| Phase | Cũ (protocol) | Đề xuất |
|-------|---------------|---------|
| Filter 7× riêng | ~5h | **~4h một lần** (scoring_only) |
| Train 7×3×10h | 210h | **~15h** screen (7×1×3ep) + **~60h** confirm (2×3×10h) ≈ **75h** |
| Baseline unfiltered | 30h | **0h** (published / cached ckpt) |
| **Phase 1a tổng** | ~260h | **~85–100h** (~4 ngày 1 GPU) |

Điều kiện: chấp nhận **2–3 seed** chỉ trên winner; ablation gates báo cáo bằng **filter metrics + 1 seed proxy train**.

---

## Phần E — Thứ tự chạy đề xuất trên GPU

```
1. generate synthetic_data_raw.json     (~3h, một lần)
2. filter scoring_only=true             (~4h, một lần)
3. offline: 8 threshold configs → 8 JSONs  (CPU, vài phút)
4. proxy train: top-4 variants × max_epoch=3 × 1 seed  (~12h)
5. pick winner → full train × 3 seeds × max_epoch=10     (~30h)
6. random-subsample + unfiltered eval (ckpt có sẵn)      (~2h eval)
7. human judge 100+100 on winner filter                (human time)
```

---

## Liên kết ideas catalog

- **SSR-1** (saturation curve) — train nhiều lần nếu làm naive; nên 1 train + varying synthetic JSON size qua duplicate counts
- **CS-X** — giảm N synthetic → ít epoch tương đương
- **CE-01..18** — có thể promote thành IDEA-39+ nếu cần
