# Hướng dẫn fine-tune CLIP-FlanT5-XL trên A-OKVQA

Tài liệu này mô tả cách **train (fine-tune) CLIP-FlanT5-XL** trên dữ liệu gốc của SelTDA (A-OKVQA), dựa trên repo upstream [linzhiqiu/CLIP-FlanT5](https://github.com/linzhiqiu/CLIP-FlanT5). Sau khi train xong, checkpoint có thể dùng làm **VQAScore judge** trong GRPO teacher (`filtering/vqascore_adapter.py`).

**Khác với GRPO teacher hiện tại:** pipeline SelTDA dùng **frozen** `zhiqiulin/clip-flant5-xl` làm reward. Fine-tune XL tạo judge **domain-adapted** cho A-OKVQA — đây là side project / ablation, không thay thế `train_vqa.py` hay `train_vqg.py`.

---

## 1. CLIP-FlanT5-XL là gì?

| Thành phần | Chi tiết |
|------------|----------|
| **Backbone text** | [google/flan-t5-xl](https://huggingface.co/google/flan-t5-xl) (3B params) |
| **Vision encoder** | CLIP ViT-L/14 @ 336px |
| **Mục đích** | VQA + VQAScore (Lin et al., ECCV 2024) — đánh giá khớp ảnh–(Q,A) qua template yes/no |
| **Checkpoint published** | [zhiqiulin/clip-flant5-xl](https://huggingface.co/zhiqiulin/clip-flant5-xl) |
| **VRAM inference** | ~7 GB load + ~1–3 GB forward (batch=1) |

**Hai stage training (upstream):**

1. **Stage-1 (alignment):** Nối CLIP → FlanT5 bằng MLP projector trên LAION-CC-SBU 558K. **Không cần train lại** — dùng `zhiqiulin/clip-flant5-xl-stage-1`.
2. **Stage-2 (VQA fine-tune):** Fine-tune toàn model trên mix VQA (~665K trong paper). **Chúng ta chỉ dùng A-OKVQA** (~18K) thay cho full mix.

---

## 2. Yêu cầu phần cứng & disk

### Train (stage-2)

| Cấu hình | Khả thi? | Ghi chú |
|----------|----------|---------|
| **8× A100 40G** | Khuyến nghị | Khớp script upstream (`clip-flant5-xl.sh`) |
| **4× A100 40G** | Có thể | Giảm `PER_DEVICE_BS` và tăng `GRAD_ACCUM` |
| **1× L4 23G** | **Không** | Chỉ đủ cho **inference** (reward GRPO), không train XL |

Paper ước tính stage-2 XL: ~60 giờ trên 8× A100 40G (full LLaVA mix). Với chỉ A-OKVQA (~18K samples, 3 epoch) thường **vài giờ đến ~1 ngày** tùy GPU.

### Disk

| Artifact | Kích thước ước tính |
|----------|---------------------|
| COCO images (đã có trong SelTDA) | ~20 GB |
| `clip-flant5-xl` weights (HF) | ~12 GB |
| `clip-flant5-xl-stage-1` projector | ~100 MB |
| Output checkpoint sau train | ~12 GB |
| JSON converted | ~50–100 MB |

Cần **≥ 50 GB trống** trên instance train (Vast.ai).

### Inference trên máy L4 (sau train)

Checkpoint fine-tuned load qua `reward.vqascore_checkpoint` — cùng VRAM ~12–16 GB như XL gốc khi chạy reward GRPO.

---

## 3. Tổng quan pipeline

```
A-OKVQA train.json + val.json
        │
        ▼
convert_aokvqa_to_clip_flant5.py  →  aokvqa_vqa.json (LLaVA format)
        │
        ▼
CLIP-FlanT5 repo (external/)  +  stage-1 projector
        │
        ▼
deepspeed t5_train_mem.py  →  checkpoints/clip-flant5-xl-aokvqa/
        │
        ▼
SelTDA GRPO: reward.vqascore_checkpoint=<output_dir>
```

---

## 4. Chuẩn bị dữ liệu SelTDA

### 4.1 Dataset cần có

Chạy từ repo SelTDA (đã có nếu đã `bash dataset.sh` + `convert_aokvqa.py`):

```
datasets/aokvqa/train.json      # ~17,056 samples
datasets/aokvqa/val.json        # ~1,145 samples
datasets/coco2017/              # ảnh COCO (flat hoặc train2017/)
```

### 4.2 Convert sang LLaVA JSON

**Mode VQA** (khuyến nghị cho stage-2 — học Q→A):

```bash
cd SelTDA
export PYTHONNOUSERSITE=1

python scripts/convert_aokvqa_to_clip_flant5.py \
  --config configs/convert_clip_flant5.yaml
```

Output mặc định: `datasets/clip_flant5/aokvqa_vqa.json` (~18,201 samples).

Mỗi record có dạng:

```json
{
  "id": "22MexNkBPpdZGX6sxbxVBH",
  "image": "coco/train2017/000000299207.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nWhat is the man by the bags awaiting?"},
    {"from": "gpt", "value": "ride"}
  ]
}
```

**Mode VQAScore** (optional — học template yes/no của judge):

```bash
python scripts/convert_aokvqa_to_clip_flant5.py \
  --config configs/convert_clip_flant5_vqascore.yaml
```

→ `datasets/clip_flant5/aokvqa_vqascore.json` (có thể gồm `synthetic_data.json` đã filter).

**Override tùy chỉnh:**

```bash
python scripts/convert_aokvqa_to_clip_flant5.py \
  --config configs/convert_clip_flant5.yaml \
  --overrides \
    splits=[train] \
    synthetic=datasets/aokvqa/synthetic_data.json \
    output=datasets/clip_flant5/aokvqa_train_plus_synth.json
```

### 4.3 Cấu trúc ảnh cho CLIP-FlanT5

Training script join `image_folder` + `image` field. Script setup tạo symlink:

```
external/CLIP-FlanT5/playground/data/coco/train2017/
    → <SelTDA>/datasets/coco2017/
```

Ảnh trong JSON dùng path `coco/train2017/<filename>.jpg` — converter đã set `image_prefix: coco/train2017`.

---

## 5. Setup trên Vast.ai (hoặc server multi-GPU)

### 5.1 Thuê instance

Trên [Vast.ai](https://vast.ai), chọn template:

- **GPU:** 4–8× A100 40GB (hoặc tương đương ≥ 40GB/GPU)
- **Disk:** ≥ 100 GB
- **Image:** PyTorch 2.x + CUDA 12.x

### 5.2 Clone & setup một lần

```bash
# Trên instance — giả sử SelTDA đã có tại ~/SelTDA
cd ~/SelTDA
export PYTHONNOUSERSITE=1

bash scripts/clip_flant5/setup_vast_aokvqa.sh
```

Script này sẽ:

1. Clone `external/CLIP-FlanT5` từ GitHub
2. Symlink COCO images vào `playground/data/coco/train2017`
3. Chạy converter → `datasets/clip_flant5/aokvqa_vqa.json`
4. Tải stage-1 projector `zhiqiulin/clip-flant5-xl-stage-1/mm_projector.bin`
5. Cài `deepspeed`, `wandb`, `accelerate` (nếu chưa có)

**Biến môi trường tùy chọn:**

```bash
export SELTDA=~/SelTDA
export CLIP_ROOT=~/SelTDA/external/CLIP-FlanT5
export DATA_JSON=~/SelTDA/datasets/clip_flant5/aokvqa_vqa.json
export OUTPUT_DIR=~/SelTDA/external/CLIP-FlanT5/checkpoints/clip-flant5-xl-aokvqa
```

### 5.3 Cài dependency CLIP-FlanT5 (nếu setup báo lỗi)

```bash
cd external/CLIP-FlanT5
pip install -e .
pip install deepspeed wandb accelerate
# Theo README upstream nếu thiếu package:
pip install timm einops sentencepiece protobuf
```

Đọc thêm: [CLIP-FlanT5 README — Finetune](https://github.com/linzhiqiu/CLIP-FlanT5#finetune-training-for-vqa).

---

## 6. Chạy training

### 6.1 Lệnh mặc định (8 GPU, 3 epoch)

```bash
cd ~/SelTDA
bash scripts/clip_flant5/train_xl_aokvqa.sh
```

### 6.2 Tùy chỉnh hyperparameter

```bash
NUM_GPUS=4 \
PER_DEVICE_BS=4 \
GRAD_ACCUM=4 \
EPOCHS=3 \
LR=2e-5 \
OUTPUT_DIR=~/SelTDA/external/CLIP-FlanT5/checkpoints/clip-flant5-xl-aokvqa \
bash scripts/clip_flant5/train_xl_aokvqa.sh
```

| Biến | Mặc định | Ý nghĩa |
|------|----------|---------|
| `NUM_GPUS` | 8 | Số GPU DeepSpeed |
| `PER_DEVICE_BS` | 6 | Batch size / GPU |
| `GRAD_ACCUM` | 2 | Gradient accumulation |
| `EPOCHS` | 3 | Số epoch trên A-OKVQA |
| `LR` | 2e-5 | Learning rate (khớp upstream) |

**Global batch size** = `NUM_GPUS × PER_DEVICE_BS × GRAD_ACCUM`  
Mặc định: 8 × 6 × 2 = **96**.

### 6.3 Lệnh train thủ công (tham khảo)

Tương đương script, chạy từ `external/CLIP-FlanT5/`:

```bash
deepspeed --num_gpus=8 llava/train/t5_train_mem.py \
  --deepspeed ./scripts/zero3.json \
  --model_name_or_path google/flan-t5-xl \
  --version t5_v1 \
  --data_path /path/to/aokvqa_vqa.json \
  --image_folder ./playground/data \
  --vision_tower openai/clip-vit-large-patch14-336 \
  --pretrain_mm_mlp_adapter ./checkpoints/clip-flant5-xl-stage-1/mm_projector.bin \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --image_aspect_ratio pad \
  --bf16 True \
  --output_dir ./checkpoints/clip-flant5-xl-aokvqa \
  --num_train_epochs 3 \
  --per_device_train_batch_size 6 \
  --gradient_accumulation_steps 2 \
  --learning_rate 2e-5 \
  --gradient_checkpointing True \
  --lazy_preprocess True
```

**Quan trọng:** `--version t5_v1` phải khớp conversation template của FlanT5 (không dùng `v1` của LLaMA).

---

## 7. Sau khi train — dùng trong SelTDA

### 7.1 Checkpoint output

Thư mục output (ví dụ `checkpoints/clip-flant5-xl-aokvqa/`) chứa weights HuggingFace format (`config.json`, `pytorch_model.bin` hoặc shards).

Copy về máy L4 nếu train trên Vast:

```bash
rsync -avz vast:~/SelTDA/external/CLIP-FlanT5/checkpoints/clip-flant5-xl-aokvqa/ \
  ~/SelTDA/cache/judges/clip-flant5-xl-aokvqa/
```

### 7.2 Load làm VQAScore reward

Trong config GRPO (`configs/rl_teacher_vqascore_conf.yaml` hoặc override):

```yaml
reward:
  vqascore_model: clip-flant5-xl
  vqascore_checkpoint: cache/judges/clip-flant5-xl-aokvqa
  vqascore_device: cuda
```

Hoặc CLI:

```bash
python train_vqg_rl.py \
  --config configs/rl_teacher_vqascore_conf.yaml \
  --overrides \
    reward.vqascore_checkpoint=cache/judges/clip-flant5-xl-aokvqa
```

`VQAScoreAdapter` sẽ load weights local thay vì `zhiqiulin/clip-flant5-xl` trên HuggingFace.

### 7.3 Lưu ý khoa học

- Fine-tune judge rồi dùng làm reward → có nguy cơ **reward hacking** (teacher tối ưu theo judge đã adapt). Spec C-RT+ khuyến nghị **frozen off-the-shelf** judge; dùng custom checkpoint như **ablation** và so sánh với frozen baseline.
- **Audit** (`filtering/audit.py`) vẫn dùng ITM-Large + BLIP-VQA — độc lập với VQAScore reward.

---

## 8. Troubleshooting

### OOM khi train

- Giảm `PER_DEVICE_BS` (4 → 2)
- Tăng `GRAD_ACCUM` để giữ global batch
- Giảm `NUM_GPUS` nhưng tăng `GRAD_ACCUM` tương ứng
- Đảm bảo `--gradient_checkpointing True` (đã bật trong script)

### `FileNotFoundError` ảnh

- Kiểm tra symlink: `playground/data/coco/train2017/<file>.jpg` tồn tại
- Chạy lại converter với `skip_missing_images: true` (mặc định) — log báo số skipped

### DeepSpeed / NCCL lỗi multi-GPU

```bash
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

### OOM khi chạy reward trên L4

- Giữ `vqascore_model: clip-flant5-xl` (không dùng xxl)
- Nếu vẫn OOM: `reward.vqascore_device=cpu` (chậm hơn nhiều)

### Stage-1 projector sai

MODEL_ZOO upstream nhấn mạnh: **phải** dùng projector khớp base LLM (FlanT5-XL) và CLIP-L-336. Không mix XL projector với XXL LLM.

---

## 9. File tham chiếu trong repo

| File | Vai trò |
|------|---------|
| `scripts/convert_aokvqa_to_clip_flant5.py` | Converter A-OKVQA → LLaVA JSON |
| `configs/convert_clip_flant5.yaml` | Config mode `vqa` |
| `configs/convert_clip_flant5_vqascore.yaml` | Config mode `vqascore` |
| `scripts/clip_flant5/setup_vast_aokvqa.sh` | Setup clone + data + stage-1 |
| `scripts/clip_flant5/train_xl_aokvqa.sh` | Launch DeepSpeed train |
| `filtering/vqascore_adapter.py` | Load frozen hoặc custom checkpoint |
| `tests/test_convert_clip_flant5.py` | Unit tests converter |

---

## 10. Tài liệu ngoài

- [CLIP-FlanT5 GitHub](https://github.com/linzhiqiu/CLIP-FlanT5)
- [MODEL_ZOO](https://github.com/linzhiqiu/CLIP-FlanT5/blob/master/docs/MODEL_ZOO.md)
- [VQAScore paper](https://linzhiqiu.github.io/papers/vqascore/)
- [t2v_metrics](https://github.com/linzhiqiu/t2v_metrics) — inference / scoring API
- SelTDA spec C-RT+: `docs/superpowers/specs/2026-06-04-reasoning-teacher-rl-cot-design.md` §2.2

---

## 11. Checklist nhanh

- [ ] `datasets/aokvqa/train.json` + `val.json` có sẵn
- [ ] `datasets/coco2017/` có ảnh
- [ ] Chạy converter → `datasets/clip_flant5/aokvqa_vqa.json`
- [ ] Thuê Vast.ai 4–8× A100 40G
- [ ] `bash scripts/clip_flant5/setup_vast_aokvqa.sh`
- [ ] `bash scripts/clip_flant5/train_xl_aokvqa.sh`
- [ ] Copy checkpoint về `cache/judges/clip-flant5-xl-aokvqa/`
- [ ] (Optional) GRPO với `reward.vqascore_checkpoint=...`
