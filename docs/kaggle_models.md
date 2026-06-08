# Kaggle Models — SelTDA Teacher & Student

Tài liệu mô tả hai **Kaggle Model** cho SelTDA (`phong2004`). Mỗi checkpoint là một **variant** (model instance riêng), không phải nhiều version dưới một instance `default`.

## Trạng thái (2026-06-03)

**Đã tạo** qua Legacy API (`username`: `phong2004`):

| Model | URL | Kaggle ID |
|-------|-----|-----------|
| Teacher (VQG) | https://www.kaggle.com/models/phong2004/seltda-teacher-vqg | 691760 |
| Student (VQA) | https://www.kaggle.com/models/phong2004/seltda-student-vqa | 691761 |

Tạo lại (nếu cần): `bash scripts/create_kaggle_models.sh` — `ownerSlug` trong metadata phải khớp `username` trong `~/.kaggle/kaggle.json`.

**Cấu trúc Kaggle**

| Khái niệm | Ví dụ |
|-----------|--------|
| Model | `phong2004/seltda-teacher-vqg` |
| Framework | `PyTorch` |
| **Variant** (instance slug) | `published-epoch04`, `iterative-round1-final`, … |
| URL đầy đủ | `phong2004/seltda-teacher-vqg/PyTorch/published-epoch04` |

**Upload variants (khuyến nghị):**
```bash
# Tất cả variants (~30GB+, chạy lâu)
bash scripts/upload_kaggle_model_variants.sh

# Một variant
ONLY=published-epoch04 MODEL=teacher bash scripts/upload_kaggle_model_variants.sh
ONLY=published-epoch09 MODEL=student bash scripts/upload_kaggle_model_variants.sh

# Bỏ qua variant đã có trên Kaggle
SKIP_EXISTING=1 bash scripts/upload_kaggle_model_variants.sh

bash scripts/upload_kaggle_model_variants.sh --dry-run
```

Manifest: `scripts/kaggle_model_versions.manifest` (cột `version_id` = **variant slug**).

**Không dùng:** `scripts/upload_kaggle_model_versions.sh` — gắn nhiều file vào instance `default` dưới dạng *versions* (khác ý “mỗi checkpoint một variant”). Instance `default` có thể xóa trên UI nếu chỉ cần variant.

---

## 1. Teacher — `seltda-teacher-vqg`

| Trường | Giá trị |
|--------|---------|
| **URL** | https://www.kaggle.com/models/phong2004/seltda-teacher-vqg |
| **slug** | `seltda-teacher-vqg` |
| **title** | SelTDA Teacher (VQG) |
| **subtitle** | BLIP visual question generation — A-OKVQA |
| **Variants** | `…/PyTorch/published-epoch04`, `…/iterative-round1-final`, … (xem bảng) |

**Mô tả ngắn:** Teacher VQG (BLIP `blip_decoder`, ViT-base) sinh cặp (question, answer) từ ảnh COCO unlabeled; config `configs/aokvqg.yaml`, generation `configs/generate_questions_aokvqa.yaml`.

### Variants (mỗi dòng = một instance slug trên Kaggle)

| Variant ID (instance slug) | File local | Kích thước ~ | Ghi chú |
|----------------------|------------|--------------|---------|
| `published-epoch04` | `cache/teacher_weights/checkpoint_04.pth` | 2.5 GB | Bản chính thức SelTDA (CVPR 2023), epoch 4. Tải qua `dataset.sh` / Google Drive. Dùng cho generate mặc định. |
| `iterative-round1-final` | `orchestration/state/teacher_1.pth` | 2.5 GB | Teacher sau RL round 1 (epoch 2, `mean_reward≈1.62`). Pipeline: `run_rl_round1_student.sh`. |
| `iterative-round1-best` | `orchestration/state/teacher_1_best.pth` | 2.5 GB | Checkpoint tốt nhất validation trong round 1. |
| `iterative-round1-epoch0` | `orchestration/state/teacher_1_epoch0.pth` | 2.5 GB | Sau epoch 0 (`mean_reward≈1.24`, 80 steps). |
| `iterative-round1-epoch1` | `orchestration/state/teacher_1_epoch1.pth` | 2.5 GB | Sau epoch 1 (`mean_reward≈1.59`, judge≈0.36). |
| `iterative-round1-epoch2` | `orchestration/state/teacher_1_epoch2.pth` | 2.5 GB | Sau epoch 2 (trùng final trước khi copy `teacher_1.pth`). |

**Usage:**
```bash
python generate_questions.py \
  --config configs/generate_questions_aokvqa.yaml \
  --overrides pretrained=<path-to-checkpoint.pth>
```

---

## 2. Student — `seltda-student-vqa`

| Trường | Giá trị |
|--------|---------|
| **URL** | https://www.kaggle.com/models/phong2004/seltda-student-vqa |
| **slug** | `seltda-student-vqa` |
| **title** | SelTDA Student (VQA) |
| **subtitle** | BLIP visual question answering — A-OKVQA |
| **Variants** | `…/PyTorch/published-epoch09`, `…/rl-base-filter`, … (xem bảng) |

**Mô tả ngắn:** Student VQA (BLIP `blip_vqa`, ViT-base, rank inference); train `configs/aokvqa.yaml` với `train_files=[train,synthetic_data]`; dùng cho filter (gate `xcons`) và eval.

### Variants (mỗi dòng = một instance slug trên Kaggle)

| Variant ID (instance slug) | File local | Kích thước ~ | Ghi chú |
|----------------------|------------|--------------|---------|
| `published-epoch09` | `cache/student_weights/checkpoint_09.pth` | 4.0 GB | Bản chính thức SelTDA self-trained, epoch 9. `dataset.sh`. |
| `rl-base-filter` | `cache/student_weights/student_base_rl.pth` | 4.0 GB | Student cơ sở cho gate `xcons` trong RL round 1 (`run_rl_round1_student.sh`). |
| `rl-round1-epoch00` | `cache/rl_round1_student/checkpoint_00.pth` | 4.0 GB | RL round 1 student, epoch 0 (train + synthetic raw, không filter_pseudo). |
| `rl-round1-epoch01` | `cache/rl_round1_student/checkpoint_01.pth` | 4.0 GB | RL round 1 student, epoch 1 (mới nhất tại thời điểm ghi doc). |

**Usage:**
```bash
python -m torch.distributed.run --nproc_per_node=1 train_vqa.py \
  --config configs/aokvqa.yaml \
  --overrides pretrained=<path> "train_files=[train,synthetic_data]" wandb=false

python -m torch.distributed.run --nproc_per_node=1 train_vqa.py \
  --evaluate --config configs/aokvqa.yaml \
  --overrides pretrained=<path> use_validation_set_as_test_set=true wandb=false
```

---

## MCP payloads (sau khi có `models.create`)

### Teacher
```json
{
  "ownerSlug": "phong2004",
  "slug": "seltda-teacher-vqg",
  "title": "SelTDA Teacher (VQG)",
  "subtitle": "BLIP visual question generation — A-OKVQA",
  "description": "<xem bảng versions ở trên + usage trong repo CLAUDE.md>",
  "isPrivate": false,
  "provenanceSources": "https://github.com/salesforce/BLIP, SelTDA CVPR 2023"
}
```

### Student
```json
{
  "ownerSlug": "phong2004",
  "slug": "seltda-student-vqa",
  "title": "SelTDA Student (VQA)",
  "subtitle": "BLIP visual question answering — A-OKVQA",
  "description": "<xem bảng versions ở trên + usage trong repo CLAUDE.md>",
  "isPrivate": false,
  "provenanceSources": "https://github.com/salesforce/BLIP, SelTDA CVPR 2023"
}
```

Sau `create_model`, chạy `bash scripts/upload_kaggle_model_variants.sh`: mỗi dòng manifest → `kaggle models instances create` với `instanceSlug` = **Variant ID**. File >2GB dùng CLI multipart (script symlink file vào staging). MCP không upload weights trực tiếp.

---

## Provenance

- Paper: SelTDA (CVPR 2023)
- Base: [salesforce/BLIP](https://github.com/salesforce/BLIP)
- Repo: SelTDA (`feat/iterative-seltda`)
