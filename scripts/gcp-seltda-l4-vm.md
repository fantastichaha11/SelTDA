# GCP VM `seltda-l4` — L4 + 200 GB persistent disk

## Instance

| Field | Value |
|-------|--------|
| **Name** | `seltda-l4` |
| **Project** | `thesis-497813` |
| **Zone** | `asia-southeast1-b` (L4 stock in US zones was exhausted at create time) |
| **Machine** | `g2-standard-8` — 8 vCPU, 32 GB RAM, **1× NVIDIA L4 (24 GB)** |
| **Boot disk** | **200 GB** `pd-balanced`, **`auto-delete=false`** |
| **Image** | Deep Learning VM: `pytorch-2-9-cu129-ubuntu-2204-nvidia-580` |

## Data survives **stop** (and even **delete** of VM)

- **`gcloud compute instances stop`** — VM tắt, **disk giữ nguyên**, mọi file trên `/` còn khi **start** lại.
- Boot disk có flag **`--no-boot-disk-auto-delete`**: nếu bạn **xóa VM**, disk **vẫn còn** trên GCP (có thể gắn lại VM mới).

**Không dùng** `gcloud compute instances delete` nếu bạn không chắc — dùng **stop** để tiết kiệm tiền GPU.

```bash
# Tắt VM (giữ data) — không tính phí GPU, vẫn tính phí disk ~200GB
gcloud compute instances stop seltda-l4 --zone=asia-southeast1-b --project=thesis-497813

# Bật lại
gcloud compute instances start seltda-l4 --zone=asia-southeast1-b --project=thesis-497813
```

## SSH

```bash
gcloud compute ssh seltda-l4 --zone=asia-southeast1-b --project=thesis-497813
```

Sau khi vào VM, kiểm tra GPU:

```bash
nvidia-smi
```

## Setup đã chạy trên VM (2026-06-01)

| Bước | Trạng thái | Log |
|------|------------|-----|
| Clone `feat/iterative-seltda` → `~/SelTDA` | ✅ | commit `9766d47` |
| Miniconda + `bash setup.sh` → env `vqa` | ✅ | `~/seltda-setup.log` — torch `2.12.0+cu130`, CUDA OK |
| `bash dataset.sh` | 🔄 chạy nền | `~/seltda-dataset.log` (COCO + A-OKVQA + checkpoints, vài giờ) |

Mỗi lần SSH, `.bashrc` tự `conda activate vqa` và `cd ~/SelTDA`.

```bash
# Theo dõi tải data
gcloud compute ssh seltda-l4 --zone=asia-southeast1-b --project=thesis-497813 \
  --command='tail -f ~/seltda-dataset.log'

# Kiểm tra GPU + env sau khi login
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Sau khi `dataset.sh` xong, trên VM:

```bash
python convert_aokvqa.py --config configs/aokvqa.yaml
pytest -m "not slow" -q   # smoke tests
```

## Tạo lại VM (nếu đã xóa instance nhưng giữ disk)

Liệt kê disk:

```bash
gcloud compute disks list --project=thesis-497813 --filter="name~seltda-l4"
```

## Recreate command (reference)

```bash
gcloud compute instances create seltda-l4 \
  --project=thesis-497813 \
  --zone=asia-southeast1-b \
  --machine-type=g2-standard-8 \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-balanced \
  --no-boot-disk-auto-delete \
  --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --provisioning-model=STANDARD
```
