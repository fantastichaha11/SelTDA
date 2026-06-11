# GCP VM — SelTDA

## VQAScore experiment (`seltda-vqascore`)

Pipeline: filter chỉ gate `vqascore` → train student → eval A-OKVQA val.

```bash
export PATH="$HOME/google-cloud-sdk/bin:$PATH"
# Tạo VM (zone có GPU L4 khả dụng, mặc định us-central1-a)
bash scripts/gcp/create_vm_vqascore.sh

# SSH / theo dõi
gcloud compute ssh seltda-vqascore --zone=us-central1-a --project=thesis-497813
sudo -u ubuntu tail -f /home/ubuntu/SelTDA/cache/logs/vqascore_run/pipeline.log
```

| Field | Value |
|-------|--------|
| Name | `seltda-vqascore` |
| Project | `thesis-497813` |
| Zone | `us-central1-a` |
| Type | `g2-standard-8` + **1× NVIDIA L4** |
| Boot disk | 200 GB |

Chạy lại pipeline trên VM đã có:

```bash
cd ~/SelTDA && git pull && tmux new -s vqascore 'bash scripts/gcp/run_vqascore_pipeline.sh'
```

## Instance cũ (C2)

| Field | Value |
|-------|--------|
| Name | `seltda-c2` |
| Zone | `us-central1-b` |

```bash
gcloud compute ssh seltda-c2 --zone=us-central1-b --project=thesis-497813
```

## Hạn chế quota (cần tăng nếu chạy full C2)

- **SSD region**: không tạo được disk `seltda-data` 1 TB (vượt quota `SSD_TOTAL_GB` tại `asia-southeast1`).
- VM dùng **một boot disk 100 GB** — **không đủ** cho toàn bộ COCO 4 split; cần xin tăng quota và gắn thêm disk.

Tăng quota: [Console → IAM & Quotas](https://console.cloud.google.com/iam-admin/quotas?project=thesis-497813) — tìm `SSD total GB` và `GPUs (all regions)` (hiện **1 GPU**).

Gắn data disk sau khi quota OK:

```bash
gcloud compute disks create seltda-data --zone=us-central1-b --size=500GB --type=pd-balanced
gcloud compute instances attach-disk seltda-c2 --zone=us-central1-b --disk=seltda-data --device-name=seltda-data
# SSH vào VM, format + mount (xem seltda-startup.sh)
```

## Setup SelTDA trên VM (thủ công)

```bash
git clone <your-repo-url> ~/thesis && cd ~/thesis/SelTDA
conda env create -f environment.yaml && conda activate blip
export PYTHONNOUSERSITE=1
# Chỉnh configs: vqa_root, ann_root, pretrained → path local
bash dataset.sh   # cần disk lớn
```
