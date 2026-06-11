# GCP VM — SelTDA / C2

## Instance (đã tạo)

| Field | Value |
|-------|--------|
| Name | `seltda-c2` |
| Project | `thesis-497813` |
| Zone | `us-central1-b` |
| Type | `g2-standard-8` (8 vCPU, 32 GB RAM, **1× NVIDIA L4**) |
| Boot disk | 100 GB `pd-balanced` |
| External IP | `34.172.51.251` (có thể đổi khi stop/start) |

## SSH

```bash
export PATH="$HOME/google-cloud-sdk/bin:$PATH"
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
