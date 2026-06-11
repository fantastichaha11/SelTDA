#!/bin/bash
# GCP startup: mount data disk, install deps for SelTDA dataset.sh
set -euo pipefail

DATA_DISK=/dev/disk/by-id/google-seltda-data
MOUNT=/data

if [ -b "$DATA_DISK" ]; then
  if ! blkid "$DATA_DISK" &>/dev/null; then
    mkfs.ext4 -F "$DATA_DISK"
  fi
  mkdir -p "$MOUNT"
  if ! grep -q "$MOUNT" /etc/fstab; then
    echo "$DATA_DISK $MOUNT ext4 defaults,nofail 0 2" >> /etc/fstab
  fi
  mount -a
  chown -R "$(logname 2>/dev/null || echo ubuntu):$(id -gn)" "$MOUNT" 2>/dev/null || chown -R ubuntu:ubuntu "$MOUNT" || true
fi

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git aria2 unzip curl tmux htop

mkdir -p "$MOUNT/thesis-cache" "$MOUNT/thesis-datasets" 2>/dev/null || true

echo "SelTDA GCP startup done. Data mount: $MOUNT" | tee /var/log/seltda-startup.log
