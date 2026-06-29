#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/yolopose"

# NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 python "${ROOT}/train/train_detect.py" --config "${ROOT}/config/detect/v9.2.yaml"

NCCL_P2P_DISABLE=0 NCCL_IB_DISABLE=1 python "${ROOT}/train/train_detect.py" \
  --config "${ROOT}/config/detect/v9.2.yaml"
