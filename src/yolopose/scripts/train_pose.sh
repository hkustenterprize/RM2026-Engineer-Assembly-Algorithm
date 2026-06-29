#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/yolopose"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
NCCL_P2P_DISABLE=0 NCCL_IB_DISABLE=1 python "${ROOT}/train/train.py" \
  --config "${ROOT}/config/train_config.yaml" \
  "$@"
