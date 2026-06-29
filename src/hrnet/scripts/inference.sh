#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

YOLO_WEIGHTS="${YOLO_WEIGHTS:-/path/to/yolo_detect_best.pt}"
HRNET_CONFIG="${HRNET_CONFIG:-${ROOT_DIR}/model_configs/td-hm_litehrnet18_exchange12_v11.0.py}"
HRNET_CHECKPOINT="${HRNET_CHECKPOINT:-/path/to/hrnet_best.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/vis/inference_images}"
DEVICE="${DEVICE:-cuda:0}"
BBOX_CLASS_ID="${BBOX_CLASS_ID:-1}"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 image1 [image2 ...]" >&2
  exit 1
fi

"${PYTHON_BIN}" "${ROOT_DIR}/train/inference.py" \
  --yolo-weights "${YOLO_WEIGHTS}" \
  --hrnet-config "${HRNET_CONFIG}" \
  --hrnet-checkpoint "${HRNET_CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --bbox-class-id "${BBOX_CLASS_ID}" \
  --images "$@"
