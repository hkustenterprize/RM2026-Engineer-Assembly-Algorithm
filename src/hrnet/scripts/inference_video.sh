#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

YOLO_WEIGHTS="${YOLO_WEIGHTS:-/path/to/yolo_detect_best.pt}"
HRNET_CONFIG="${HRNET_CONFIG:-${ROOT_DIR}/model_configs/td-hm_litehrnet18_exchange12_v11.0.py}"
HRNET_CHECKPOINT="${HRNET_CHECKPOINT:-/path/to/hrnet_best.pth}"
DEVICE="${DEVICE:-cuda:0}"
BBOX_CLASS_ID="${BBOX_CLASS_ID:-1}"

SOURCE="${SOURCE:-}"
OUTPUT="${OUTPUT:-}"

if [ -z "${SOURCE}" ] || [ -z "${OUTPUT}" ]; then
  echo "Set SOURCE=/path/to/input.mp4 and OUTPUT=/path/to/output.mp4 before running." >&2
  exit 1
fi

"${PYTHON_BIN}" "${ROOT_DIR}/train/inference_video.py" \
  --yolo-weights "${YOLO_WEIGHTS}" \
  --hrnet-config "${HRNET_CONFIG}" \
  --hrnet-checkpoint "${HRNET_CHECKPOINT}" \
  --source "${SOURCE}" \
  --output "${OUTPUT}" \
  --device "${DEVICE}" \
  --bbox-class-id "${BBOX_CLASS_ID}"
