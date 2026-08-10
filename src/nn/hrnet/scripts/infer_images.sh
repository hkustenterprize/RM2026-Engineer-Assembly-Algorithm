#!/usr/bin/env bash
set -euo pipefail

HRNET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NN_ROOT="$(cd "${HRNET_ROOT}/.." && pwd)"
SRC_ROOT="$(cd "${NN_ROOT}/.." && pwd)"
export PYTHONPATH="${SRC_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

YOLO_WEIGHTS="${YOLO_WEIGHTS:-}"
HRNET_CONFIG="${HRNET_CONFIG:-${HRNET_ROOT}/configs/td-hm_litehrnet18_exchange12_v11.0.py}"
HRNET_CHECKPOINT="${HRNET_CHECKPOINT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${HRNET_ROOT}/vis/inference_images}"
DEVICE="${DEVICE:-cuda:0}"
DETECTOR_DEVICE="${DETECTOR_DEVICE:-cpu}"
BBOX_CLASS_ID="${BBOX_CLASS_ID:-1}"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 image1 [image2 ...]" >&2
  exit 1
fi

if [ -z "${YOLO_WEIGHTS}" ] || [ -z "${HRNET_CHECKPOINT}" ]; then
  echo "Set YOLO_WEIGHTS and HRNET_CHECKPOINT before running." >&2
  exit 1
fi

uv run --project "${NN_ROOT}" python -m nn.hrnet.tools.infer \
  --yolo-weights "${YOLO_WEIGHTS}" \
  --hrnet-config "${HRNET_CONFIG}" \
  --hrnet-checkpoint "${HRNET_CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --detector-device "${DETECTOR_DEVICE}" \
  --bbox-class-id "${BBOX_CLASS_ID}" \
  --images "$@"
