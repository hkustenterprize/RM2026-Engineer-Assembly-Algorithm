#!/usr/bin/env bash
set -euo pipefail

HRNET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NN_ROOT="$(cd "${HRNET_ROOT}/.." && pwd)"
SRC_ROOT="$(cd "${NN_ROOT}/.." && pwd)"
export PYTHONPATH="${SRC_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

DATASET_ROOT="${DATASET_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

if [ -z "${DATASET_ROOT}" ] || [ -z "${OUTPUT_DIR}" ]; then
  echo "Set DATASET_ROOT and OUTPUT_DIR before running." >&2
  exit 1
fi

uv run --project "${NN_ROOT}" python -m nn.hrnet.data.convert_yolo_to_coco \
  "${DATASET_ROOT}" \
  --output "${OUTPUT_DIR}"
