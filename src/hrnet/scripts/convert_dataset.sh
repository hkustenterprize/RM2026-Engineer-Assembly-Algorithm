#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET_ROOT="${DATASET_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

if [ -z "${DATASET_ROOT}" ] || [ -z "${OUTPUT_DIR}" ]; then
  echo "Set DATASET_ROOT and OUTPUT_DIR before running." >&2
  exit 1
fi

uv run --project "${ROOT_DIR}" python "${ROOT_DIR}/data/convert_yolo_to_coco.py" \
  "${DATASET_ROOT}" \
  --output "${OUTPUT_DIR}"
