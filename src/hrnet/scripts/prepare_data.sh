#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATASET_ROOT="${DATASET_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

if [ -z "${DATASET_ROOT}" ] || [ -z "${OUTPUT_DIR}" ]; then
  echo "Set DATASET_ROOT=/path/to/yolo_pose_dataset and OUTPUT_DIR=/path/to/annotations before running." >&2
  exit 1
fi

"${PYTHON_BIN}" "${ROOT_DIR}/data_process/prepare_data_new.py" \
  "${DATASET_ROOT}" \
  --output "${OUTPUT_DIR}" \
  --mode exchange12
