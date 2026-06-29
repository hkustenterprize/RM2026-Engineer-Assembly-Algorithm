#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MMPPOSE_ROOT="${MMPPOSE_ROOT:-/path/to/mmpose}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_NAME="${1:-td-hm_litehrnet18_exchange12_v11.0.py}"
OUTPUT_DIR="${2:-${ROOT_DIR}/vis/aug_preview_${CONFIG_NAME%.py}}"

export PYTHONPATH="${MMPPOSE_ROOT}:${PYTHONPATH:-}"

"${PYTHON_BIN}" "${MMPPOSE_ROOT}/tools/misc/browse_dataset.py" \
  "${ROOT_DIR}/model_configs/${CONFIG_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --not-show \
  --show-interval 0 \
  --max-item-per-dataset 80 \
  --phase train \
  --mode transformed \
  --draw-bbox
