#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_NAME="${1:-td-hm_litehrnet18_exchange12_v11.0.py}"
OUTPUT_DIR="${2:-${ROOT_DIR}/vis/aug_preview_${CONFIG_NAME%.py}}"

uv run --project "${ROOT_DIR}" python "${ROOT_DIR}/tools/preview_augmentations.py" \
  "${ROOT_DIR}/configs/${CONFIG_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-items 80
