#!/usr/bin/env bash
set -euo pipefail

HRNET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NN_ROOT="$(cd "${HRNET_ROOT}/.." && pwd)"
SRC_ROOT="$(cd "${NN_ROOT}/.." && pwd)"
export PYTHONPATH="${SRC_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
CONFIG_NAME="${1:-td-hm_litehrnet18_exchange12_v11.0.py}"
OUTPUT_DIR="${2:-${HRNET_ROOT}/vis/aug_preview_${CONFIG_NAME%.py}}"

uv run --project "${NN_ROOT}" python -m nn.hrnet.tools.preview_augmentations \
  "${HRNET_ROOT}/configs/${CONFIG_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-items 80
