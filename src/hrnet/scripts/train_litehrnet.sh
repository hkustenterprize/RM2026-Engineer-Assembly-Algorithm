#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MMPPOSE_ROOT="${MMPPOSE_ROOT:-/path/to/mmpose}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_NAME="${1:-td-hm_litehrnet18_exchange12_v11.0.py}"
WORK_DIR="${2:-${ROOT_DIR}/runs/${CONFIG_NAME%.py}}"

export PYTHONPATH="${ROOT_DIR}/train:${MMPPOSE_ROOT}:${PYTHONPATH:-}"

"${PYTHON_BIN}" "${MMPPOSE_ROOT}/tools/train.py" \
  "${ROOT_DIR}/model_configs/${CONFIG_NAME}" \
  --work-dir "${WORK_DIR}"
