#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLENDER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SCENE="${SCENE:?Set SCENE to a local .blend file}"
CONFIG="${CONFIG:-${BLENDER_ROOT}/configs/example.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${BLENDER_ROOT}/../../datasets/rendered_assets}"
N_IMAGES="${N_IMAGES:-1000}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
blender -b "${SCENE}" \
  -P "${SCRIPT_DIR}/render_dataset_new.py" \
  -- --config "${CONFIG}" \
  --output_dir "${OUTPUT_DIR}" \
  --n_images "${N_IMAGES}" \
  "$@"
