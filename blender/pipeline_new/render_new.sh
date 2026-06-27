#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLENDER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CUDA_VISIBLE_DEVICES=2 \
blender -b "${BLENDER_ROOT}/exchange_removed.blend" \
  -P "${SCRIPT_DIR}/render_dataset_new.py" \
  -- --config "${BLENDER_ROOT}/configs/exchange_new.yaml" \
  --strip_light_color off \
  --output_dir "${BLENDER_ROOT}/../data_v10.0/18" \
  --n_images 1000 --light_type off \
  --seed 40