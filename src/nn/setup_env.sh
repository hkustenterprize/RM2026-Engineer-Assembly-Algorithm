#!/usr/bin/env bash
set -euo pipefail

NN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${NN_ROOT}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export NO_ALBUMENTATIONS_UPDATE=1

UV_SYNC_ARGS=(--extra export)
if [ -f "${NN_ROOT}/uv.lock" ]; then
  UV_SYNC_ARGS+=(--locked)
fi

# Chumpy and MMCV use legacy build scripts that expect pip, setuptools and
# wheel to exist in the target environment.
if [ ! -x "${NN_ROOT}/.venv/bin/pip" ]; then
  uv venv --python 3.10 --seed --allow-existing
fi

# MMCV imports PyTorch while compiling its native extension. Install the rest
# first so the non-isolated MMCV build can reuse the locked PyTorch version.
uv sync "${UV_SYNC_ARGS[@]}" \
  --no-install-package mmcv \
  --no-build-isolation-package chumpy

# LiteHRNet does not use MMCV CUDA operators. Building the complete CPU
# extension satisfies MMPose's module imports without requiring a CUDA toolkit.
CUDA_VISIBLE_DEVICES="" MMCV_WITH_OPS=1 \
  uv sync "${UV_SYNC_ARGS[@]}" --no-build-isolation-package mmcv

# The published xtcocotools wheel targets the NumPy 1.x ABI. Rebuild only its
# native extension against the locked NumPy 2.x installation.
uv pip install --python "${NN_ROOT}/.venv/bin/python" \
  --reinstall \
  --no-deps \
  --no-binary xtcocotools \
  --no-build-isolation \
  "xtcocotools==1.14.3"

uv run rm26-nn check
