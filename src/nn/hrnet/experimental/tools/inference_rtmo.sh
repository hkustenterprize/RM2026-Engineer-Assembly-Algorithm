#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/datasets/zguobd/RM2026-Engineer-Host"
PYTHON="/data/datasets/zguobd/miniconda3/envs/mm/bin/python"

export PYTHONPATH="${ROOT}/cv/nn/hrnet/train:/data/datasets/zguobd/mmpose:${PYTHONPATH:-}"

EVAL_SAMPLES="${ROOT}/cv/nn/data/eval/eval_samples"
export MPLCONFIGDIR="/tmp/matplotlib"

RTMO_RUN="${ROOT}/cv/nn/hrnet/runs/rtmo-s_exchange12_640_v11.0"
RTMO_CONFIG="${RTMO_RUN}/rtmo-s_exchange12_640.py"
RTMO_CHECKPOINT="${RTMO_RUN}/epoch_120.pth"
OUTPUT_DIR="${ROOT}/cv/nn/hrnet/vis/rtmo-s_exchange12_640_v11.0_epoch120"
DEVICE="${DEVICE:-cuda:0}"

IMAGES=(
  "${EVAL_SAMPLES}/image0.png"
  "${EVAL_SAMPLES}/image1.png"
  "${EVAL_SAMPLES}/image2.png"
  "${EVAL_SAMPLES}/image3.png"
  "${EVAL_SAMPLES}/image4.png"
  "${EVAL_SAMPLES}/image5.png"
  "${EVAL_SAMPLES}/image6.png"
  "${EVAL_SAMPLES}/image7.png"
  "${EVAL_SAMPLES}/image8.png"
  "${EVAL_SAMPLES}/image9.png"
  "${EVAL_SAMPLES}/image10.png"
  "${EVAL_SAMPLES}/image11.png"
  "${EVAL_SAMPLES}/image12.png"
  "${EVAL_SAMPLES}/image13.png"
  "${EVAL_SAMPLES}/image14.png"
  "${EVAL_SAMPLES}/image15.png"
  "${EVAL_SAMPLES}/image16.png"
  "${EVAL_SAMPLES}/image17.png"
  "${EVAL_SAMPLES}/image18.png"
  "${EVAL_SAMPLES}/image19.png"
  "${EVAL_SAMPLES}/image20.png"
  "${EVAL_SAMPLES}/image21.png"
  "${EVAL_SAMPLES}/image22.png"
  "${EVAL_SAMPLES}/image23.png"
  "${EVAL_SAMPLES}/image24.png"
  "${EVAL_SAMPLES}/image25.png"
  "${EVAL_SAMPLES}/image26.png"
  "${EVAL_SAMPLES}/image27.png"
)

"${PYTHON}" "${ROOT}/cv/nn/hrnet/train/inference.py" \
  --mode rtmo \
  --rtmo-config "${RTMO_CONFIG}" \
  --rtmo-checkpoint "${RTMO_CHECKPOINT}" \
  --images "${IMAGES[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  "$@"
