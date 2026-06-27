#!/usr/bin/env bash
set -euo pipefail

ROOT="/data/datasets/zguobd/RM2026-Engineer-Host"
PYTHON="/data/datasets/zguobd/miniconda3/envs/mm/bin/python"

export PYTHONPATH="${ROOT}/cv/nn/hrnet/train:/data/datasets/zguobd/mmpose:${PYTHONPATH:-}"

EVAL_SAMPLES="${ROOT}/cv/nn/data/eval/eval_samples"
# Avoid matplotlib cache warnings in restricted environments.
export MPLCONFIGDIR="/tmp/matplotlib"

YOLO_WEIGHTS="${ROOT}/cv/nn/yolopose/runs/detect/yolo26s_det_v10.12/weights/best_openvino_model"
HRNET_CONFIG="${ROOT}/cv/nn/hrnet/runs/litehrnet30_exchange12_384_v11.1/td-hm_litehrnet30_exchange12_384.py"
HRNET_CHECKPOINT="${ROOT}/cv/nn/hrnet/runs/litehrnet30_exchange12_384_v11.1/best_coco_AP_epoch_280.pth"
OUTPUT_DIR="${ROOT}/cv/nn/hrnet/vis/litehrnet30_exchange12_384_v11.1_epoch280"
DEVICE="cuda:0"

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

# New outputs:
#   *_kpt.png           final keypoint overlay
#   *_heatmap.png       per-channel heatmap grid
#   *_corner_debug.png  before/after corner-refine comparison
# Notes:
#   - exchange12 model defaults to YOLO class 1 (exchange bbox) in inference.py
#   - corner_refine is ON by default; pass --no-corner-refine to disable
"${PYTHON}" "${ROOT}/cv/nn/hrnet/train/inference.py" \
  --yolo-weights "${YOLO_WEIGHTS}" \
  --hrnet-config "${HRNET_CONFIG}" \
  --hrnet-checkpoint "${HRNET_CHECKPOINT}" \
  --images "${IMAGES[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --device "${DEVICE}" \
  --crop-margin 0.05 \
  --vis-heatmap \
  --vis-corner-debug \
  --heatmap-kpt -1 \
  --heatmap-alpha 0.5 \
  --bbox-class-id 1 \
  --no-draw-scores \
  "$@"
