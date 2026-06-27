ROOT="/data/datasets/zguobd/RM2026-Engineer-Host"
PYTHONPATH="${ROOT}/cv/nn/hrnet/train:/data/datasets/zguobd/mmpose:${PYTHONPATH:-}" \
/data/datasets/zguobd/miniconda3/envs/mm/bin/python \
"${ROOT}/cv/nn/hrnet/train/inference_video.py" \
  --yolo-weights "${ROOT}/cv/nn/yolopose/runs/detect/yolo26s_det_v10.12/weights/best_openvino_model" \
  --hrnet-config "${ROOT}/cv/nn/hrnet/model_configs/td-hm_litehrnet30_exchange12_v9.1.py" \
  --hrnet-checkpoint "${ROOT}/cv/nn/hrnet/runs/litehrnet30_exchange12_v9.1_bs48/best_coco_AP_epoch_200.pth" \
  --source "${ROOT}/cv/nn/data/video/official.mp4" \
  --output "${ROOT}/cv/nn/hrnet/yolo10.12_hrnet9.2_best_official.mp4" \
  --device cuda:1 \
  --crop-margin 0.05 \
  --frame-stride 2 \
  --start-frame 0 \
  --end-frame -1 \
  --yolo-conf 0.3
