python /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/yolopose/export_openvino.py \
  --weights /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/yolopose/runs/detect/yolo26s_det_v10.12/weights/best.pt \
  --imgsz 640 \
  --half

    # --no-half、--dynamic、--no-nms、--device cuda:0