python /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/yolopose/inference/inference_video.py \
    --model /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/yolopose/runs/detect/yolo26s_det_v10.12/weights/best.pt \
    --source /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/data/video/official.mp4 \
    --output /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/yolopose/eval_results/yolo_v10.12_eval_results/video \
    --conf 0.5 \
    --device 3