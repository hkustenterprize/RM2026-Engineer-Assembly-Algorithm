#!/usr/bin/env bash
# compose.sh -- merge all blender dirs in final_data_a1 into one dataset

BASE=/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/data_v7.0
BG_DIR=/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/data/background
OUTPUT_DIR=/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/dataset_v8.0

python /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/yolopose/data_process/compose_dataset.py \
    --blender_dir /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/data_v10.0 \
    --output_dir /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/dataset_v10.0_near \
    --bg_dir /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/data/background \
    --seed 30 --vis_output /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/v10.0_near_vis --val_split 0.1 \
    --min_pillars 1 --max_pillars 1 --n_output 5000 \
    --workers 16 \
    --auto_crop_alpha
