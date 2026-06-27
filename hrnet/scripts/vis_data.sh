mkdir -p /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/aug_vis_td-hm_litehrnet18_exchange12_640
PYTHONPATH=/data/datasets/zguobd/mmpose:$PYTHONPATH \
/data/datasets/zguobd/miniconda3/envs/mm/bin/python \
/data/datasets/zguobd/mmpose/tools/misc/browse_dataset.py \
/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/model_configs/td-hm_litehrnet18_exchange12_v11.0.py \
--output-dir /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/aug_vis_td-hm_litehrnet18_exchange12_640 \
--not-show \
--show-interval 0 \
--max-item-per-dataset 80 \
--phase train \
--mode transformed \
--draw-bbox \
