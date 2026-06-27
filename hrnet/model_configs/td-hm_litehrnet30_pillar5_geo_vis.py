"""
td-hm_litehrnet30_pillar5_geo_vis.py
====================================
基于 td-hm_litehrnet30_pillar5.py，保持 pillar5(v9.1) 的训练配置不变，
仅将 head 升级为带 visibility 分支的 PillarHeatmapHeadWithVis，并启用
几何一致性损失。

训练命令:
  PYTHONPATH=/data/datasets/zguobd/mmpose:$PYTHONPATH \
  CUDA_VISIBLE_DEVICES=4 \
  /data/datasets/zguobd/miniconda3/envs/mm/bin/python \
  /data/datasets/zguobd/mmpose/tools/train.py \
  /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/model_configs/td-hm_litehrnet30_pillar5_geo_vis.py \
  --work-dir /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/runs/litehrnet30_pillar5_geo_vis
"""

_base_ = ["./td-hm_litehrnet30_pillar5.py"]

model = dict(
    head=dict(
        type="PillarHeatmapHeadWithVis",
        use_vis=True,
        vis_label_mode="in_frame",
        vis_target_mode="binary",
        vis_loss=dict(
            type="BCELoss",
            use_target_weight=False,
            use_sigmoid=True,
        ),
        geo_loss=dict(
            type="GeometricConsistencyLoss",
            aspect_ratio=1.0,
            input_size=256,
            heatmap_size=128,
            hom_weight=0.005,
            chiral_weight=0.001,
            chiral_margin=0.0,
        ),
    )
)
