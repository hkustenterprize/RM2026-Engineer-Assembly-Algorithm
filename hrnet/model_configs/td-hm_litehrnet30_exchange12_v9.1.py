_base_ = ["./td-hm_litehrnet30_pillar_geo.py"]

# ──────────────────────────────────────────────────────────────────────────────
#  Exchange12 experiment
#
#  Based on td-hm_litehrnet30_pillar_geo.py, but:
#    1. dataset_root / ann_root -> dataset_v9.1 + exchange12 annotations
#    2. heatmap head out_channels: 5 -> 12
#    3. geo_loss disabled because GeometricConsistencyLoss is defined only for
#       the 5 pillar keypoints (TL/TR/BL/BR/ring)
#    4. lower batch size for 4-GPU training
# ──────────────────────────────────────────────────────────────────────────────

data_root = "/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/dataset_v11.0/"
ann_root = "/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/dataset_v11.0_annotations/"
load_from = "/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/runs/litehrnet30_exchange12_v10.1/epoch_80.pth"

batch_size = 32
auto_scale_lr = dict(base_batch_size=batch_size * 4)  # for 4-GPU training

model = dict(
    head=dict(
        type="PillarHeatmapHeadWithVis",
        use_vis=True,
        vis_label_mode="in_frame",
        vis_loss=dict(
            type="BCELoss",
            use_target_weight=False,
            use_sigmoid=True,
        ),
        geo_kpt_indices=(0, 1, 2, 3, 4),
        # out_channels=5,   # pillar-only heatmap head
        out_channels=12,  # pillar 5 + exchange 7
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

metainfo = dict(
    dataset_name="exchange12",
    keypoint_info={
        0: dict(name="TL", id=0, color=[255, 0, 0], type="pillar", swap="TR"),
        1: dict(name="TR", id=1, color=[0, 255, 0], type="pillar", swap="TL"),
        2: dict(name="BL", id=2, color=[0, 0, 255], type="pillar", swap="BR"),
        3: dict(name="BR", id=3, color=[255, 255, 0], type="pillar", swap="BL"),
        4: dict(name="ring", id=4, color=[255, 0, 255], type="pillar", swap=""),
        5: dict(
            name="light_BR", id=5, color=[0, 255, 255], type="exchange", swap="light_BL"
        ),
        6: dict(
            name="light_TR", id=6, color=[0, 200, 255], type="exchange", swap="light_TL"
        ),
        7: dict(
            name="shell_R", id=7, color=[255, 128, 0], type="exchange", swap="shell_L"
        ),
        8: dict(
            name="shell_M", id=8, color=[255, 180, 0], type="exchange", swap="shell_M"
        ),
        9: dict(
            name="shell_L", id=9, color=[255, 220, 0], type="exchange", swap="shell_R"
        ),
        10: dict(
            name="light_TL",
            id=10,
            color=[128, 255, 255],
            type="exchange",
            swap="light_TR",
        ),
        11: dict(
            name="light_BL",
            id=11,
            color=[180, 255, 255],
            type="exchange",
            swap="light_BR",
        ),
    },
    skeleton_info={
        0: dict(link=("TL", "TR"), id=0, color=[255, 255, 255]),
        1: dict(link=("TR", "BR"), id=1, color=[255, 255, 255]),
        2: dict(link=("BR", "BL"), id=2, color=[255, 255, 255]),
        3: dict(link=("BL", "TL"), id=3, color=[255, 255, 255]),
        4: dict(link=("light_BR", "light_TR"), id=4, color=[255, 255, 255]),
        5: dict(link=("light_TR", "light_TL"), id=5, color=[255, 255, 255]),
        6: dict(link=("light_TL", "light_BL"), id=6, color=[255, 255, 255]),
        7: dict(link=("shell_R", "shell_M"), id=7, color=[255, 255, 255]),
        8: dict(link=("shell_M", "shell_L"), id=8, color=[255, 255, 255]),
    },
    joint_weights=[1.0] * 12,
    sigmas=[0.02] * 12,
)

train_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
        ann_file=ann_root + "exchange12_train.json",
        data_root=data_root,
        data_prefix=dict(img=data_root + "images/train/"),
        metainfo=metainfo,
    ),
)

val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file=ann_root + "exchange12_val.json",
        data_root=data_root,
        data_prefix=dict(img=data_root + "images/val/"),
        metainfo=metainfo,
    ),
)

test_dataloader = val_dataloader

val_evaluator = [
    dict(type="CocoMetric", ann_file=ann_root + "exchange12_val.json"),
    dict(type="PCKAccuracy", thr=0.05, prefix="pck05"),
    dict(type="PCKAccuracy", thr=0.10, prefix="pck10"),
]
test_evaluator = val_evaluator
