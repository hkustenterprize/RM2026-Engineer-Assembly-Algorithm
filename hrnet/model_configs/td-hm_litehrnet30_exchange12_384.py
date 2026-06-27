_base_ = ["./td-hm_litehrnet30_pillar5_geo_vis.py"]

# ──────────────────────────────────────────────────────────────────────────────
#  Exchange12 experiment with input_size=384 (based on pillar5_geo_vis)
#
#  Based on td-hm_litehrnet30_pillar5_geo_vis.py, but:
#    1. input_size: 256 → 384
#    2. heatmap_size: 128 → 192 (maintains stride=2.0)
#    3. 12 keypoints (5 pillar + 7 exchange)
#    4. Includes visibility branch from pillar5_geo_vis
# ──────────────────────────────────────────────────────────────────────────────

data_root = "/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/dataset_v11.0/"
ann_root = "/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/dataset_v11.0_annotations/"
load_from = None
batch_size = 32  # reduced for larger 384×384 input
auto_scale_lr = dict(base_batch_size=batch_size * 4)

# ──────────────────────────────────────────────────────────────────────────────
#  Codec  — 改为 384×384 输入，192×192 热力图
# ──────────────────────────────────────────────────────────────────────────────
codec = dict(
    type="MSRAHeatmap",
    input_size=(384, 384),
    heatmap_size=(192, 192),
    sigma=2.5,
)

model = dict(
    head=dict(
        type="PillarHeatmapHeadWithVis",
        decoder=codec,
        use_vis=True,
        vis_label_mode="in_frame",
        vis_loss=dict(
            type="BCELoss",
            use_target_weight=False,
            use_sigmoid=True,
            loss_weight=0.0005,
        ),
        geo_kpt_indices=(0, 1, 2, 3, 4),
        out_channels=12,  # pillar 5 + exchange 7
        geo_loss=dict(
            type="GeometricConsistencyLoss",
            aspect_ratio=1.0,
            input_size=384,
            heatmap_size=192,
            hom_weight=0.005,
            chiral_weight=0.0005,
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

train_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(
        type="RandomBBoxTransform",
        rotate_factor=30,
        scale_factor=(0.8, 2.5),
        shift_factor=0.4,
        shift_prob=0.4,
    ),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(
        type="PhotometricDistortion",
        brightness_delta=60,
        contrast_range=(0.6, 1.4),
        saturation_range=(0.6, 1.4),
        hue_delta=30,
    ),
    dict(
        type="Albumentation",
        transforms=[
            dict(type="GaussianBlur", blur_limit=(3, 7), p=0.3),
            dict(type="GaussNoise", var_limit=(10.0, 50.0), p=0.2),
            dict(type="MotionBlur", blur_limit=5, p=0.2),
            dict(type="ImageCompression", quality_lower=50, quality_upper=100, p=0.3),
            dict(type="RandomShadow", num_shadows_lower=1, num_shadows_upper=3, p=0.4),
            dict(
                type="RandomBrightnessContrast",
                brightness_limit=0.4,
                contrast_limit=0.4,
                p=0.3,
            ),
            dict(
                type="ISONoise", color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2
            ),
            dict(type="Downscale", scale_min=0.5, scale_max=0.9, p=0.2),
            dict(type="RandomFog", fog_coef_lower=0.1, fog_coef_upper=0.4, p=0.2),
            dict(type="CLAHE", clip_limit=4.0, p=0.3),
            dict(type="Sharpen", alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.2),
            dict(
                type="OneOf",
                transforms=[
                    dict(
                        type="CoarseDropout",
                        max_holes=1,
                        max_height=0.4,
                        max_width=0.4,
                        min_holes=1,
                        min_height=0.2,
                        min_width=0.2,
                        fill_value=0,
                        p=1.0,
                    )
                ],
                p=0.2,
            ),
            dict(type="RandomGamma", gamma_limit=(80, 120), p=0.3),
            dict(type="PixelDropout", dropout_prob=0.01, per_channel=True, p=0.2),
            dict(
                type="GridDropout", ratio=0.3, unit_size_min=10, unit_size_max=80, p=0.1
            ),
            dict(type="MedianBlur", blur_limit=5, p=0.2),
            dict(type="ChannelDropout", channel_drop_range=(1, 2), fill_value=0, p=0.2),
            dict(type="Posterize", num_bits=(4, 6), p=0.2),
            dict(type="RandomToneCurve", scale=0.3, p=0.3),
        ],
    ),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]

val_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="PackPoseInputs"),
]

train_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
        ann_file=ann_root + "exchange12_train.json",
        data_root=data_root,
        data_prefix=dict(img=data_root + "images/train/"),
        metainfo=metainfo,
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file=ann_root + "exchange12_val.json",
        data_root=data_root,
        data_prefix=dict(img=data_root + "images/val/"),
        metainfo=metainfo,
        test_mode=True,
        pipeline=val_pipeline,
    ),
)

test_dataloader = val_dataloader

val_evaluator = [
    dict(type="CocoMetric", ann_file=ann_root + "exchange12_val.json"),
    dict(type="PCKAccuracy", thr=0.05, prefix="pck05"),
    dict(type="PCKAccuracy", thr=0.10, prefix="pck10"),
]
test_evaluator = val_evaluator
