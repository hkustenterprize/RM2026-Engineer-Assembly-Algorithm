_base_ = ["mmpose::_base_/default_runtime.py"]

custom_imports = dict(imports=["pillar_models"], allow_failed_imports=False)

data_root = "/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/dataset_v11.0/"
ann_root = "/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/dataset_v11.0_annotations/"
load_from = "/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/runs/td-hm_litehrnet18_exchange12_v11.0/best_coco_AP_epoch_22.pth"

max_epochs = 150
train_cfg = dict(max_epochs=max_epochs, val_interval=2)
default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=5, save_best="coco/AP", rule="greater")
)
custom_hooks = [
    dict(
        type="EarlyStoppingHook",
        monitor="coco/AP",
        rule="greater",
        patience=50,
        min_delta=0.001,
    )
]

base_lr = 1e-3
batch_size = 160
auto_scale_lr = dict(base_batch_size=batch_size * 4)
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
)
param_scheduler = [
    dict(type="LinearLR", start_factor=1.0e-5, by_epoch=False, begin=0, end=5),
    dict(
        type="CosineAnnealingLR",
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

codec = dict(
    type="MSRAHeatmap",
    input_size=(256, 256),
    heatmap_size=(128, 128),
    sigma=2.5,
)

model = dict(
    type="TopdownPoseEstimator",
    data_preprocessor=dict(
        type="PoseDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type="LiteHRNet",
        in_channels=3,
        extra=dict(
            stem=dict(stem_channels=32, out_channels=32, expand_ratio=1),
            num_stages=3,
            stages_spec=dict(
                # Official LiteHRNet-18 depth. LiteHRNet-30 uses (3, 8, 3).
                num_modules=(2, 4, 2),
                num_branches=(2, 3, 4),
                num_blocks=(2, 2, 2),
                module_type=("LITE", "LITE", "LITE"),
                with_fuse=(True, True, True),
                reduce_ratios=(8, 8, 8),
                num_channels=((40, 80), (40, 80, 160), (40, 80, 160, 320)),
            ),
            with_head=True,
        ),
    ),
    head=dict(
        type="PillarHeatmapHeadWithVis",
        in_channels=40,
        out_channels=12,
        # Keep 128x128 heatmaps with one 2x upsampling layer, but avoid the
        # older heavy 40->256 channel expansion used by SimpleBaseline-style heads.
        deconv_out_channels=(40,),
        deconv_kernel_sizes=(4,),
        loss=dict(type="KeypointMSELoss", use_target_weight=False),
        decoder=codec,
        use_vis=True,
        vis_label_mode="in_frame",
        vis_loss=dict(type="BCELoss", use_target_weight=False, use_sigmoid=True, loss_weight=0.0001),
        geo_kpt_indices=(0, 1, 2, 3, 4),
        # geo_loss=dict(
        #     type="GeometricConsistencyLoss",
        #     aspect_ratio=1.0,
        #     input_size=256,
        #     heatmap_size=128,
        #     hom_weight=0.005,
        #     chiral_weight=0.001,
        #     chiral_margin=0.0,
        # ),
        geo_loss=None
    ),
    test_cfg=dict(flip_test=True, flip_mode="heatmap", shift_heatmap=True, use_dark=True),
)

metainfo = dict(
    dataset_name="exchange12",
    keypoint_info={
        0: dict(name="TL", id=0, color=[255, 0, 0], type="pillar", swap="TR"),
        1: dict(name="TR", id=1, color=[0, 255, 0], type="pillar", swap="TL"),
        2: dict(name="BL", id=2, color=[0, 0, 255], type="pillar", swap="BR"),
        3: dict(name="BR", id=3, color=[255, 255, 0], type="pillar", swap="BL"),
        4: dict(name="ring", id=4, color=[255, 0, 255], type="pillar", swap=""),
        5: dict(name="light_BR", id=5, color=[0, 255, 255], type="exchange", swap="light_BL"),
        6: dict(name="light_TR", id=6, color=[0, 200, 255], type="exchange", swap="light_TL"),
        7: dict(name="shell_R", id=7, color=[255, 128, 0], type="exchange", swap="shell_L"),
        8: dict(name="shell_M", id=8, color=[255, 180, 0], type="exchange", swap="shell_M"),
        9: dict(name="shell_L", id=9, color=[255, 220, 0], type="exchange", swap="shell_R"),
        10: dict(name="light_TL", id=10, color=[128, 255, 255], type="exchange", swap="light_TR"),
        11: dict(name="light_BL", id=11, color=[180, 255, 255], type="exchange", swap="light_BR"),
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
        rotate_factor=5,
        scale_factor=(0.7, 1.2),
        shift_factor=0.1,
        shift_prob=0.1,
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
            dict(type="GaussianBlur", blur_limit=(3, 7), p=0.01),
            dict(type="GaussNoise", var_limit=(10.0, 50.0), p=0.03),
            dict(type="MotionBlur", blur_limit=5, p=0.2),
            dict(type="ImageCompression", quality_lower=50, quality_upper=100, p=0.4),
            dict(type="RandomShadow", num_shadows_lower=1, num_shadows_upper=3, p=0.1),
            dict(type="RandomBrightnessContrast", brightness_limit=0.4, contrast_limit=0.4, p=0.3),
            dict(type="ISONoise", color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.4),
            dict(type="Downscale", scale_min=0.5, scale_max=0.9, p=0.1),
            dict(type="RandomFog", fog_coef_lower=0.1, fog_coef_upper=0.4, p=0.02),
            dict(type="CLAHE", clip_limit=4.0, p=0.05),
            dict(type="Sharpen", alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.2),
            dict(
                type="OneOf",
                transforms=[
                    dict(
                        type="CoarseDropout",
                        max_holes=2,
                        max_height=0.3,
                        max_width=0.3,
                        min_holes=1,
                        min_height=0.1,
                        min_width=0.1,
                        fill_value=0,
                        p=1.0,
                    )
                ],
                p=0.1,
            ),
            dict(type="RandomGamma", gamma_limit=(80, 120), p=0.4),
            dict(type="PixelDropout", dropout_prob=0.01, per_channel=True, p=0.01),
            dict(type="GridDropout", ratio=0.3, unit_size_min=10, unit_size_max=80, p=0.02),
            dict(type="MedianBlur", blur_limit=5, p=0.03),
            dict(type="ChannelDropout", channel_drop_range=(1, 2), fill_value=0, p=0.02),
            dict(type="Posterize", num_bits=(4, 6), p=0.1),
            dict(type="RandomToneCurve", scale=0.3, p=0.1),
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

dataset_type = "CocoDataset"
data_mode = "topdown"
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file=ann_root + "exchange12_train.json",
        data_prefix=dict(img=data_root + "images/train/"),
        metainfo=metainfo,
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file=ann_root + "exchange12_val.json",
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

vis_backends = [dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")]
visualizer = dict(type="PoseLocalVisualizer", vis_backends=vis_backends, name="visualizer")
