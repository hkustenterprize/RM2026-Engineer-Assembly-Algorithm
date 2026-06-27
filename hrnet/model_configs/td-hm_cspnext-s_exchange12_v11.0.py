# Standalone CSPNeXt-S exchange12 heatmap config.
# Copied from litehrnet30_exchange12_v9.1 behavior, with only backbone/neck and
# optimizer changed. No _base_ dependency.

custom_imports = dict(imports=["pillar_models"], allow_failed_imports=False)
default_scope = "mmpose"

data_root = "/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/dataset_v11.0/"
ann_root = "/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/dataset_v11.0_annotations/"
pretrained = "/data/datasets/zguobd/.cache/torch/hub/checkpoints/cspnext-s_udp-aic-coco_210e-256x192-92f5a029_20230130.pth"

max_epochs = 50
batch_size = 64
base_lr = 0.004

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=5, save_best="coco/AP", rule="greater"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="PoseVisualizationHook", enable=False),
    badcase=dict(type="BadCaseAnalysisHook", enable=False, out_dir="badcase", metric_type="loss", badcase_thr=5),
)
env_cfg = dict(cudnn_benchmark=False, mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0), dist_cfg=dict(backend="nccl"))
vis_backends = [dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")]
visualizer = dict(type="PoseLocalVisualizer", vis_backends=vis_backends, name="visualizer")
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True, num_digits=6)
log_level = "INFO"
load_from = None
resume = False

train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=2)
val_cfg = dict()
test_cfg = dict()
custom_hooks = [dict(type="EarlyStoppingHook", monitor="coco/AP", rule="greater", patience=50, min_delta=0.001)]

# MuSGD comes from ultralytics.optim and is registered in pillar_models.py.
optim_wrapper = dict(
    type="OptimWrapper",
    constructor="MuonSGDOptimWrapperConstructor",
    optimizer=dict(type="MuSGD", lr=base_lr, momentum=0.9, weight_decay=0.05, nesterov=True, muon=0.2, sgd=1.0),
    paramwise_cfg=dict(bypass_duplicate=True),
)
param_scheduler = [
    dict(type="CosineAnnealingLR", eta_min=base_lr * 0.05, begin=0, end=max_epochs, T_max=100, by_epoch=True, convert_to_iter_based=True),
]
auto_scale_lr = dict(base_batch_size=256)

codec = dict(type="MSRAHeatmap", input_size=(256, 256), heatmap_size=(128, 128), sigma=2.5)
norm_cfg = dict(type="BN", momentum=0.03, eps=0.001)
act_cfg = dict(type="SiLU")

model = dict(
    type="TopdownPoseEstimator",
    data_preprocessor=dict(type="PoseDataPreprocessor", mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], bgr_to_rgb=True),
    backbone=dict(
        type="CSPNeXt",
        arch="P5",
        deepen_factor=0.33,
        widen_factor=0.5,
        expand_ratio=0.5,
        out_indices=(1, 2, 3, 4),
        use_depthwise=False,
        channel_attention=True,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg,
        init_cfg=dict(type="Pretrained", prefix="backbone.", checkpoint=pretrained),
    ),
    neck=dict(
        type="FullCSPNeXtPAFPNFuse",
        in_channels=[64, 128, 256, 512],
        out_indices=(0, 1, 2, 3),
        fuse_out_channels=64,
        num_csp_blocks=1,
        use_depthwise=False,
        expand_ratio=0.5,
        upsample_cfg=dict(scale_factor=2, mode="nearest"),
        norm_cfg=norm_cfg,
        act_cfg=act_cfg,
    ),
    head=dict(
        type="PillarHeatmapHeadWithVis",
        in_channels=64,
        out_channels=12,
        deconv_out_channels=(256,),
        deconv_kernel_sizes=(4,),
        loss=dict(type="KeypointMSELoss", use_target_weight=False),
        decoder=codec,
        geo_loss=None,
        use_vis=True,
        vis_label_mode="in_frame",
        vis_loss=dict(
            type="BCELoss",
            use_target_weight=False,
            use_sigmoid=True,
            loss_weight=0.0005,
        ),
    ),
    test_cfg=dict(flip_test=True, flip_mode="heatmap", shift_heatmap=True, use_dark=True),
)

keypoints = [
    ("TL", [255, 0, 0], "pillar", "TR"),
    ("TR", [0, 255, 0], "pillar", "TL"),
    ("BL", [0, 0, 255], "pillar", "BR"),
    ("BR", [255, 255, 0], "pillar", "BL"),
    ("ring", [255, 0, 255], "pillar", ""),
    ("light_BR", [0, 255, 255], "exchange", "light_BL"),
    ("light_TR", [0, 200, 255], "exchange", "light_TL"),
    ("shell_R", [255, 128, 0], "exchange", "shell_L"),
    ("shell_M", [255, 180, 0], "exchange", "shell_M"),
    ("shell_L", [255, 220, 0], "exchange", "shell_R"),
    ("light_TL", [128, 255, 255], "exchange", "light_TR"),
    ("light_BL", [180, 255, 255], "exchange", "light_BR"),
]
skeleton = [("TL", "TR"), ("TR", "BR"), ("BR", "BL"), ("BL", "TL"), ("light_BR", "light_TR"), ("light_TR", "light_TL"), ("light_TL", "light_BL"), ("shell_R", "shell_M"), ("shell_M", "shell_L")]
metainfo = dict(
    dataset_name="exchange12",
    keypoint_info={i: dict(name=n, id=i, color=c, type=t, swap=s) for i, (n, c, t, s) in enumerate(keypoints)},
    skeleton_info={i: dict(link=link, id=i, color=[255, 255, 255]) for i, link in enumerate(skeleton)},
    joint_weights=[1.0] * 12,
    sigmas=[0.02] * 12,
)

albumentations = [
    dict(type="GaussianBlur", blur_limit=(3, 7), p=0.01),
    dict(type="GaussNoise", var_limit=(10.0, 50.0), p=0.2),
    dict(type="MotionBlur", blur_limit=5, p=0.2),
    dict(type="ImageCompression", quality_lower=50, quality_upper=100, p=0.3),
    dict(type="RandomShadow", num_shadows_lower=1, num_shadows_upper=3, p=0.1),
    dict(type="RandomBrightnessContrast", brightness_limit=0.4, contrast_limit=0.4, p=0.3),
    dict(type="ISONoise", color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.1),
    dict(type="Downscale", scale_min=0.5, scale_max=0.9, p=0.2),
    dict(type="RandomFog", fog_coef_lower=0.1, fog_coef_upper=0.4, p=0.1),
    dict(type="CLAHE", clip_limit=4.0, p=0.1),
    dict(type="Sharpen", alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.2),
    dict(type="OneOf", transforms=[dict(type="CoarseDropout", max_holes=1, max_height=0.4, max_width=0.4, min_holes=1, min_height=0.2, min_width=0.2, fill_value=0, p=1.0)], p=0.2),
    dict(type="RandomGamma", gamma_limit=(80, 120), p=0.3),
    dict(type="PixelDropout", dropout_prob=0.01, per_channel=True, p=0.1),
    dict(type="GridDropout", ratio=0.3, unit_size_min=10, unit_size_max=80, p=0.1),
    dict(type="MedianBlur", blur_limit=5, p=0.2),
    dict(type="ChannelDropout", channel_drop_range=(1, 2), fill_value=0, p=0.1),
    dict(type="Posterize", num_bits=(4, 6), p=0.1),
    dict(type="RandomToneCurve", scale=0.3, p=0.1),
]
train_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomBBoxTransform", rotate_factor=30, scale_factor=(0.80, 1.25), shift_factor=0.2, shift_prob=0.1),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="PhotometricDistortion", brightness_delta=60, contrast_range=(0.6, 1.4), saturation_range=(0.6, 1.4), hue_delta=30),
    dict(type="Albumentation", transforms=albumentations),
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
    dict(type="PCKAccuracy", thr=0.1, prefix="pck10"),
]
test_evaluator = val_evaluator
