_base_ = ["mmpose::_base_/default_runtime.py"]

default_scope = "mmpose"
load_from = "https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-s_8xb32-600e_coco-640x640-8db55a59_20231211.pth"
resume = False

# Runtime: shorter schedule for synthetic exchange12 data, while preserving
# the official RTMO two-stage schedule proportions.
max_epochs = 120
rtmo_switch_epoch = 56
stage2_epoch = 100
train_cfg = dict(max_epochs=max_epochs, val_interval=2, dynamic_intervals=[(stage2_epoch, 1)])
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(base_batch_size=256)

default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=10, max_keep_ckpts=3)
)

optim_wrapper = dict(
    type="OptimWrapper",
    constructor="ForceDefaultOptimWrapperConstructor",
    optimizer=dict(type="AdamW", lr=0.00175, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0,
        bias_decay_mult=0,
        bypass_duplicate=True,
        force_default_settings=True,
        custom_keys=dict({"neck.encoder": dict(lr_mult=0.05)}),
    ),
    clip_grad=dict(max_norm=0.1, norm_type=2),
)

param_scheduler = [
    dict(type="QuadraticWarmupLR", by_epoch=True, begin=0, end=5, convert_to_iter_based=True),
    dict(type="CosineAnnealingLR", eta_min=0.0000875, begin=5, T_max=rtmo_switch_epoch - 5, end=rtmo_switch_epoch, by_epoch=True, convert_to_iter_based=True),
    dict(type="ConstantLR", by_epoch=True, factor=2.5, begin=rtmo_switch_epoch, end=rtmo_switch_epoch + 1),
    dict(type="CosineAnnealingLR", eta_min=0.0000875, begin=rtmo_switch_epoch + 1, T_max=stage2_epoch - rtmo_switch_epoch - 1, end=stage2_epoch, by_epoch=True, convert_to_iter_based=True),
    dict(type="ConstantLR", by_epoch=True, factor=1, begin=stage2_epoch, end=max_epochs),
]

# Data.
input_size = (640, 640)
num_keypoints = 12
data_mode = "bottomup"
data_root = "/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/dataset_v11.0/"
ann_root = "/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/dataset_v11.0_annotations/"
metafile = "/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/model_configs/exchange12_meta.py"
codec = dict(type="YOLOXPoseAnnotationProcessor", input_size=input_size)

# We intentionally do not use Mosaic/MixUp because exchange has usually one
# target and compositing creates unrealistic multi-station scenes.
train_pipeline_stage1 = [
    dict(type="LoadImage", backend_args=None),
    dict(
        type="BottomupRandomAffine",
        input_size=input_size,
        shift_factor=0.1,
        rotate_factor=10,
        scale_factor=(0.75, 1.0),
        pad_val=114,
        distribution="uniform",
        transform_mode="affine",
        bbox_keep_corner=False,
        clip_border=True,
    ),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip"),
    dict(type="FilterAnnotations", by_kpt=True, by_box=True, keep_empty=False),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]

train_pipeline_stage2 = [
    dict(type="LoadImage"),
    dict(
        type="BottomupRandomAffine",
        input_size=input_size,
        shift_prob=0,
        rotate_prob=0,
        scale_prob=0,
        scale_type="long",
        pad_val=(114, 114, 114),
        bbox_keep_corner=False,
        clip_border=True,
    ),
    dict(type="YOLOXHSVRandomAug"),
    dict(type="RandomFlip"),
    dict(type="BottomupGetHeatmapMask", get_invalid=True),
    dict(type="FilterAnnotations", by_kpt=True, by_box=True, keep_empty=False),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]

dataset_exchange12 = dict(
    type="CocoDataset",
    data_root=data_root,
    data_mode=data_mode,
    ann_file=ann_root + "exchange12_train.json",
    data_prefix=dict(img=data_root + "images/train/"),
    metainfo=dict(from_file=metafile),
    pipeline=train_pipeline_stage1,
)

train_dataloader = dict(
    batch_size=28,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dataset_exchange12,
)

val_pipeline = [
    dict(type="LoadImage"),
    dict(type="BottomupResize", input_size=input_size, pad_val=(114, 114, 114)),
    dict(
        type="PackPoseInputs",
        meta_keys=("id", "img_id", "img_path", "ori_shape", "img_shape", "input_size", "input_center", "input_scale"),
    ),
]

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        data_mode=data_mode,
        ann_file=ann_root + "exchange12_val.json",
        data_prefix=dict(img=data_root + "images/val/"),
        metainfo=dict(from_file=metafile),
        test_mode=True,
        pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=ann_root + "exchange12_val.json",
    score_mode="bbox",
    nms_mode="none",
)
test_evaluator = val_evaluator

custom_hooks = [
    dict(
        type="YOLOXPoseModeSwitchHook",
        num_last_epochs=max_epochs - stage2_epoch,
        new_train_pipeline=train_pipeline_stage2,
        priority=48,
    ),
    dict(
        type="RTMOModeSwitchHook",
        epoch_attributes={
            rtmo_switch_epoch: {
                "proxy_target_cc": True,
                "loss_mle.loss_weight": 5.0,
                "loss_oks.loss_weight": 10.0,
            },
        },
        priority=48,
    ),
    dict(type="SyncNormHook", priority=48),
    dict(
        type="EMAHook",
        ema_type="ExpMomentumEMA",
        momentum=0.0002,
        update_buffers=True,
        strict_load=False,
        priority=49,
    ),
]

# Model: official RTMO-s backbone/neck/head, only num_keypoints/metainfo changed.
widen_factor = 0.5
deepen_factor = 0.33

model = dict(
    type="BottomupPoseEstimator",
    init_cfg=dict(
        type="Kaiming",
        layer="Conv2d",
        a=2.23606797749979,
        distribution="uniform",
        mode="fan_in",
        nonlinearity="leaky_relu",
    ),
    data_preprocessor=dict(
        type="PoseDataPreprocessor",
        pad_size_divisor=32,
        mean=[0, 0, 0],
        std=[1, 1, 1],
        batch_augments=[
            dict(
                type="BatchSyncRandomResize",
                random_size_range=(480, 800),
                size_divisor=32,
                interval=1,
            ),
        ],
    ),
    backbone=dict(
        type="CSPDarknet",
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        out_indices=(2, 3, 4),
        spp_kernal_sizes=(5, 9, 13),
        norm_cfg=dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg=dict(type="Swish"),
    ),
    neck=dict(
        type="HybridEncoder",
        in_channels=[128, 256, 512],
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        hidden_dim=256,
        output_indices=[1, 2],
        encoder_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=1024, ffn_drop=0.0, act_cfg=dict(type="GELU")),
        ),
        projector=dict(
            type="ChannelMapper",
            in_channels=[256, 256],
            kernel_size=1,
            out_channels=256,
            act_cfg=None,
            norm_cfg=dict(type="BN"),
            num_outs=2,
        ),
    ),
    head=dict(
        type="RTMOHead",
        num_keypoints=num_keypoints,
        featmap_strides=(16, 32),
        head_module_cfg=dict(
            num_classes=1,
            in_channels=256,
            cls_feat_channels=256,
            channels_per_group=36,
            pose_vec_channels=256,
            widen_factor=widen_factor,
            stacked_convs=2,
            norm_cfg=dict(type="BN", momentum=0.03, eps=0.001),
            act_cfg=dict(type="Swish"),
        ),
        assigner=dict(
            type="SimOTAAssigner",
            dynamic_k_indicator="oks",
            oks_calculator=dict(type="PoseOKS", metainfo=metafile),
            use_keypoints_for_center=True,
        ),
        prior_generator=dict(type="MlvlPointGenerator", centralize_points=True, strides=[16, 32]),
        dcc_cfg=dict(
            in_channels=256,
            feat_channels=128,
            num_bins=(192, 256),
            spe_channels=128,
            gau_cfg=dict(s=128, expansion_factor=2, dropout_rate=0.0, drop_path=0.0, act_fn="SiLU", pos_enc="add"),
        ),
        overlaps_power=0.5,
        loss_cls=dict(type="VariFocalLoss", reduction="sum", use_target_weight=True, loss_weight=1.0),
        loss_bbox=dict(type="IoULoss", mode="square", eps=1e-16, reduction="sum", loss_weight=5.0),
        loss_oks=dict(type="OKSLoss", reduction="none", metainfo=metafile, loss_weight=30.0),
        loss_vis=dict(type="BCELoss", use_target_weight=True, reduction="mean", loss_weight=0.0),
        loss_mle=dict(type="MLECCLoss", use_target_weight=True, loss_weight=1.0),
        loss_bbox_aux=dict(type="L1Loss", reduction="sum", loss_weight=1.0),
    ),
    test_cfg=dict(input_size=input_size, score_thr=0.1, nms_thr=0.65),
)
