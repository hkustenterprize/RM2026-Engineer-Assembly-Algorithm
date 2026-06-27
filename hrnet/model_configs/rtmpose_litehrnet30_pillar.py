import sys as _sys, os as _os; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); del _sys, _os
_base_ = ["mmpose::_base_/default_runtime.py"]

# 注册自定义模块（PillarRTMCCHead，即使 geo_loss=None 也需导入）
custom_imports = dict(
    imports=["pillar_models"],
    allow_failed_imports=False,
)

# ──────────────────────────────────────────────────────────────────────────────
#  RTMPose-Lite  — PillarRTMCCHead (GAU self-attention) baseline
#  对照组：无几何一致性损失，仅 KLDiscretLoss
#
#  与 rtmpose_lite_pillar_geo.py 的唯一差异：head 中 geo_loss=None
#  （4路消融实验：HeatmapHead vs RTMCCHead，无/有 geo_loss）
#
# 训练命令 (4 卡):
#   PYTHONPATH=/data/datasets/zguobd/mmpose:$PYTHONPATH \
#   CUDA_VISIBLE_DEVICES=4,5,6,7 \
#   /data/datasets/zguobd/miniconda3/envs/mm/bin/python \
#     -m torch.distributed.launch --nproc_per_node=4 --master_port=29501 \
#     /data/datasets/zguobd/mmpose/tools/train.py \
#     cv/nn/hrnet/rtmpose_litehrnet30_pillar.py \
#     --launcher pytorch \
#     --work-dir cv/nn/hrnet/runs/rtmpose_litehrnet30_v2.0
# ──────────────────────────────────────────────────────────────────────────────

data_root = "/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/dataset_v7.0/"
ann_root = "/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/dataset_v7.0_annotations/"

load_from = "/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/runs/litehrnet30_v1.0_Adam_128x128/epoch_10.pth"

# ──────────────────────────────────────────────────────────────────────────────
#  Runtime
# ──────────────────────────────────────────────────────────────────────────────
max_epochs = 420
train_cfg = dict(max_epochs=max_epochs, val_interval=5)

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=10,
        save_best="coco/AP",
        rule="greater",
    )
)

custom_hooks = [
    dict(
        type="EarlyStoppingHook",
        monitor="coco/AP",
        rule="greater",
        patience=80,
        min_delta=0.001,
    )
]

# ──────────────────────────────────────────────────────────────────────────────
#  Optimizer & LR
# ──────────────────────────────────────────────────────────────────────────────
base_lr = 4e-3

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),
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

auto_scale_lr = dict(base_batch_size=1024)

# ──────────────────────────────────────────────────────────────────────────────
#  Codec
# ──────────────────────────────────────────────────────────────────────────────
codec = dict(
    type="SimCCLabel",
    input_size=(256, 256),
    sigma=(5.66, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False,
)

# ──────────────────────────────────────────────────────────────────────────────
#  Model
# ──────────────────────────────────────────────────────────────────────────────
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
                num_modules=(3, 8, 3),
                num_branches=(2, 3, 4),
                num_blocks=(2, 2, 2),
                module_type=("LITE", "LITE", "LITE"),
                with_fuse=(True, True, True),
                reduce_ratios=(8, 8, 8),
                num_channels=(
                    (40, 80),
                    (40, 80, 160),
                    (40, 80, 160, 320),
                ),
            ),
            with_head=True,
        ),
    ),
    head=dict(
        type="PillarRTMCCHead",
        spatial_pool_size=(8, 8),
        in_channels=40,
        out_channels=5,
        input_size=codec["input_size"],
        in_featuremap_size=(8, 8),
        simcc_split_ratio=codec["simcc_split_ratio"],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.0,
            drop_path=0.0,
            act_fn="SiLU",
            use_rel_bias=False,
            pos_enc=False,
        ),
        loss=dict(
            type="KLDiscretLoss",
            use_target_weight=True,
            beta=10.0,
            label_softmax=True,
        ),
        decoder=codec,
        geo_loss=None,  # 无几何一致性损失（baseline）
    ),
    test_cfg=dict(flip_test=True),
)

# ──────────────────────────────────────────────────────────────────────────────
#  Dataset metainfo
# ──────────────────────────────────────────────────────────────────────────────
metainfo = dict(
    dataset_name="pillar5",
    keypoint_info={
        0: dict(name="TL", id=0, color=[255, 0, 0], type="upper", swap="TR"),
        1: dict(name="TR", id=1, color=[0, 255, 0], type="upper", swap="TL"),
        2: dict(name="BL", id=2, color=[0, 0, 255], type="lower", swap="BR"),
        3: dict(name="BR", id=3, color=[255, 255, 0], type="lower", swap="BL"),
        4: dict(name="ring", id=4, color=[255, 0, 255], type="", swap=""),
    },
    skeleton_info={
        0: dict(link=("TL", "TR"), id=0, color=[255, 255, 255]),
        1: dict(link=("TR", "BR"), id=1, color=[255, 255, 255]),
        2: dict(link=("BR", "BL"), id=2, color=[255, 255, 255]),
        3: dict(link=("BL", "TL"), id=3, color=[255, 255, 255]),
    },
    joint_weights=[1.0, 1.0, 1.0, 1.0, 1.0],
    sigmas=[0.02, 0.02, 0.02, 0.02, 0.02],
)

# ──────────────────────────────────────────────────────────────────────────────
#  Data pipelines
# ──────────────────────────────────────────────────────────────────────────────

# ── 训练阶段一：合成数据预训练，开大增强（当前激活）────────────────────────────
train_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(
        type="RandomBBoxTransform",
        rotate_factor=30,
        scale_factor=(1.0, 4.0),   # 合成数据场景多样，允许大倍率缩放
        shift_factor=0.3,
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
            dict(type="RandomShadow", num_shadows_lower=1, num_shadows_upper=3, p=0.2),
            dict(
                type="RandomBrightnessContrast",
                brightness_limit=0.5,
                contrast_limit=0.5,
                p=0.3,
            ),
            dict(
                type="ISONoise", color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2
            ),
            dict(type="Downscale", scale_min=0.5, scale_max=0.9, p=0.2),
            dict(type="RandomFog", fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
            dict(type="CLAHE", clip_limit=4.0, p=0.3),
            dict(type="Sharpen", alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.2),
            dict(
                type="OneOf",
                transforms=[
                    dict(
                        type="CoarseDropout",
                        max_holes=6, max_height=64, max_width=64,
                        min_holes=1, min_height=16, min_width=16,
                        fill_value=0, p=1.0,
                    ),
                    dict(
                        type="CoarseDropout",
                        max_holes=6, max_height=64, max_width=64,
                        min_holes=1, min_height=16, min_width=16,
                        fill_value=128, p=1.0,
                    ),
                    dict(
                        type="CoarseDropout",
                        max_holes=6, max_height=64, max_width=64,
                        min_holes=1, min_height=16, min_width=16,
                        fill_value=255, p=1.0,
                    ),
                ],
                p=0.3,
            ),
        ],
    ),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]

# ── 训练阶段二：真实数据微调，降低增强（注释保留，微调时切换）─────────────────────
# train_pipeline = [
#     dict(type="LoadImage"),
#     dict(type="GetBBoxCenterScale"),
#     dict(
#         type="RandomBBoxTransform",
#         rotate_factor=30,
#         scale_factor=(0.6, 1.4),   # 真实数据不做大倍率缩放
#         shift_factor=0.2,
#     ),
#     dict(type="TopdownAffine", input_size=codec["input_size"]),
#     dict(type="mmdet.YOLOXHSVRandomAug"),
#     dict(
#         type="Albumentation",
#         transforms=[
#             dict(type="Blur", blur_limit=3, p=0.1),
#             dict(type="MedianBlur", blur_limit=3, p=0.1),
#             dict(
#                 type="CoarseDropout",
#                 max_holes=1, max_height=0.4, max_width=0.4,
#                 min_holes=1, min_height=0.2, min_width=0.2,
#                 p=0.5,
#             ),
#             dict(type="ImageCompression", quality_lower=60, quality_upper=100, p=0.2),
#             dict(type="RandomShadow", num_shadows_lower=1, num_shadows_upper=2, p=0.15),
#             dict(type="GaussNoise", var_limit=(5.0, 25.0), p=0.15),
#         ],
#     ),
#     dict(type="GenerateTarget", encoder=codec),
#     dict(type="PackPoseInputs"),
# ]

val_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="PackPoseInputs"),
]

# ──────────────────────────────────────────────────────────────────────────────
#  Data loaders
# ──────────────────────────────────────────────────────────────────────────────
dataset_type = "CocoDataset"
data_mode = "topdown"

train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file=ann_root + "pillar_train.json",
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
        ann_file=ann_root + "pillar_val.json",
        data_prefix=dict(img=data_root + "images/val/"),
        metainfo=metainfo,
        test_mode=True,
        pipeline=val_pipeline,
    ),
)

test_dataloader = val_dataloader

# ──────────────────────────────────────────────────────────────────────────────
#  Evaluators
# ──────────────────────────────────────────────────────────────────────────────
val_evaluator = [
    dict(type="CocoMetric", ann_file=ann_root + "pillar_val.json"),
    dict(type="PCKAccuracy", thr=0.05, prefix="pck05"),
    dict(type="PCKAccuracy", thr=0.10, prefix="pck10"),
]
test_evaluator = val_evaluator

# ──────────────────────────────────────────────────────────────────────────────
#  Visualizer
# ──────────────────────────────────────────────────────────────────────────────
vis_backends = [dict(type="LocalVisBackend"), dict(type="TensorboardVisBackend")]
visualizer = dict(
    type="PoseLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)
