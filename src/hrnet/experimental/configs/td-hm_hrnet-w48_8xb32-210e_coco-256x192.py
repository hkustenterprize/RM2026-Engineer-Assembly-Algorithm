_base_ = ["mmpose::_base_/default_runtime.py"]

# ──────────────────────────────────────────────────────────────────────────────
#  Paths  — edit these two lines before training
# ──────────────────────────────────────────────────────────────────────────────
# data_root  : root of the YOLO-format dataset produced by compose_dataset.py
#              must contain  images/train/   images/val/
# ann_root   : directory that holds pillar_train.json / pillar_val.json
#              produced by prepare_data.py
#              simplest: run  prepare_data.py <data_root> --output <data_root>/annotations
#              so that ann_root == data_root + 'annotations/'

"""
bash /data/datasets/zguobd/mmpose/tools/dist_train.sh \
  /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/model_configs/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
  4 \
  --work-dir /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/runs/hrnet_v7.0

PYTHONPATH=/data/datasets/zguobd/mmpose:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 \
/data/datasets/zguobd/miniconda3/envs/mm/bin/python \
/data/datasets/zguobd/mmpose/tools/train.py \
/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/model_configs/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
--work-dir /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/runs/hrnet_v7.3
# --resume /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/runs/hrnet_v7.0/epoch_60.pth
"""

data_root = "/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/dataset_v7.0/"
ann_root = "/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/dataset_v7.0_annotations/"

# load_from = '/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/runs/pillar_hrnet_w48/best_coco_AP_epoch_110.pth'
# resume = False

# ──────────────────────────────────────────────────────────────────────────────
#  Runtime
# ──────────────────────────────────────────────────────────────────────────────
train_cfg = dict(max_epochs=150, val_interval=1)

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=5,
        save_best="coco/AP",
        rule="greater",
    )
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

# ──────────────────────────────────────────────────────────────────────────────
#  Optimizer & LR schedule
# ──────────────────────────────────────────────────────────────────────────────
optim_wrapper = dict(
    optimizer=dict(
        type="SGD",
        lr=5e-3,  # higher base lr for faster convergence
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True,
    )
)

param_scheduler = [
    dict(
        type="LinearLR",
        begin=0,
        end=500,
        start_factor=0.1,  # initial lr = 0.1 * 5e-3 = 5e-4 (reasonable for fine-tuning)
        by_epoch=False,
    ),
    dict(
        type="MultiStepLR",
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True,
    ),
]

# 4 GPU × 64 = 256 total_bs; base matches actual so no auto-scaling happens
auto_scale_lr = dict(base_batch_size=256)

# ──────────────────────────────────────────────────────────────────────────────
#  Codec  (heatmap encoder/decoder)
#  input_size  : (W, H) fed into the network after TopdownAffine crop
#  heatmap_size: (W, H) of the output heatmap  (input / 4)
# ──────────────────────────────────────────────────────────────────────────────
codec = dict(
    type="MSRAHeatmap",
    input_size=(256, 256),
    heatmap_size=(64, 64),
    sigma=2,
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
        type="HRNet",
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block="BOTTLENECK",
                num_blocks=(4,),
                num_channels=(64,),
            ),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block="BASIC",
                num_blocks=(4, 4),
                num_channels=(48, 96),
            ),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block="BASIC",
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192),
            ),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block="BASIC",
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384),
            ),
        ),
        # HRNet-W48 backbone pretrained on COCO (backbone weights only, no head)
        init_cfg=dict(
            type="Pretrained",
            checkpoint="https://download.openmmlab.com/mmpose/"
            "pretrain_models/hrnet_w48-8ef0771d.pth",
        ),
    ),
    head=dict(
        type="HeatmapHead",
        in_channels=48,  # HRNet-W48 high-res output channels
        out_channels=5,  # 5 pillar keypoints: TL TR BL BR ring
        deconv_out_channels=None,
        loss=dict(type="KeypointMSELoss", use_target_weight=True),
        decoder=codec,
    ),
    test_cfg=dict(
        flip_test=False,
        flip_mode="heatmap",
        shift_heatmap=True,
        # DARK: distribution-aware sub-pixel refinement around heatmap peak
        # Increases keypoint precision from ±stride/2 to ~±0.5px at no extra cost.
        use_dark=True,
    ),
)

# ──────────────────────────────────────────────────────────────────────────────
#  Dataset metainfo  — 5 pillar keypoints
#  swap pairs: TL↔TR  BL↔BR  ring→ring  (used by RandomFlip)
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
train_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    # dict(type="RandomFlip", direction="horizontal"),
    dict(
        type="RandomBBoxTransform",
        rotate_factor=45,
        scale_factor=(1.0, 4.0),
        shift_factor=0.25,
        shift_prob=0.5,
    ),  # 平移: 最多偏移 25% bbox 宽高，50% 概率触发
    dict(type="RandomAffine", degrees=45, translate=0.2, scale=(0.9, 4.0)),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(
        type="PhotometricDistortion",
        brightness_delta=60,
        contrast_range=(0.4, 1.6),
        saturation_range=(0.4, 1.6),
        hue_delta=30,
    ),
    dict(
        type="Albumentation",  # mmpose 注册名，无 's'
        transforms=[
            # albumentations 1.3.x API
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
            dict(type="ISONoise", color_shift=(0.05, 0.1), intensity=(0.1, 0.6), p=0.2),
            dict(type="Downscale", scale_min=0.5, scale_max=0.9, p=0.2),
            dict(type="RandomFog", fog_coef_lower=0.3, fog_coef_upper=0.5, p=0.3),
            dict(type="CLAHE", clip_limit=5.0, p=0.3),
            dict(type="Sharpen", alpha=(0.3, 0.5), lightness=(0.5, 1.0), p=0.3),
            dict(
                type="OneOf",
                transforms=[
                    dict(
                        type="CoarseDropout",
                        max_holes=6,
                        max_height=64,
                        max_width=64,
                        min_holes=1,
                        min_height=16,
                        min_width=16,
                        fill_value=0,
                        p=1.0,
                    ),
                    dict(
                        type="CoarseDropout",
                        max_holes=6,
                        max_height=64,
                        max_width=64,
                        min_holes=1,
                        min_height=16,
                        min_width=16,
                        fill_value=128,
                        p=1.0,
                    ),
                    dict(
                        type="CoarseDropout",
                        max_holes=6,
                        max_height=64,
                        max_width=64,
                        min_holes=1,
                        min_height=16,
                        min_width=16,
                        fill_value=255,
                        p=1.0,
                    ),
                ],
                p=0.3,
            ),
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
    dict(
        type="CocoMetric",
        ann_file=ann_root + "pillar_val.json",
    ),
    # PCK@0.05 and PCK@0.10 — fraction of bbox diagonal within which prediction is "correct"
    dict(type="PCKAccuracy", thr=0.05, prefix="pck05"),
    dict(type="PCKAccuracy", thr=0.10, prefix="pck10"),
]
test_evaluator = val_evaluator

# ──────────────────────────────────────────────────────────────────────────────
#  Visualizer  — enable TensorBoard logging
# ──────────────────────────────────────────────────────────────────────────────
vis_backends = [
    dict(type="LocalVisBackend"),
    dict(type="TensorboardVisBackend"),
]
visualizer = dict(
    type="PoseLocalVisualizer",
    vis_backends=vis_backends,
    name="visualizer",
)
