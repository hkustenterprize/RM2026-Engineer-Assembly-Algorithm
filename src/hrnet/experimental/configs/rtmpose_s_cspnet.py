_base_ = ["mmpose::_base_/default_runtime.py"]


data_root = "/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/dataset_v7.0/"
ann_root = "/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/dataset_v7.0_annotations/"

# runtime
max_epochs = 200
stage2_num_epochs = 30
base_lr = 4e-3

train_cfg = dict(max_epochs=max_epochs, val_interval=10)
randomness = dict(seed=21)

# optimizer
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
)

# learning rate
param_scheduler = [
    dict(type="LinearLR", start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),
    dict(
        # use cosine lr from 210 to 420 epoch
        type="CosineAnnealingLR",
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=256)

# codec settings
codec = dict(
    type="SimCCLabel",
    input_size=(256, 256),
    sigma=(5.66, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False,
)

# model settings
model = dict(
    type="TopdownPoseEstimator",
    data_preprocessor=dict(
        type="PoseDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        _scope_="mmdet",
        type="CSPNeXt",
        arch="P5",
        expand_ratio=0.5,
        deepen_factor=0.33,
        widen_factor=0.5,
        out_indices=(4,),
        channel_attention=True,
        norm_cfg=dict(type="SyncBN"),
        act_cfg=dict(type="SiLU"),
        init_cfg=dict(
            type="Pretrained",
            prefix="backbone.",
            checkpoint="https://download.openmmlab.com/mmpose/v1/projects/"
            "rtmposev1/cspnext-s_udp-aic-coco_210e-256x192-92f5a029_20230130.pth",  # noqa
        ),
    ),
    head=dict(
        type="RTMCCHead",
        in_channels=512,
        out_channels=5,
        input_size=codec["input_size"],
        in_featuremap_size=tuple([s // 32 for s in codec["input_size"]]),
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
            type="KLDiscretLoss", use_target_weight=True, beta=10.0, label_softmax=True
        ),
        decoder=codec,
    ),
    test_cfg=dict(
        flip_test=True,
    ),
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

# base dataset settings
dataset_type = "CocoDataset"
data_mode = "topdown"
# data_root 已在文件顶部定义为绝对路径，此处不覆盖

backend_args = dict(backend="local")
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         f'{data_root}': 's3://openmmlab/datasets/',
#         f'{data_root}': 's3://openmmlab/datasets/'
#     }))
#
train_pipeline = [
    dict(type="LoadImage", backend_args=backend_args),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="RandomHalfBody"),
    dict(
        type="RandomBBoxTransform",
        scale_factor=[0.9, 2.0],
        rotate_factor=80,
    ),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="mmdet.YOLOXHSVRandomAug"),
    dict(
        type="Albumentation",
        transforms=[
            dict(type="Blur", p=0.1),
            dict(type="MedianBlur", p=0.1),
            dict(
                type="CoarseDropout",
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=1.0,
            ),
        ],
    ),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]
val_pipeline = [
    dict(type="LoadImage", backend_args=backend_args),
    dict(type="GetBBoxCenterScale"),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="PackPoseInputs"),
]

train_pipeline_stage2 = [
    dict(type="LoadImage", backend_args=backend_args),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="RandomHalfBody"),
    dict(
        type="RandomBBoxTransform",
        shift_factor=0.0,
        scale_factor=[0.9, 2.0],
        rotate_factor=60,
    ),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="mmdet.YOLOXHSVRandomAug"),
    dict(
        type="Albumentation",
        transforms=[
            dict(type="Blur", p=0.1),
            dict(type="MedianBlur", p=0.1),
            dict(
                type="CoarseDropout",
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5,
            ),
        ],
    ),
    dict(type="GenerateTarget", encoder=codec),
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
