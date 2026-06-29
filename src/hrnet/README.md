# HRNet / LiteHRNet Keypoint Pipeline

装配站 12 点关键点训练与推理，基于 **MMPose** 的 top-down heatmap 流程。

当前公开主线只保留两条训练配置：

- `td-hm_litehrnet18_exchange12_v11.0.py`
- `td-hm_litehrnet30_exchange12_v11.0.py`

两条配置都对应同一个任务定义：

- 输入：YOLO 检测框裁剪后的 `256x256` 图像块
- 输出：12 个关键点热力图
- 附加分支：每个关键点的 `in-frame` 可见性分类

非主线实验配置、旧 RTMPose / RTMO 路线和几何损失相关文件已移到 `experimental/`。

## Directory Layout

```text
hrnet/
├── data_process/
│   ├── prepare_data_new.py
│   └── viz_data.py
├── model_configs/
│   ├── td-hm_litehrnet18_exchange12_v11.0.py
│   └── td-hm_litehrnet30_exchange12_v11.0.py
├── scripts/
│   ├── prepare_data.sh
│   ├── train_litehrnet.sh
│   ├── inference.sh
│   ├── inference_video.sh
│   └── vis_data.sh
├── train/
│   ├── pillar_models.py
│   ├── inference.py
│   └── inference_video.py
├── export_hrnet_openvino.py
└── experimental/
```

## Data Preparation

`prepare_data_new.py` 将 YOLOPose 风格数据集转换为 MMPose 使用的 COCO keypoint JSON。

输入目录约定：

```text
dataset_root/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

当前主线使用：

- `--mode exchange12`
- pillar 行与 exchange 行成对组合成 12 点监督
- 保留 `keypoints_raw_visibility` 与 `keypoints_in_frame` 字段

示例：

```bash
python data_process/prepare_data_new.py /path/to/yolo_pose_dataset \
  --output /path/to/exchange12_annotations \
  --mode exchange12
```

## Training

训练依赖：

- `mmpose`
- `mmengine`
- `mmcv`
- `torch`
- `albumentations`

需要把 `train/` 加入 `PYTHONPATH`，让 MMPose 能注册 `PillarHeatmapHeadWithVis`：

```bash
export PYTHONPATH="$(pwd)/train:/path/to/mmpose:${PYTHONPATH}"
```

训练入口：

```bash
bash scripts/train_litehrnet.sh td-hm_litehrnet18_exchange12_v11.0.py
bash scripts/train_litehrnet.sh td-hm_litehrnet30_exchange12_v11.0.py
```

## Inference

公开主线只保留两阶段推理：

1. YOLO 检测模型输出装配站框
2. LiteHRNet 在裁剪块上回归 12 个关键点

图片推理：

```bash
bash scripts/inference.sh image1.jpg image2.jpg
```

视频推理：

```bash
SOURCE=/path/to/input.mp4 OUTPUT=/path/to/output.mp4 \
bash scripts/inference_video.sh
```

默认情况下，`bbox_class_id=1`，对应 exchange bbox。

## OpenVINO Export

`export_hrnet_openvino.py` 保留为主线工具，用于导出 heatmap 子图：

```bash
python export_hrnet_openvino.py \
  --config model_configs/td-hm_litehrnet18_exchange12_v11.0.py \
  --checkpoint /path/to/best.pth \
  --output-dir /path/to/openvino_out
```

导出的 IR 只包含：

- `RGB patch -> heatmaps`

以下部分仍在宿主侧处理：

- YOLO 检测框
- top-down 裁剪与仿射变换
- heatmap 解码与坐标还原

## Notes

- 两条主线配置目前仍保留原始训练路径占位，需要按你的本地数据目录修改 `data_root`、`ann_root` 和 `load_from`。
- 如果只需要公开最小训练配置，下一步可以继续把两个 config 中与本地路径强耦合的字段改为更中性的占位路径。
