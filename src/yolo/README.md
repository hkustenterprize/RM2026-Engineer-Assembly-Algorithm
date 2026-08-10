# YOLO Pose and Detection

本目录提供基于 Ultralytics 的两个训练入口：YOLO26-s-pose 的 12 关键点姿态检测，以及 YOLO26-s 的目标框检测。检测模型作为 top-down LiteHRNet 流程的前级检测器，姿态模型作为 bottom-up 关键点方案和对照实现。

## 目录结构

```text
yolo/
├── configs/
│   ├── train_pose.yaml
│   ├── train_detect.yaml
│   ├── exchange_pose_dataset.yaml
│   └── exchange_detect_dataset.yaml
├── train/
│   ├── train.py          # 统一训练入口
│   ├── augmentations.py  # 配置驱动的图像和标签增强
│   └── trainers.py       # pose/detect trainer 适配层
├── pose2detect.py        # YOLO Pose 数据转检测数据
└── export_openvino.py    # YOLO 权重导出 OpenVINO
```

两个训练 YAML 都是自包含配置。`train_pose.yaml` 定义关键点损失和姿态增强；`train_detect.yaml` 定义检测损失、目标框裁剪、加权裁剪选择以及分组 Albumentations 增强。

增强项使用 `src/utils.py` 中的配置驱动对象工厂构造，YAML 中的 `_class_name` 对应目标类的完整名称，其余字段直接对应构造函数参数：

```yaml
GaussianBlur:
  _class_name: albumentations.GaussianBlur
  blur_limit: [3, 5]
  sigma_limit: [0.5, 3.0]
  p: 0.1
```

## 数据集

公开数据集见 [hkustenterprize/RM26_engineer_exchange](https://huggingface.co/datasets/hkustenterprize/RM26_engineer_exchange)。训练入口使用标准 YOLO 目录：

```text
dataset_root/
├── images/{train,val}/
├── labels/{train,val}/
└── dataset.yaml
```

姿态数据集包含 `pillar`、`exchange` 两类和 12 个关键点。检测训练只使用同一数据集中的目标框标注；如需要单独的检测数据集，可以执行：

```bash
python pose2detect.py \
  --source /absolute/path/to/exchange_pose \
  --output /absolute/path/to/exchange_detect
```

## 训练

YOLO 训练复用 `src/hrnet` 的 uv 环境：

```bash
cd public_archive
uv run --project src/hrnet python src/yolo/train/train.py \
  --config src/yolo/configs/train_pose.yaml
```

```bash
cd public_archive
uv run --project src/hrnet python src/yolo/train/train.py \
  --config src/yolo/configs/train_detect.yaml
```

数据、模型和训练参数可以通过命令行覆盖：

```bash
uv run --project src/hrnet python src/yolo/train/train.py \
  --config src/yolo/configs/train_pose.yaml \
  --data /absolute/path/to/dataset.yaml \
  --model /absolute/path/to/yolo26s-pose.pt \
  --epochs 1 --batch 8 --imgsz 256 --workers 0 \
  --name smoke_pose
```

训练结果默认写入 `src/yolo/runs/pose/` 或 `src/yolo/runs/detect/`。配置中的 `plots: true` 会让 Ultralytics 生成 `train_batch*.jpg`、`labels.jpg`、`results.png`、验证集标签/预测图和 `weights/`。

## OpenVINO 导出

```bash
uv run --project src/hrnet python src/yolo/export_openvino.py \
  --weights /absolute/path/to/best.pt \
  --imgsz 640
```

训练入口也支持在训练完成后设置 `export_openvino: true` 自动导出。
