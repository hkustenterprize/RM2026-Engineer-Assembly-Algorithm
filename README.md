# 香港科技大学 ENTERPRIZE 战队 RoboMaster 2026 工程辅助装配算法

这是香港科技大学 ENTERPRIZE 战队 RoboMaster 2026 赛季工程辅助装配算法的开源仓库。项目面向能量单元装配任务，提供合成数据生成、视觉模型训练、关键点位姿估计、运动规划与仿真验证相关的代码和配置。

## 项目内容

装配流程采用半自动方式。上位机根据 RGB 图像估计装配站位姿，负责接近规划、受约束动作段规划和恢复规划；插入阶段由操作手通过自定义控制器完成。仓库中的主要算法链路包括：

```text
Blender 场景与渲染
        |
        v
目标素材合成与 YOLO 数据集生成
        |
        +--> YOLO26-s 姿态模型：目标框 + 12 个关键点
        |
        +--> YOLO26-s 检测模型：目标框
                              |
                              v
                    LiteHRNet：局部关键点
                              |
                              v
                         PnP 位姿估计
                              |
                              v
                   规划、执行与 MuJoCo 验证
```

技术报告见 [RM2026 Engineer Assembly Algorithm](doc/RM2026-Engineer-Assembly-Algorithm.pdf)。

## 仓库结构

```text
public_archive/
├── doc/                         # 技术报告
├── src/
│   ├── blender/                 # Blender 渲染与数据合成
│   │   ├── configs/             # 场景和渲染配置示例
│   │   ├── pipeline/            # 渲染、合成、可视化脚本
│   │   └── setup_blender_env.sh
│   ├── yolo/                   # Ultralytics YOLO 训练与导出
│   │   ├── configs/             # 训练配置和数据集配置
│   │   ├── train/               # 统一训练入口、增强和 trainer
│   │   ├── pose2detect.py       # YOLO Pose 转检测数据集
│   │   └── export_openvino.py
│   ├── hrnet/                  # MMPose LiteHRNet 训练、推理和导出
│   │   ├── configs/             # LiteHRNet-18/30 主线配置
│   │   ├── data/                # YOLO 到 COCO 的转换和可视化
│   │   ├── tools/               # 训练、推理、导出和环境检查
│   │   └── experimental/        # 非主线实验代码
│   └── utils.py                # 配置驱动的对象构造工具
├── LICENSE
└── README.md
```

`src/hrnet` 提供当前 Python 环境和 `uv.lock`。YOLO 训练复用该环境，但训练入口和配置保留在 `src/yolo`，便于区分两条视觉模型主线。

## 环境准备

LiteHRNet 和 YOLO 共用 `src/hrnet` 下的 uv 环境。首次安装：

```bash
cd public_archive/src/hrnet
bash scripts/setup_env.sh
```

检查环境：

```bash
cd public_archive/src/hrnet
uv run python tools/check_environment.py
```

Blender 渲染使用 Blender 自带 Python 和 Cycles。Blender 不属于 `src/hrnet` 的 Python 环境；请先安装 Blender，再根据本机路径执行：

```bash
cd public_archive/src/blender
bash setup_blender_env.sh
```

训练时由 PyTorch wheel 提供 CUDA runtime。MMCV 的编译设置、CUDA 版本和具体硬件要求请以 [src/hrnet/README.md](src/hrnet/README.md) 中的环境说明为准。

## 数据集准备

完整合成数据集发布在 [hkustenterprize/RM26_engineer_exchange](https://huggingface.co/datasets/hkustenterprize/RM26_engineer_exchange)。数据集的主要 YOLO 目录结构为：

```text
dataset_root/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── dataset.yaml
```

训练集包含 40,860 张图片，验证集包含 4,540 张图片，共 45,400 张。下载后，将数据集路径填入 `src/yolo/configs/exchange_pose_dataset.yaml`，或在训练命令中通过 `--data` 显式指定自己的 `dataset.yaml`。

姿态数据集包含 `pillar` 和 `exchange` 两类，以及分布在两个目标上的 12 个关键点。检测模型只需要目标框，可以由姿态数据集转换得到：

```bash
cd public_archive
python src/yolo/pose2detect.py \
  --source /absolute/path/to/exchange_pose \
  --output /absolute/path/to/exchange_detect
```

如果使用公开数据集中的分片目录，应先将各分片合并为标准的 `images/{train,val}` 和 `labels/{train,val}` 结构，再执行训练。

## 合成数据生成

数据生成分为两个阶段。Blender 管线首先渲染带几何标注的目标素材，然后 `compose_dataset.py` 将目标素材与背景合成，并写出 YOLO 标签：

```text
Blender 场景
    -> src/blender/pipeline/render_dataset_new.py
    -> 目标图片、关键点和可见性标注
    -> src/blender/pipeline/compose_dataset.py
    -> YOLO Pose 数据集
```

渲染配置示例位于 [src/blender/configs/example.yaml](src/blender/configs/example.yaml)，具体参数说明见 [src/blender/configs/example.md](src/blender/configs/example.md)。渲染脚本需要通过命令行指定 Blender 场景、配置和输出目录；数据合成的基本形式为：

```bash
cd public_archive
blender -b /absolute/path/to/scene.blend \
  -P src/blender/pipeline/render_dataset_new.py -- \
  --config src/blender/configs/example.yaml \
  --output_dir /absolute/path/to/rendered_assets \
  --n_images 1000
```

随后将 Blender 输出素材合成为 YOLO 数据集：

```bash
python src/blender/pipeline/compose_dataset.py \
  --blender_dir /absolute/path/to/rendered_assets \
  --output_dir /absolute/path/to/exchange_pose
```

生成结果可以用以下脚本抽查关键点和目标框：

```bash
python src/blender/pipeline/visualize_keypoints.py \
  --dataset_dir /absolute/path/to/rendered_assets \
  --n 20 \
  --out_dir /absolute/path/to/visualized_samples
```

## YOLO 训练

YOLO 主线由一个训练入口支持姿态和检测两种任务。两个 YAML 配置是自包含的，分别保留完整训练参数、损失权重和增强范围：

| 配置 | 任务 | 模型输出 | 主要用途 |
| --- | --- | --- | --- |
| `src/yolo/configs/train_pose.yaml` | 姿态 | 目标框 + 12 个关键点 | bottom-up 关键点检测 |
| `src/yolo/configs/train_detect.yaml` | 检测 | 目标框 | top-down LiteHRNet 的前级检测 |

使用 `uv` 环境训练：

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

训练参数可以通过命令行覆盖。第一次建议使用少量 epoch 检查数据和增强：

```bash
uv run --project src/hrnet python src/yolo/train/train.py \
  --config src/yolo/configs/train_pose.yaml \
  --data /absolute/path/to/dataset.yaml \
  --epochs 1 --batch 8 --imgsz 256 --workers 0 \
  --name smoke_pose
```

训练结果默认写入 `src/yolo/runs/pose/` 或 `src/yolo/runs/detect/`。由于配置中启用了 `plots: true`，Ultralytics 会生成：

```text
train_batch0.jpg       # 训练批次中的目标框和关键点/目标框
labels.jpg             # 标签分布
results.png            # 训练曲线
val_batch*_labels.jpg  # 验证集标签
val_batch*_pred.jpg    # 验证集预测
weights/best.pt
```

## LiteHRNet 训练与推理

LiteHRNet 使用 top-down 流程：YOLO 检测目标框，随后对局部区域进行 LiteHRNet 关键点预测。公开训练配置为：

```text
src/hrnet/configs/td-hm_litehrnet18_exchange12_v11.0.py
src/hrnet/configs/td-hm_litehrnet30_exchange12_v11.0.py
```

MMPose 训练需要先将 YOLO Pose 标签转换为 COCO keypoint 标注：

```bash
cd public_archive
uv run --project src/hrnet python src/hrnet/data/convert_yolo_to_coco.py \
  /absolute/path/to/exchange_pose \
  --output /absolute/path/to/exchange_pose_annotations
```

设置数据路径后训练：

```bash
export HRNET_DATA_ROOT=/absolute/path/to/exchange_pose
export HRNET_ANN_ROOT=/absolute/path/to/exchange_pose_annotations

cd public_archive/src/hrnet
bash scripts/train_litehrnet.sh td-hm_litehrnet18_exchange12_v11.0.py
bash scripts/train_litehrnet.sh td-hm_litehrnet30_exchange12_v11.0.py
```

图片和视频推理共用 `src/hrnet/tools/infer.py`。具体参数和 OpenVINO 导出方式见 [src/hrnet/README.md](src/hrnet/README.md)。

## 发布物

### Dataset

[RM26_engineer_exchange](https://huggingface.co/datasets/hkustenterprize/RM26_engineer_exchange)

### Checkpoints

[RM26_engineer_exchange_model](https://huggingface.co/hkustenterprize/RM26_engineer_exchange_model)

| 模型 | 任务 | 输入尺寸 | 原生权重 | OpenVINO |
| --- | --- | --- | --- | --- |
| LiteHRNet-30-v9.1 | 12-keypoint | 256x256 | [best.pth](https://huggingface.co/hkustenterprize/RM26_engineer_exchange_model/resolve/main/litehr/litehrnet30_v9.1/best.pth?download=true) | [openvino_best](https://huggingface.co/hkustenterprize/RM26_engineer_exchange_model/tree/main/litehr/litehrnet30_v9.1/openvino_best) |
| YOLO26-s-v10.12 | Box Detection | 640x640 | [best.pt](https://huggingface.co/hkustenterprize/RM26_engineer_exchange_model/resolve/main/yolo/v10.12/best.pt?download=true) | [openvino_best](https://huggingface.co/hkustenterprize/RM26_engineer_exchange_model/tree/main/yolo/v10.12/openvino_best) |
| YOLO26-s-pose-v11.02 | Box Detection + 12-keypoint | 640x640 | [best.pt](https://huggingface.co/hkustenterprize/RM26_engineer_exchange_model/resolve/main/yolopose/v11.02/best.pt?download=true) | [openvino_best](https://huggingface.co/hkustenterprize/RM26_engineer_exchange_model/tree/main/yolopose/v11.02/openvino_best) |

## Release Log

### 2026-06-24

- 发布技术报告 PDF、项目说明和 Hugging Face 数据集入口。
- 发布 LiteHRNet、YOLO 检测和 YOLO Pose 模型权重入口。

### Current development

- 整理 Blender 合成数据管线。
- 整理 YOLO 和 LiteHRNet 的训练、推理与 OpenVINO 导出入口。
- 将历史实验代码移入 `src/hrnet/experimental/`。

## License

代码按 MIT License 发布，详见 [LICENSE](LICENSE)。数据集和模型权重的具体使用条款请以对应 Hugging Face 页面为准。

## Acknowledgement

项目使用了 MuJoCo、OMPL、hpp-fcl、Ultralytics、MMPose、SciPy 和 Albumentations 等开源工具。
