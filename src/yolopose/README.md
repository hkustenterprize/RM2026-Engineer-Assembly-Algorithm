# YOLO Pose / Detect

装配柱 + 兑换站视觉训练与推理，基于 **Ultralytics YOLO**。

| 任务 | 模型 | 用途 |
|------|------|------|
| Pose | YOLO11s-pose | 单阶段检测 + 关键点（12 点：pillar 5 + exchange 7） |
| Detect | YOLO26s | 仅 bbox，供两阶段 HRNet 或纯检测 |

---

## 数据流水线

```
Blender 渲染 (cv/nn/blender/pipeline_new/)
  → images/ + annotations.json
  → compose_dataset.py（贴 cv/nn/data/background 背景）
  → YOLO Pose 数据集 (images/ + labels/ + dataset.yaml)
  → [可选] merge_yolo_datasets / make_pillar_* / pose_to_detect_dataset
  → train.py 或 train_detect.py
  → export_openvino.py
```

---

## 目录结构

```
yolopose/
├── config/
│   ├── train_config.yaml         # Pose 训练
│   ├── train_detect_config.yaml  # Detect 训练默认配置
│   ├── pillar_pose.yaml          # compose 后同步的数据集 yaml
│   └── detect/
│       ├── v9.2.yaml
│       └── v9.2_no_oneof.yaml
├── data_process/
├── train/
├── inference/
├── scripts/
├── export_openvino.py
├── runs/                         # 训练输出 (gitignore)
└── eval_results/                 # 推理输出 (gitignore)
```

---

## Python 脚本

### `data_process/`

| 文件 | 用途 |
|------|------|
| `compose_dataset.py` | Blender RGBA + `annotations.json` → 贴背景 → YOLO Pose 格式（双类 12 kp） |
| `merge_raw_data.py` | 合并多个 **Blender 原始** 目录（`images/` + `annotations.json`）为单一 annotations |
| `merge_yolo_datasets.py` | 合并多个 **YOLO 格式** 数据集（train/val images+labels） |
| `make_pillar_only_dataset.py` | 12 kp 全量集 → **1 类 5 kp**（仅保留 pillar 行） |
| `make_pillar_kpt_dataset.py` | 12 kp 全量集 → **2 类 5 kp**（保留所有 bbox，非 pillar 关键点置零） |
| `pose_to_detect_dataset.py` | YOLO Pose 标签 → YOLO Detect 标签（去掉关键点，仅 bbox） |

### `train/`

| 文件 | 用途 |
|------|------|
| `train.py` | Pose 训练入口（YAML 配置、Albumentations、random crop/mask 增强） |
| `train_detect.py` | Detect 训练入口（bbox only，供 HRNet 两阶段） |
| `random_crop_aug.py` | Pose 自定义增强：随机裁剪 / mask，保持关键点一致 |
| `detect_aug.py` | Detect 自定义增强：`MaskDetectTrainer` 等 |
| `test_random_crop_aug.py` | `random_crop_aug.py` 单元测试 |

### `inference/`

| 文件 | 用途 |
|------|------|
| `infer_video.py` | 视频推理 CLI（`--model --source --output`） |
| `inference.py` | 检测模型批量图片推理（硬编码路径，调试用） |
| `inference_video.py` | Pose 模型批量图片推理（硬编码路径，调试用） |

### 根目录

| 文件 | 用途 |
|------|------|
| `export_openvino.py` | `.pt` → OpenVINO IR（`best.xml` + `best.bin`） |

---

## Shell 脚本 (`scripts/`)

| 文件 | 用途 |
|------|------|
| `compose.sh` | 调用 `compose_dataset.py` 的示例（改内部路径后直接用） |
| `merge.sh` | 调用 `merge_yolo_datasets.py` 的示例（**注意**：当前仍引用旧名 `merge_dataset.py`，应改为 `merge_yolo_datasets.py` 或 `merge_raw_data.py`） |
| `make_kpt.sh` | 调用 `make_pillar_kpt_dataset.py` 的示例 |
| `train_pose.sh` | `train.py --config config/train_config.yaml` |
| `train_detect.sh` | `train_detect.py --config config/detect/v9.2.yaml` |
| `infer_video.sh` | 调用 `infer_video.py` 的示例 |
| `export_openvino.sh` | 调用 `export_openvino.py` 的示例 |

---

## 快速开始

### 1. Blender 渲染

见 `cv/nn/blender/README.md`，入口 `pipeline_new/render_new.sh`。

### 2. 背景合成

```bash
python cv/nn/yolopose/data_process/compose_dataset.py \
  --blender_dir cv/nn/data_v10.0/18 \
  --bg_dir cv/nn/data/background \
  --output_dir cv/nn/data/dataset_xxx \
  --n_output 5000 --val_split 0.1 --seed 42
```

### 3. 合并 / 转换数据集

```bash
# 合并多个 YOLO 数据集
python cv/nn/yolopose/data_process/merge_yolo_datasets.py \
  --datasets ds_a ds_b --output ds_merged --symlink

# 12 kp → 5 kp（仅 pillar）
python cv/nn/yolopose/data_process/make_pillar_only_dataset.py SOURCE OUTPUT

# 12 kp → 5 kp（保留 exchange bbox）
python cv/nn/yolopose/data_process/make_pillar_kpt_dataset.py \
  --source SOURCE --output OUTPUT

# Pose → Detect
python cv/nn/yolopose/data_process/pose_to_detect_dataset.py \
  --source SOURCE --output OUTPUT
```

### 4. 训练

```bash
bash cv/nn/yolopose/scripts/train_pose.sh
# 或
python cv/nn/yolopose/train/train.py --config cv/nn/yolopose/config/train_config.yaml

bash cv/nn/yolopose/scripts/train_detect.sh
```

### 5. 推理 / 导出

```bash
python cv/nn/yolopose/inference/infer_video.py \
  --model runs/pose/.../best.pt --source video.mp4 --output eval_results/

python cv/nn/yolopose/export_openvino.py \
  --weights runs/pose/.../best.pt --imgsz 640 --half
```

---

## 关键点（pillar 5 点）

```
TL ──── TR          ● ring（柱顶圆心）
│       │
BL ──── BR
```

`flip_idx: [1, 0, 3, 2, 4]`。全量 12 点 schema 见 `dataset_finalversion/README.md`。

---

## 配置说明

| 文件 | 说明 |
|------|------|
| `config/train_config.yaml` | Pose：`data` 指向 `dataset_finalversion/dataset.yaml`，含 Albumentations / random_crop / random_mask |
| `config/train_detect_config.yaml` | Detect 默认配置 |
| `config/detect/v9.2.yaml` | Detect 实验配置（`train_detect.sh` 使用） |
| `config/pillar_pose.yaml` | `compose_dataset.py` 生成后自动同步 |

---

## compose_dataset.py 主要参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--blender_dir` | 必填 | Blender 输出（`images/` + `annotations.json`） |
| `--bg_dir` | None | 背景图目录；省略则纯色背景 |
| `--output_dir` | `./pillar_dataset` | 输出根目录 |
| `--min_pillars` / `--max_pillars` | 1 / 2 | 每帧柱子数 |
| `--bg_size` | `auto` | 自适应 4:3 背景尺寸 |
| `--val_split` | 0.2 | 验证集比例 |
| `--max_mask_overlap_ratio` | 0 | mask 重叠上限 |
| `--max_kp_occlusion_delta` | 0 | 允许新增遮挡 kp 数 |
| `--vis_output` | None | 调试可视化目录 |
| `--auto_crop_alpha` | 关 | 裁掉 Blender 图透明边 |

---

## 参考

- 数据集说明：`cv/nn/data/dataset_finalversion/README.md`
- Blender 管线：`cv/nn/blender/README.md`
- [Ultralytics YOLO Pose](https://docs.ultralytics.com/tasks/pose/)
