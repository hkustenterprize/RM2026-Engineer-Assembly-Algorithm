# HRNet / LiteHRNet Pose (MMPose)

装配柱 + 兑换站关键点训练与推理，基于 **MMPose**（Top-Down Heatmap / RTMPose SimCC / RTMO Bottom-Up）。

| 任务 | 模型 | 用途 |
|------|------|------|
| Top-Down 5 kp | LiteHRNet-30 + `PillarHeatmapHead` | 两阶段：YOLO Detect bbox → HRNet 柱顶 5 点 |
| Top-Down 12 kp | LiteHRNet-18/30 + `PillarHeatmapHeadWithVis` | 两阶段：YOLO Detect → HRNet 12 点（pillar 5 + exchange 7） |
| 单阶段 12 kp | RTMO-s | 无需 YOLO，整图 bottom-up 检测 + 关键点 |
| 对照 / 旧实验 | HRNet-W48、CSPNeXt-S、RTMPose SimCC | 历史配置，部分仍可用于对比 |

两阶段推理入口：`train/inference.py`（`PillarHrnetPipeline`）。  
单阶段推理：同一脚本 `--mode rtmo`（`RtmoPipeline`）。

**注意**：两阶段管线中 YOLO 权重应为 **Detect** 模型（`.pt`），不是 YOLO Pose。

---

## 数据流水线

```
YOLO Pose 数据集 (yolopose/compose_dataset.py 等产出)
  → prepare_data.py 或 prepare_data_new.py（YOLO labels → COCO keypoint JSON）
  → MMPose config 指向 images/ + annotations/
  → mmpose/tools/train.py + model_configs/*.py
  → train/inference.py（YOLO Detect .pt + HRNet .pth）
  → export_hrnet_openvino.py（可选，仅 heatmap 子图）
```

**annotation 命名约定**（`prepare_data*.py` 输出）：

| mode | 输出文件 | 用途 |
|------|----------|------|
| `pillar5` | `pillar_train.json`, `pillar_val.json` | 仅 pillar 行，5 关键点 |
| `exchange12` | `exchange12_train.json`, `exchange12_val.json` | exchange bbox + 合并 12 关键点 |
| `both` | 以上四类 | 同时导出 |

exchange12 模式要求 YOLO 标签文件中 **先 pillar 行、后 exchange 行** 成对出现（FIFO 配对）；未配对的 exchange 行与末尾多余的 pillar 行计入日志 `skipped_unpaired_rows`。

---

## 目录结构

```
hrnet/
├── model_configs/          # MMPose 训练配置（.py）
│   └── exchange12_meta.py  # RTMO 等引用的 12 点 metainfo 片段
├── data_process/
│   ├── prepare_data.py     # YOLO → COCO（基础版，默认 mode=pillar5）
│   ├── prepare_data_new.py # 同上 + in-frame 扩展字段 + 内置预览图（默认 exchange12）
│   └── viz_data.py         # COCO / YOLO 标注可视化（OpenCV 风格）
├── train/
│   ├── pillar_models.py    # 自定义 head / geo loss / optimizer（MMPose 注册）
│   ├── inference.py        # 两阶段 YOLO+HRNet 与 RTMO 推理 CLI + API
│   ├── inference_video.py  # 视频推理
│   └── test_geo_loss.py    # GeometricConsistencyLoss 单元测试
├── scripts/                # Shell 示例（硬编码路径，改后直接用）
├── export_hrnet_openvino.py
├── runs/                   # 训练输出 (gitignore 或本地产物)
└── vis/                    # 推理可视化输出（运行时创建）
```

训练实际调用 **外部** `mmpose/tools/train.py`，需设置：

```bash
export PYTHONPATH="/path/to/RM2026-Engineer-Host/cv/nn/hrnet/train:/path/to/mmpose:$PYTHONPATH"
```

`custom_imports` 中 `pillar_models` 依赖上述 `train/` 在 PYTHONPATH 中。

---

## Python 脚本

### `data_process/`

| 文件 | 用途 |
|------|------|
| `prepare_data.py` | YOLO Pose → COCO JSON；`--mode pillar5\|exchange12\|both`（默认 `pillar5`） |
| `prepare_data_new.py` | 同上，额外写入 `keypoints_raw_visibility` / `keypoints_in_frame`；默认 `--mode exchange12`；`--vis-dir` 可导出转换预览（PIL 风格） |
| `viz_data.py` | 可视化已有 COCO JSON 或 YOLO 数据集（`--yolo`）；支持 `--class-id` 过滤 pillar |

`prepare_data.py` 与 `prepare_data_new.py` 核心转换逻辑相同；后者面向带 visibility 分支的训练（`PillarHeatmapHeadWithVis`），并可选导出预览图。日常检查标注用 `viz_data.py` 即可。

### `train/`

| 文件 | 用途 |
|------|------|
| `pillar_models.py` | `GeometricConsistencyLoss`、`PillarHeatmapHead`、`PillarHeatmapHeadWithVis`、`PillarRTMCCHead`、`TopDownCSPNeXtPAFPN`、`MuonSGDOptimWrapperConstructor` 等 |
| `inference.py` | **主推理入口**：`PillarHrnetPipeline`（YOLO Detect bbox → HRNet）、`RtmoPipeline`；corner refine、heatmap grid 可视化 |
| `inference_video.py` | 视频版推理（逐帧调用 pipeline，写 mp4） |
| `test_geo_loss.py` | 几何损失 DLT / 手性 / 梯度测试（依赖 `dataset_v7.0_annotations/pillar_val.json`） |

### 根目录

| 文件 | 用途 |
|------|------|
| `export_hrnet_openvino.py` | MMPose checkpoint → ONNX → OpenVINO IR（**仅** backbone+neck+head 热力图；不含 YOLO crop、decode、坐标还原） |

---

## Shell 脚本 (`scripts/`)

| 文件 | 用途 |
|------|------|
| `prepare_data.sh` | 调用 `prepare_data.py` 示例（默认仅导出 **pillar5** 到 `dataset_v11.0_pillar_annotations/`；exchange12 命令已注释） |
| `train_litehrnet.sh` | 调用 `mmpose/tools/train.py` 的历史命令集合；使用前只保留一条未注释命令 |
| `inference.sh` | 两阶段 HRNet：eval_samples 图片 + heatmap / corner-debug 可视化（12 kp，YOLO class 1） |
| `inference_video.sh` | 视频两阶段推理示例 |
| `inference_rtmo.sh` | RTMO 单阶段 eval_samples 推理 |
| `vis_data.sh` | 调用 `mmpose/tools/misc/browse_dataset.py` 查看增强后训练样本 |

---

## 模型配置 (`model_configs/`)

| 文件 | 关键点 | 骨干 | 备注 |
|------|--------|------|------|
| `td-hm_litehrnet30_pillar5.py` | 5 | LiteHRNet-30 | pillar5 baseline（`PillarHeatmapHead`，**无** geo loss）；`ann_root` → `*_pillar_annotations/` |
| `td-hm_litehrnet30_pillar5_geo_vis.py` | 5 | LiteHRNet-30 | 继承 pillar5 + visibility 分支 + geo loss |
| `td-hm_litehrnet30_pillar_geo.py` | 5 | LiteHRNet-30 | 旧数据 `dataset_v7.0_annotations/` + geo loss |
| `td-hm_litehrnet30_pillar_geo_vis.py` | 5 | LiteHRNet-30 | v7.0 + vis（`pillar5_geo_vis` 的基类来源之一） |
| `td-hm_litehrnet30_exchange12_v9.1.py` | 12 | LiteHRNet-30 | 256×256 exchange12 |
| `td-hm_litehrnet30_exchange12_384.py` | 12 | LiteHRNet-30 | 384×384 + vis；继承 `pillar5_geo_vis` |
| `td-hm_litehrnet18_exchange12_v11.0.py` | 12 | LiteHRNet-18 | 当前主力 12 点 heatmap 配置之一 |
| `td-hm_cspnext-s_exchange12_v11.0.py` | 12 | CSPNeXt-S | exchange12 + MuSGD 优化器实验 |
| `td-hm_hrnet-w48_8xb32-210e_coco-256x192.py` | 5 | HRNet-W48 | 官方结构微调 pillar5（v7.0 数据） |
| `rtmo-s_exchange12_640.py` | 12 | RTMO | bottom-up，640×640 |
| `rtmpose_litehrnet30_pillar.py` | 5 | LiteHRNet-30 + SimCC | RTMPose 路线（v7.0） |
| `rtmpose_lite_pillar_geo.py` | 5 | LiteHRNet-30 + SimCC | + geo loss（v7.0） |
| `rtmpose_s_cspnet.py` | 5 | CSPNeXt + SimCC | 旧 RTMPose 实验（v7.0） |
| `exchange12_meta.py` | — | — | 12 点 `keypoint_info` / `skeleton_info` 片段（供 RTMO 引用） |

---

## 快速开始

### 1. 准备 COCO 标注

```bash
# pillar5（柱 bbox + 5 点）— 训练 td-hm_litehrnet30_pillar5*.py
python cv/nn/hrnet/data_process/prepare_data.py \
  /path/to/yolo_dataset \
  --output /path/to/dataset_v11.0_pillar_annotations \
  --mode pillar5

# exchange12（兑换站 bbox + 12 点）— 训练 litehrnet18/30 exchange 配置
python cv/nn/hrnet/data_process/prepare_data_new.py \
  /path/to/yolo_dataset \
  --output /path/to/dataset_v11.0_annotations \
  --mode exchange12 \
  --vis-dir vis_preview --vis-num 8
```

`prepare_data.sh` 默认只生成 pillar5；训练 `*_exchange12_*.py` 时需改用 `--mode exchange12`，且 config 中 `ann_root` 与输出目录一致。

### 2. 检查标注

```bash
# COCO JSON
python cv/nn/hrnet/data_process/viz_data.py \
  /path/to/yolo_dataset \
  --ann /path/to/annotations/exchange12_train.json \
  --img-dir images/train \
  --out-dir cv/nn/hrnet/vis_coco_check \
  --num-samples 20

# 原始 YOLO 标签
python cv/nn/hrnet/data_process/viz_data.py \
  /path/to/yolo_dataset/dataset.yaml \
  --yolo --split train --class-id 0 \
  --out-dir cv/nn/hrnet/vis_yolo_check --num-samples 20
```

### 3. 训练（MMPose）

```bash
export PYTHONPATH="cv/nn/hrnet/train:/path/to/mmpose:$PYTHONPATH"

python /path/to/mmpose/tools/train.py \
  cv/nn/hrnet/model_configs/td-hm_litehrnet18_exchange12_v11.0.py \
  --work-dir cv/nn/hrnet/runs/td-hm_litehrnet18_exchange12_v11.0
```

多卡示例见 `scripts/train_litehrnet.sh`（使用前请**只保留一条**未注释命令，并核对 `nproc_per_node` 与 `CUDA_VISIBLE_DEVICES` 数量一致）。

### 4. 两阶段推理

```bash
export PYTHONPATH="cv/nn/hrnet/train:/path/to/mmpose:$PYTHONPATH"

python cv/nn/hrnet/train/inference.py \
  --mode hrnet \
  --yolo-weights cv/nn/yolopose/runs/detect/.../weights/best.pt \
  --hrnet-config cv/nn/hrnet/model_configs/td-hm_litehrnet18_exchange12_v11.0.py \
  --hrnet-checkpoint cv/nn/hrnet/runs/.../best_coco_AP_epoch_xx.pth \
  --images cv/nn/data/eval/eval_samples/image0.png \
  --output-dir cv/nn/hrnet/vis/my_run \
  --bbox-class-id 1 \
  --crop-margin 0.05 \
  --vis-heatmap --vis-corner-debug
```

- **5 kp 模型**：默认 YOLO `class 0`（pillar bbox）
- **12 kp 模型**：默认 YOLO `class 1`（exchange bbox）；可用 `--bbox-class-id` 覆盖
- `corner_refine` 默认开启；关闭加 `--no-corner-refine`

### 5. RTMO 单阶段推理

```bash
python cv/nn/hrnet/train/inference.py \
  --mode rtmo \
  --rtmo-config cv/nn/hrnet/runs/rtmo-s_exchange12_640_v11.0/rtmo-s_exchange12_640.py \
  --rtmo-checkpoint cv/nn/hrnet/runs/rtmo-s_exchange12_640_v11.0/epoch_120.pth \
  --images cv/nn/data/eval/eval_samples/image0.png \
  --output-dir cv/nn/hrnet/vis/rtmo_run
```

### 6. 视频推理

```bash
python cv/nn/hrnet/train/inference_video.py \
  --yolo-weights ... --hrnet-config ... --hrnet-checkpoint ... \
  --source video.mp4 --output out.mp4 --frame-stride 2
```

### 7. 导出 OpenVINO（可选）

```bash
export PYTHONPATH="cv/nn/hrnet/train:/path/to/mmpose:$PYTHONPATH"

python cv/nn/hrnet/export_hrnet_openvino.py \
  --config cv/nn/hrnet/model_configs/td-hm_litehrnet18_exchange12_v11.0.py \
  --checkpoint cv/nn/hrnet/runs/.../best_coco_AP_epoch_xx.pth \
  --output-dir cv/nn/hrnet/runs/.../openvino_heatmap
```

### 8. 几何损失单元测试

```bash
export PYTHONPATH="cv/nn/hrnet/train:/path/to/mmpose:$PYTHONPATH"
python cv/nn/hrnet/train/test_geo_loss.py
```

---

## 关键点

### Pillar 5 点

```
TL ──── TR          ● ring（柱顶圆心）
│       │
BL ──── BR
```

`flip_idx: [1, 0, 3, 2, 4]`。全量 12 点 schema 见 `cv/nn/data/dataset_finalversion/README.md`。

### Exchange 12 点

前 5 个为 pillar 点，后 7 个为兑换站：`light_BR, light_TR, shell_R, shell_M, shell_L, light_TL, light_BL`。

---

## 配置与路径说明

| 配置类型 | 典型 `ann_root` | 需要的 JSON |
|----------|-----------------|-------------|
| pillar5 (`td-hm_litehrnet30_pillar5*.py`) | `dataset_v11.0_pillar_annotations/` | `pillar_{train,val}.json` |
| exchange12 (`*_exchange12_*.py`, `rtmo-*.py`) | `dataset_v11.0_annotations/` | `exchange12_{train,val}.json` |
| 旧 v7.0 实验 | `dataset_v7.0_annotations/` | `pillar_{train,val}.json` |

`prepare_data.py` / `prepare_data_new.py` 主要参数：

| 参数 | 默认（prepare_data / prepare_data_new） | 说明 |
|------|----------------------------------------|------|
| `source` | 必填 | YOLO 数据集根（`images/{train,val}` + `labels/{train,val}`） |
| `--output` | `annotations`（相对 source） | COCO JSON 输出目录 |
| `--mode` | `pillar5` / `exchange12` | 转换模式 |
| `--class-id` / `--exchange-class-id` | 0 / 1 | YOLO 类别 id |
| `--vis-dir` / `--vis-num` | 仅 `prepare_data_new.py` | 转换结果预览图 |

---

## `inference.py` 主要参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--mode` | `hrnet` | `hrnet` 两阶段 或 `rtmo` 单阶段 |
| `--yolo-conf` | 0.1 | YOLO bbox 置信度阈值 |
| `--crop-margin` | 0.5 | bbox 各边扩展比例（0.5 ≈ 宽高各扩一倍） |
| `--bbox-class-id` | 自动 | 5 kp→0，>5 kp→1 |
| `--no-corner-refine` | 关 | 关闭热力图峰值 + DLT 角点重排 |
| `--vis-heatmap` | 关 | 保存 per-channel heatmap 网格图 |
| `--vis-corner-debug` | 关 | 保存 refine 前后对比图 |

---

## 已知问题与注意事项

| 位置 | 问题 |
|------|------|
| `scripts/train_litehrnet.sh` | 使用前请只保留**一条**未注释训练命令；`nproc_per_node` 须与 `CUDA_VISIBLE_DEVICES` 数量一致 |
| `prepare_data*.py` exchange12 | 标签行顺序必须为 pillar→exchange；末尾孤立 pillar 也计入 `skipped_unpaired_rows` |
| `export_hrnet_openvino.py` | 仅导出 heatmap 子图；LiteHRNet 需 ONNX patch（脚本内 `_patch_litehrnet_for_onnx`） |
| `MMPOSE_ROOT` | 未安装 mmpose 时，`pillar_models.py` 会读环境变量 `MMPOSE_ROOT` 或常见路径 |

---

## 参考

- 上游 YOLO 数据与 Detect 模型：`cv/nn/yolopose/README.md`
- 数据集说明：`cv/nn/data/dataset_finalversion/README.md`
- 评估样例图：`cv/nn/data/eval/eval_samples/`
- [MMPose Top-Down](https://mmpose.readthedocs.io/en/latest/model_zoo_papers/algorithms.html#top-down-heatmap-based)
