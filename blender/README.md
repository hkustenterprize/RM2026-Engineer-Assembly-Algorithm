# Blender 合成数据生成

使用 **Blender 4.5 + Cycles** 生成装配柱 + 兑换站合成图与标注，输出供 `compose_dataset.py` 合成 YOLO 数据集。

当前管线：**`pipeline_new`**（固定相机 + 物体变换）。旧版移动相机管线在 `scripts/pipeline/`（已弃用）。

---

## 目录结构

```
blender/
├── exchange_removed.blend      # 当前主力场景
├── configs/
│   └── exchange_new.yaml       # pipeline_new 配置
├── pipeline_new/               # 当前渲染管线
│   ├── render_dataset_new.py   # 主驱动
│   ├── render_new.sh           # 启动模板
│   ├── ops.py / context.py / utils.py
│   └── visualize_keypoints.py
├── docs/
│   └── exchange_config.md      # 旧 exchange.yaml 配置说明
└── scripts/pipeline/           # 旧管线（移动相机）
```

---

## 环境

| 工具 | 用途 |
|------|------|
| Blender 4.5.4 LTS | Cycles 渲染 |
| opencv-python-headless | segmap / 后处理（Blender 内置 Python） |
| pyyaml | 配置解析 |

```bash
bash cv/nn/blender/setup_blender_env.sh
```

---

## 快速开始

```bash
cd cv/nn/blender/pipeline_new
bash render_new.sh
```

或手动：

```bash
blender -b cv/nn/blender/exchange_removed.blend \
  -P cv/nn/blender/pipeline_new/render_dataset_new.py -- \
  --config cv/nn/blender/configs/exchange_new.yaml \
  --output_dir ../data_v10.0/18 \
  --n_images 1000 --light_type off \
  --strip_light_color off --seed 40
```

可视化：

```bash
python cv/nn/blender/pipeline_new/visualize_keypoints.py \
  --dataset_dir ../data_v10.0/18 --n 20 --out_dir ./vis
```

---

## pipeline_new 流程

与旧管线（相机在球面上移动）不同：**相机固定**，每帧通过 Empty 父级对场景做旋转 / 距离 / 偏移采样。

```
exchange_new.yaml + CLI
        │
        ▼
render_dataset_new.py
  ├─ 固定相机 RenderCam + Cycles/GPU
  ├─ Object Index Pass → seg mask → 各 target bbox
  └─ build_ops() 每帧 pre-render
        │
        ▼
  FixedCameraOp → LightingSetupOp → ArcLightsOp → MainLightsOp
  → AuxLightsOp → GlareSetupOp → ObjectTransformOp
        │
        ▼
  预检（出画 / 遮挡 / min_visible）→ 渲染 RGBA → 写 annotations
```

| Op | 频率 | 功能 |
|----|------|------|
| `FixedCameraOp` | 首帧 | 相机位姿、FOV → K / R / t |
| `LightingSetupOp` | 每帧 | 柱灯材质、独立灯条亮灭 |
| `ArcLightsOp` | 每帧 | 背部弧灯 blink / 随机色 |
| `MainLightsOp` | 首帧 | 主侧板 Area 灯 |
| `AuxLightsOp` | 每帧 | 随机点光源（帧末删除） |
| `GlareSetupOp` | 每帧 size | Compositor 光晕 |
| `ObjectTransformOp` | 每帧 | yaw/pitch/roll/distance/offset + pillar 滑动 |

---

## 输出

```
output_dir/
├── images/{idx:05d}.png    # RGBA 全图
├── annotations.json        # 多 shard 时为 annotations_shardN.json
├── config.yaml
└── meta.json               # 仅 shard 0
```

每帧标注要点：

| 字段 | 说明 |
|------|------|
| `targets.pillar` / `targets.exchange` | 各 target 的 `bbox_2d`、`keypoints_2d` |
| `keypoints_2d`（顶层） | primary target（通常 pillar） |
| `bbox_2d` / `bbox_2d_exchange` | pillar / exchange 外接框 |
| `camera_K` / `camera_R` / `camera_t` | OpenCV 内外参 |
| `meta` | 物体变换角、距离、灯光状态等 |

**vis**：`0` 出画，`1` 遮挡，`2` 可见。

---

## 下游

```
Blender 输出 → compose_dataset.py → YOLO Pose（如 dataset_finalversion）
```

---

## 并行分片

```bash
--n_shards 4 --shard_id 0   # 输出 annotations_shard0.json，需自行合并
```

---

## 适配新场景

复制 `configs/exchange_new.yaml`，修改 `camera`、`object_transform`、`lighting`、`targets`（关键点 3D、pass_index、mesh_objects）。字段说明见 [`docs/exchange_new_config.md`](docs/exchange_new_config.md)。
