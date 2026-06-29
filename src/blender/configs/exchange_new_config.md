# Exchange 合成配置说明（`configs/exchange_new.yaml`）

本文与 `pipeline_new/ops.py`、`pipeline_new/render_dataset_new.py` 一一对应。坐标均为 **Blender 世界系**，Z 向上，单位 **米**。

与旧版 `exchange.yaml` 的核心区别：**相机固定**，物体通过 Empty 父级每帧做旋转与平移（见 `object_transform`）。

## 运行命令

```text
blender -b exchange_removed.blend \
  -P pipeline_new/render_dataset_new.py -- \
  --config configs/exchange_new.yaml \
  --n_images 1000 --output_dir ./data_v10.0/18 \
  --light_type off --strip_light_color off --seed 40
```

或 `pipeline_new/render_new.sh`。

### CLI 覆盖项

| 参数 | 覆盖 YAML 字段 |
|------|----------------|
| `--light_type` | `lighting.light_type` |
| `--strip_light_color` | `lighting.strip_lights.color` |
| `--strip_light_strength` | `lighting.strip_lights.strength`（并清除 `strength_range`） |
| `--strip_light_on_prob` | `lighting.strip_lights.on_prob` |
| `--samples` / `--width` / `--height` | `render.samples` / `render.width` / `render.height` |
| `--n_shards` / `--shard_id` | 分片渲染，输出 `annotations_shardN.json` |

`--light_type` 未指定时，使用 YAML 中 `lighting.light_type`（默认 `red`）。

---

## `render.*`

| 字段 | 含义 |
|------|------|
| `engine` | 渲染引擎，如 `CYCLES` |
| `device` | `GPU`（OptiX）或 `CPU` |
| `width`, `height` | 输出分辨率（全图 RGBA，**无** alpha 裁剪） |
| `samples` | Cycles 采样数 |
| `use_denoising` | 是否降噪 |
| `sensor_width_mm` | 传感器宽度，用于内参 `fx, fy` |

---

## `camera.*`（`FixedCameraOp`）

相机在启动时设定一次，之后每帧不变。

| 字段 | 含义 |
|------|------|
| `position` | 相机世界坐标 `[x, y, z]`（m） |
| `look_at` | 对准点，仅用于计算初始朝向 |
| `fov_h_deg` | 水平视场角（度）；换算为 `lens` 焦距 |

内参 `K`、外参 `R_w2c` / `t_w2c` 写入 `annotations.json` 的 `camera_*` 字段。

---

## `object_transform.*`（`ObjectTransformOp`）

所有 MESH 在首次运行时 parent 到 Empty `_obj_transform_pivot`，每帧采样变换。

### 旋转

绕 `pivot` 点，Euler 顺序 **XYZ**（代码中 `pitch→roll→yaw` 对应 `'XYZ'`）：

| 字段 | 轴 | 效果（口语） |
|------|-----|-------------|
| `yaw_range` | Z | 左右转头 |
| `pitch_range` | X | 上下俯仰 |
| `roll_range` | Y | 绕视线方向倾斜 |

### 平移

| 字段 | 含义 |
|------|------|
| `pivot` | 旋转中心 / 距离参考点 |
| `distance_range` | pivot 沿相机视线方向到相机的距离（m）；越小物体越大 |
| `offset_x_range` | 沿相机右方向偏移（m）；正 = 画面偏右 |
| `offset_y_range` | 沿相机上方向偏移（m）；正 = 画面偏上 |

物体新中心：

```
new_center = cam_pos + distance * cam_fwd + offset_x * cam_right + offset_y * cam_up
```

### 取景

| 字段 | 含义 |
|------|------|
| `max_retries` | 取景失败时重采样次数；用尽后仍用最后一次结果 |
| `framing_margin` | 所有关键点距画边至少像素数（`all_kps_in_frame`） |

取景使用 **全部 target 关键点**（含 `pillar_slide` 偏移后的 pillar 点）。

### `pillar_slide`（可选）

| 字段 | 含义 |
|------|------|
| `objects` | glob 模式，如 `pillar_*` |
| `target` | 关联 target 名，关键点同步 Y 偏移 |
| `skip_prob` | 保持原位概率 |
| `y_range` | Empty 局部空间 Y 轴滑动范围（m） |

---

## `lighting.*`

### 全局

| 字段 | 含义 |
|------|------|
| `light_type` | 柱灯颜色：`red` / `blue` / `off` / `random` |
| `light_type_choices` | `random` 时的候选 |
| `strips_strength` | `scene.light_objects` 灯条 Emission 强度 |
| `ambient_strength` | 世界背景亮度 |
| `ambient_color` | 世界背景 RGB |
| `sun_energy` | 太阳灯能量（0 = 关） |

### `strip_lights`（`LightingSetupOp`）

独立控制背部主灯 `back_main_light_1/2`：

| 字段 | 含义 |
|------|------|
| `color` / `color_choices` | 颜色或随机 |
| `strength` / `strength_range` | 固定或每帧随机强度 |
| `on_prob` | 亮起概率；未亮则 `off` |
| `objects` | 受控网格名列表 |

### `arc_lights`（`ArcLightsOp`）

背部 36 条弧灯，每帧可 blink：

| 字段 | 含义 |
|------|------|
| `objects` | 灯条网格名 |
| `strength` | 亮时 Emission 强度 |
| `off` | `true` 强制全灭 |
| `blink` | 每帧随机亮灭 |
| `on_prob` | blink 时整帧亮起概率 |
| `per_object` | 每根灯条独立随机 |
| `min_on_count` / `max_on_count` | `per_object=true` 时每帧亮灯数量范围 |
| `color` / `color_choices` | `white` / `red` / `blue` / `off` / `random` |
| `base_material` / `off_material` | 亮/灭材质 |
| `emission_node` | Emission 节点名 |

### `main_lights`（`MainLightsOp`）

主侧板 Diffuse 材质 + Area 灯。**仅在首帧设定**，之后不随 `light_type` 变化。

| 字段 | 含义 |
|------|------|
| `objects` | 侧板网格 |
| `area_lights` | 面光源对象名 |
| `energy` | Area 灯能量 |
| `diffuse_node` | Diffuse BSDF 节点名 |
| `base_material` | 复制用基础材质 |

### `aux_lights`（`AuxLightsOp`）

每帧在 pivot 周围球面随机放置点光源，帧末删除。

| 字段 | 含义 |
|------|------|
| `count_range` | 每帧灯数量 |
| `dist` | 灯到 pivot 距离（m） |
| `energy_range` | 能量范围（W） |

颜色分支在代码中固定（暖白 / 冷白 / 全白 / 随机），YAML 不可单独配置。

### `glare`（`GlareSetupOp`）

Compositor `FOG_GLOW` 光晕，插在 Render Layers 与 Composite 之间。

| 字段 | 含义 |
|------|------|
| `glare_type` | 默认 `FOG_GLOW` |
| `threshold` | 亮度阈值 |
| `size` / `size_range` | 固定或每帧随机扩散半径 |
| `mix` | 混合比例（-1 原图 … 1 纯辉光） |

---

## `targets` / `pass_index`

每个 target 独立 seg mask → `bbox_2d`，可选 `keypoints`。

| 字段 | 含义 |
|------|------|
| `name` | target 名，如 `pillar` / `exchange` |
| `pass_index` | Object Index Pass 整数，**须唯一** |
| `mesh_objects.include` / `exclude` | glob 匹配 MESH；`exclude` 从 include 结果中剔除 |
| `keypoints` | `{name: [x,y,z]}` 初始世界坐标，经 `object_transform` 变换 |
| `min_visible` | 该 target 至少 `vis==2` 的点数（可被顶层 `min_visible` 覆盖） |
| `bbox_from_targets` | 由列出的 target bbox 取并集（exchange 用 pillar+exchange） |

**pass_index 赋值顺序**：先 reversed 遍历 targets，后赋值的覆盖先赋值的；因此 `exchange` 用 `include: ["*"]` 时须写在 pillar 之后，再由 pillar 覆盖 `pillar_*`。

**primary target**：`targets` 中第一个含 `keypoints` 的项（通常 `pillar`），其关键点写入标注顶层 `keypoints_2d`。

---

## `scene`

与 `.blend` 绑定的对象/材质名，供 `LightingSetupOp` 使用。

| 字段 | 含义 |
|------|------|
| `sun_object` | 太阳光对象名 |
| `emission_node` | 灯条 Emission 节点名 |
| `light_objects` | 柱身 9 条灯网格 |
| `material_map` | `red` / `blue` / `off` → 材质名 |

---

## 遮挡预检（`render_dataset_new.py`）

| 配置 | 默认 | 含义 |
|------|------|------|
| `occlusion_offset_m` | 0.001 | ray 起点沿 **点→相机** 外移（m），防自交 |
| `min_visible.pillar` | target 内 `min_visible` 或 `2` | pillar 至少 `vis==2` 个数 |
| `min_visible.exchange` | 同上 | exchange 至少 `vis==2` 个数 |

**vis 编码**：

| 值 | 含义 |
|----|------|
| `0` | 出画（`project_keypoints`） |
| `1` | 在画内但被遮挡（`check_occlusion_raycasts`） |
| `2` | 可见 |

预检失败则 **不渲染**，重采样下一帧（`idx` 不变）。

---

## 输出标注

每帧写入 `annotations.json`：

| 字段 | 说明 |
|------|------|
| `targets.<name>.bbox_2d` | seg mask 外接框 `[x0,y0,x1,y1]`，全图像素 |
| `targets.<name>.keypoints_2d` | `[[u,v,vis], ...]` |
| `bbox_2d` / `bbox_2d_exchange` | pillar / exchange bbox 冗余字段 |
| `camera_K` / `camera_R` / `camera_t` | OpenCV 内外参 |
| `meta` | `yaw_deg`、`distance_m`、`light_type`、`strip_lights_*` 等 |

无径向畸变、无 alpha 裁剪（与旧 `pipeline` 不同）。

---

## 相关源文件

- `pipeline_new/ops.py` — Op 实现
- `pipeline_new/render_dataset_new.py` — 主循环、segmap、预检
- `pipeline_new/utils.py` — 投影、遮挡（复用 `scripts/pipeline/utils.py`）
- `configs/exchange_new.yaml` — 配置模板
