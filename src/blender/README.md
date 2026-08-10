# Blender 合成数据生成

本目录负责从 Blender 场景生成带几何标注的目标素材，并将素材合成为 YOLO Pose 数据集。当前公开场景文件为 `exchange.blend`。

## 目录结构

```text
blender/
├── configs/
│   ├── example.yaml       # 渲染配置示例
│   └── example.md         # 配置字段说明
├── pipeline/
│   ├── render_dataset.py      # Blender 渲染入口
│   ├── compose_dataset.py     # 素材合成与 YOLO 标签生成
│   ├── visualize_keypoints.py # 标注可视化
│   ├── context.py / ops.py / utils.py
│   └── setup_blender_env.sh
└── exchange.blend              # 公开 Blender 场景
```

## 数据生成流程

```text
Blender 场景 + configs/*.yaml
        |
        v
render_dataset.py
        |
        |  RGBA 图片、目标框、关键点、可见性、相机参数
        v
compose_dataset.py
        |
        v
YOLO Pose 数据集
```

渲染阶段固定相机，并对目标物体、灯光和场景参数进行采样。每一帧会执行关键点投影和可见性判断，再输出目标图片及其标注。合成阶段可以使用真实背景图片池，将目标素材与背景进行亮度匹配，并写出标准的 `images/{train,val}`、`labels/{train,val}` 和 `dataset.yaml`。

## 环境

需要安装 Blender 4.x，并保证 Blender 可以加载场景所需的 Python 模块。公开场景位于当前目录的 `exchange.blend`：

```bash
cd public_archive/src/blender
bash setup_blender_env.sh
```

## 渲染

渲染入口使用当前目录的公开 Blender 场景文件：

```bash
cd public_archive
blender -b src/blender/exchange.blend \
  -P src/blender/pipeline/render_dataset.py -- \
  --config src/blender/configs/example.yaml \
  --output_dir /absolute/path/to/rendered_assets \
  --n_images 1000 \
  --seed 40
```

可通过命令行覆盖渲染分辨率、采样数、灯光类型、灯条颜色以及分片参数。完整字段见 [configs/example.md](configs/example.md)。

渲染输出结构为：

```text
rendered_assets/
├── images/
├── annotations.json
├── config.yaml
└── meta.json
```

## 合成 YOLO 数据集

将一个或多个 Blender 输出目录合成为训练数据集：

```bash
cd public_archive
python src/blender/pipeline/compose_dataset.py \
  --blender_dir /absolute/path/to/rendered_assets \
  --output_dir /absolute/path/to/exchange_pose
```

如果有多个素材目录，可以使用 `--blender_dirs`。背景图片、验证集比例和随机种子等参数通过命令行设置，具体参数可以查看：

```bash
python src/blender/pipeline/compose_dataset.py --help
```

## 标注可视化

```bash
python src/blender/pipeline/visualize_keypoints.py \
  --dataset_dir /absolute/path/to/rendered_assets \
  --n 20 \
  --out_dir /absolute/path/to/visualized_samples
```

可视化工具读取渲染阶段的 `annotations.json`，用于检查目标框、关键点和可见性。合成后的 YOLO 数据集可使用 `src/hrnet/data/visualize_dataset.py` 进一步抽查。

## 下游接口

合成脚本输出的目录可以直接交给 `src/yolo/configs/exchange_pose_dataset.yaml` 所描述的 YOLO Pose 训练流程。检测数据集由 YOLO Pose 数据集转换得到：

```bash
python src/yolo/pose2detect.py \
  --source /absolute/path/to/exchange_pose \
  --output /absolute/path/to/exchange_detect
```
