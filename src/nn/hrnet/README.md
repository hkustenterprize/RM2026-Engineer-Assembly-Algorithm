# LiteHRNet 12-Keypoint Pipeline

该目录包含装配目标 12 点关键点模型的数据转换、训练、推理与 OpenVINO 导出工具。训练和推理采用
MMPose 的 top-down heatmap 流程：YOLO 先给出目标框，LiteHRNet 再在目标区域内预测关键点。

公开主线只保留两组配置：

- `td-hm_litehrnet18_exchange12_v11.0.py`
- `td-hm_litehrnet30_exchange12_v11.0.py`

两者使用相同的 12 点定义和 `256x256` 网络输入，区别仅在 LiteHRNet 主干深度。模型输出关键点热力图，
并通过附加分支预测各关键点是否位于图像内。

## Directory Layout

```text
hrnet/
├── configs/                  # 两组公开训练配置
├── custom/
│   └── visibility_head.py    # 带可见性分支的 heatmap head
├── data/
│   ├── convert_yolo_to_coco.py
│   └── visualize_dataset.py
├── tools/
│   ├── train.py              # 基于 MMEngine Runner 的训练入口
│   ├── infer.py              # 图片与视频共用的推理入口
│   ├── export_openvino.py
│   └── check_environment.py
└── experimental/             # 非主线实验与历史实现
```

`experimental/` 中的文件用于保留实验过程，不属于公开主线，也不保证与当前精简接口兼容。
环境定义和锁文件位于上一级 `src/nn/`，与 YOLO 共用。

## Environment

训练、推理与导出使用 `src/nn` 下的共享 uv 环境。当前锁文件固定 Python 3.10、NumPy 2.2、PyTorch CUDA 13.0
以及与两组配置兼容的 OpenMMLab 依赖。首次安装运行：

```bash
bash src/nn/setup_env.sh
source src/nn/.venv/bin/activate
```

安装脚本先同步普通依赖，再编译 MMCV 的 CPU 扩展。LiteHRNet 和 Ultralytics YOLO 的 GPU 计算由
PyTorch wheel 自带的 CUDA 运行时执行；当前主线不调用 MMCV CUDA 算子，因此安装过程不要求本机提供
CUDA Toolkit。由于 PyPI 发布的 `xtcocotools` wheel 仍使用 NumPy 1.x ABI，脚本还会在当前 NumPy 2.2
环境中从源码重编译该扩展。安装结束后会加载两份配置并构建模型和数据增强组件，以检查环境完整性。

因此，首次创建环境或删除 `src/nn/.venv` 后应运行 `bash src/nn/setup_env.sh`，不能只运行一次普通的
`uv sync`。环境建立完成后，可以直接使用已注册的 `rm26-nn` 命令；锁文件未变化时无需重复编译上述扩展。

检查安装：

```bash
rm26-nn check
```

## Data Conversion

源数据采用 YOLO pose 目录结构：

```text
dataset_root/
├── images/{train,val}/
└── labels/{train,val}/
```

转换脚本将标签整理为 MMPose 使用的 COCO keypoint JSON，并保留关键点原始可见性和 in-frame 标记：

```bash
rm26-nn hrnet convert DATASET_ROOT \
  --output ANNOTATION_DIR
```

`hrnet visualize` 可用于抽查转换后的 COCO 标注或原始 YOLO pose 标签，具体参数见
`rm26-nn hrnet visualize --help`。

## Training

数据集和标注路径通过环境变量传入，因此配置文件不依赖开发机器上的目录：

```bash
export HRNET_DATA_ROOT=/absolute/path/to/dataset
export HRNET_ANN_ROOT=/absolute/path/to/annotations

rm26-nn hrnet train \
  src/nn/hrnet/configs/td-hm_litehrnet18_exchange12_v11.0.py
rm26-nn hrnet train \
  src/nn/hrnet/configs/td-hm_litehrnet30_exchange12_v11.0.py
```

训练输出默认写入 `runs/<config-name>/`。可以在配置名后使用 `--work-dir` 或 `--resume` 覆盖运行参数。
需要加载初始权重时，设置 `HRNET_LOAD_FROM`；不设置时从随机初始化开始训练。

训练前可检查随机增强后的样本：

```bash
rm26-nn hrnet preview \
  src/nn/hrnet/configs/td-hm_litehrnet18_exchange12_v11.0.py \
  --output-dir /tmp/hrnet-augmentation-preview
```

## Inference

图片和视频共用 [tools/infer.py](tools/infer.py) 及同一个 `hrnet infer` 子命令。

图片推理：

```bash
rm26-nn hrnet infer \
  --yolo-weights /absolute/path/to/yolo_detect.pt \
  --hrnet-config src/nn/hrnet/configs/td-hm_litehrnet18_exchange12_v11.0.py \
  --hrnet-checkpoint /absolute/path/to/litehrnet.pth \
  --output-dir /tmp/hrnet-inference \
  --images image_1.jpg image_2.jpg
```

视频推理：

```bash
rm26-nn hrnet infer \
  --yolo-weights /absolute/path/to/yolo_detect.pt \
  --hrnet-config src/nn/hrnet/configs/td-hm_litehrnet18_exchange12_v11.0.py \
  --hrnet-checkpoint /absolute/path/to/litehrnet.pth \
  --source input.mp4 \
  --output-video output.mp4
```

`YOLO_WEIGHTS` 可指向 PyTorch、ONNX 等单文件权重，也可指向包含 `.xml`、`.bin` 和
`metadata.yaml` 的 Ultralytics OpenVINO 模型目录。检测类别默认为 `1`，检测器运行于 CPU，LiteHRNet
运行于 `cuda:0`；对应参数可通过 `--bbox-class-id`、`--detector-device` 与 `--device` 覆盖。

## OpenVINO Export

导出工具先生成 ONNX，再转换为 OpenVINO IR：

```bash
rm26-nn hrnet export \
  --config src/nn/hrnet/configs/td-hm_litehrnet30_exchange12_v11.0.py \
  --checkpoint checkpoints/litehrnet30_best.pth \
  --output-dir exports/litehrnet30
```

导出模型只包含归一化图像块到关键点热力图的网络计算。YOLO 检测、top-down 裁剪、热力图解码和原图
坐标还原仍由宿主程序完成。
