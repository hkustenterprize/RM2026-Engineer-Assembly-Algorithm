# Neural Network Pipelines

该目录集中管理 YOLO 与 LiteHRNet 两条视觉模型主线，并提供二者共用的 uv 环境。

```text
nn/
├── __main__.py
├── pyproject.toml
├── uv.lock
├── setup_env.sh
├── yolo/
└── hrnet/
```

## 环境要求

当前依赖锁文件使用 Python 3.10，并固定为 PyTorch 2.11.0、TorchVision 0.26.0 和 PyTorch 官方的
CUDA 13.0（`cu130`）wheel。该组合已在 NVIDIA 驱动 580.159.03 和 RTX 5070 Laptop GPU 上完成环境
自检，YOLO 与 LiteHRNet 均可正常加载。PyTorch wheel 自带所需的 CUDA 运行时，构建当前 LiteHRNet
主线使用的 MMCV CPU 算子时不要求单独安装 CUDA Toolkit；宿主机仍需安装能够支持 CUDA 13.0 的
NVIDIA 驱动。

`pyproject.toml` 和 `uv.lock` 目前只保证上述组合可复现。使用其他 CUDA 版本时，需要自行选择匹配的
PyTorch、TorchVision 和官方 wheel 索引，更新 `pyproject.toml`，重新生成 `uv.lock`，并重新构建及
验证 MMCV。由于 PyTorch、MMCV、MMPose 与显卡驱动之间存在版本约束，不建议直接沿用当前锁文件后
单独替换某一个包。

环境由 [uv](https://docs.astral.sh/uv/getting-started/installation/) 管理，仓库不包含 `uv` 可执行文件。
请先按照 uv 官方文档完成安装，并确认以下命令可用：

```bash
uv --version
```

仓库统一将 `src` 作为 Python 导入根目录。首次创建环境时，从仓库根目录运行：

```bash
bash src/nn/setup_env.sh
source src/nn/.venv/bin/activate
```

`setup_env.sh` 会按照锁文件创建 Python 3.10 环境、构建 MMCV 和 `xtcocotools` 的本地扩展，并将当前
仓库以 editable 形式注册到共享环境。首次安装不应以普通 `uv sync` 替代该脚本。安装完成后可执行
环境自检：

```bash
rm26-nn check
```

激活环境后，训练、推理和数据工具统一使用以下形式：

```bash
rm26-nn <command> [arguments]
```

`rm26-nn` 会自动将仓库的 `src` 目录加入当前进程的模块搜索路径，并通过 `PYTHONPATH` 传递给训练子进程，
因此无需在每条命令前手动设置该环境变量。

使用 `rm26-nn --help` 查看全部子命令。例如，检查环境和启动 YOLO 姿态训练：

```bash
rm26-nn check

rm26-nn yolo train \
  --config src/nn/yolo/configs/train_pose.yaml
```

不激活环境时，也可以使用 `uv run --project src/nn rm26-nn <command>`。

模型、数据集和训练输出不属于 Python 包。请通过配置文件、环境变量或命令行传入路径。具体用法分别见 [yolo/README.md](yolo/README.md) 和 [hrnet/README.md](hrnet/README.md)。
