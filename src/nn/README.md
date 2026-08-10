# Neural Network Pipelines

该目录集中管理 YOLO 与 LiteHRNet 两条视觉模型主线，并提供二者共用的 uv 环境。

```text
nn/
├── pyproject.toml
├── uv.lock
├── setup_env.sh
├── yolo/
└── hrnet/
```

仓库统一将 `src` 作为 Python 导入根目录。首次安装环境时，从仓库根目录运行：

```bash
bash src/nn/setup_env.sh
```

直接运行 Python 模块时使用以下形式：

```bash
PYTHONPATH=./src uv run --project src/nn python -m nn.<module>
```

例如，检查环境和启动 YOLO 姿态训练：

```bash
PYTHONPATH=./src uv run --project src/nn \
  python -m nn.hrnet.tools.check_environment

PYTHONPATH=./src uv run --project src/nn \
  python -m nn.yolo.train.train \
  --config src/nn/yolo/configs/train_pose.yaml
```

模型、数据集和训练输出不属于 Python 包。请通过配置文件、环境变量或命令行传入路径。具体用法分别见 [yolo/README.md](yolo/README.md) 和 [hrnet/README.md](hrnet/README.md)。
