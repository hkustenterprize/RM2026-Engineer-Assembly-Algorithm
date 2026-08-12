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

仓库统一将 `src` 作为 Python 导入根目录。首次安装环境时，从仓库根目录运行：

```bash
bash src/nn/setup_env.sh
source src/nn/.venv/bin/activate
```

安装过程会将当前仓库以 editable 形式注册到共享环境，并提供 `rm26-nn` 命令。激活环境后，训练、推理和
数据工具统一使用以下形式：

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
