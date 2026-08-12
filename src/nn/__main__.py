from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Sequence
from pathlib import Path


_COMMANDS = {
    ("check",): ("nn.hrnet.tools.check_environment", "检查共享神经网络环境"),
    ("hrnet", "train"): ("nn.hrnet.tools.train", "训练 LiteHRNet"),
    ("hrnet", "infer"): ("nn.hrnet.tools.infer", "对图片或视频执行两阶段推理"),
    ("hrnet", "convert"): (
        "nn.hrnet.data.convert_yolo_to_coco",
        "将 YOLO Pose 标签转换为 COCO 关键点标注",
    ),
    ("hrnet", "preview"): (
        "nn.hrnet.tools.preview_augmentations",
        "预览 LiteHRNet 训练增强",
    ),
    ("hrnet", "visualize"): (
        "nn.hrnet.data.visualize_dataset",
        "可视化 YOLO 或 COCO 关键点标注",
    ),
    ("hrnet", "export"): (
        "nn.hrnet.tools.export_openvino",
        "导出 LiteHRNet OpenVINO 模型",
    ),
    ("yolo", "train"): ("nn.yolo.train.train", "训练 YOLO pose 或 detect 模型"),
    ("yolo", "pose2detect"): (
        "nn.yolo.pose2detect",
        "将 YOLO Pose 数据集转换为检测数据集",
    ),
    ("yolo", "export"): ("nn.yolo.export_openvino", "导出 YOLO OpenVINO 模型"),
}


def _configure_python_path() -> None:
    src_root = str(Path(__file__).resolve().parents[1])
    sys.path[:] = list(dict.fromkeys([src_root, *sys.path]))
    inherited = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    os.environ["PYTHONPATH"] = os.pathsep.join(
        dict.fromkeys([src_root, *(path for path in inherited if path)])
    )


def _print_help(group: str | None = None) -> None:
    program = "rm26-nn" if Path(sys.argv[0]).name == "rm26-nn" else "python -m nn"
    print(f"Usage: {program} <command> [arguments]\n")
    print("Commands:")
    for command, (_, description) in _COMMANDS.items():
        if group is not None and command[0] != group:
            continue
        print(f"  {' '.join(command):<20} {description}")
    print(f"\nRun '{program} <command> --help' for command-specific options.")


def _resolve_command(args: Sequence[str]) -> tuple[str, list[str]] | None:
    for command_length in (2, 1):
        key = tuple(args[:command_length])
        if key in _COMMANDS:
            return _COMMANDS[key][0], list(args[command_length:])
    return None


def main(argv: Sequence[str] | None = None) -> int:
    _configure_python_path()
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_help()
        return 0

    if args[0] in {"hrnet", "yolo"} and (
        len(args) == 1 or (len(args) == 2 and args[1] in {"-h", "--help"})
    ):
        _print_help(args[0])
        return 0

    resolved = _resolve_command(args)
    if resolved is None:
        print(f"Unknown command: {' '.join(args[:2])}", file=sys.stderr)
        _print_help()
        return 2

    module_name, command_args = resolved
    module = importlib.import_module(module_name)
    command_main = getattr(module, "main", None)
    if not callable(command_main):
        raise RuntimeError(f"Command module has no callable main(): {module_name}")

    original_argv = sys.argv
    try:
        sys.argv = [module_name, *command_args]
        result = command_main()
    finally:
        sys.argv = original_argv
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
