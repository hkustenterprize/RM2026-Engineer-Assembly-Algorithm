#!/usr/bin/env python3
"""Unified Ultralytics training entry point for pose and detection models."""

import argparse
import os
import sys
from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parent
YOLO_ROOT = TRAIN_DIR.parent
SRC_ROOT = YOLO_ROOT.parent
DEFAULT_CONFIG = YOLO_ROOT / "configs" / "train_pose.yaml"

for path in (TRAIN_DIR, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

CLI_OVERRIDES = {
    "task": ("task",),
    "model": ("model",),
    "data": ("data",),
    "resume": ("resume",),
    "epochs": ("train", "epochs"),
    "batch": ("train", "batch"),
    "imgsz": ("train", "imgsz"),
    "device": ("train", "device"),
    "workers": ("train", "workers"),
    "project": ("train", "project"),
    "name": ("train", "name"),
    "optimizer": ("train", "optimizer"),
    "export_openvino": ("export_openvino",),
    "random_crop": ("random_crop", "enabled"),
    "random_occlusion": ("random_occlusion", "enabled"),
    "albumentations": ("albumentations", "enabled"),
}


def _merge_config(base, override):
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path, stack=()):
    import yaml

    path = path.resolve()
    if path in stack:
        raise ValueError(f"Circular config inheritance: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing training config: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Training config must be a YAML mapping: {path}")

    base_name = config.pop("_base_", None)
    if not base_name:
        return config
    base_path = Path(base_name).expanduser()
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    return _merge_config(_read_yaml(base_path, stack + (path,)), config)


def load_config(path):
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    config = _read_yaml(config_path)
    config["_config_path"] = str(config_path)

    for key in ("data", "resume"):
        value = config.get(key)
        if value and not Path(value).expanduser().is_absolute():
            config[key] = str((config_path.parent / value).resolve())

    project = (config.get("train") or {}).get("project")
    if project and not Path(project).expanduser().is_absolute():
        config["train"]["project"] = str((config_path.parent / project).resolve())
    return config


def _set_nested(config, path, value):
    current = config
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = value


def apply_cli_overrides(config, args):
    for argument, path in CLI_OVERRIDES.items():
        value = getattr(args, argument, None)
        if value is not None:
            _set_nested(config, path, value)
    return config


def _add_boolean_override(parser, name, help_text):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        f"--{name.replace('_', '-')}",
        dest=name,
        action="store_true",
        default=None,
        help=f"Enable {help_text}",
    )
    group.add_argument(
        f"--no-{name.replace('_', '-')}",
        dest=name,
        action="store_false",
        help=f"Disable {help_text}",
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--task", choices=("pose", "detect"))
    parser.add_argument("--model")
    parser.add_argument("--data")
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--device")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--project")
    parser.add_argument("--name")
    parser.add_argument("--optimizer")
    _add_boolean_override(parser, "export_openvino", "OpenVINO export")
    _add_boolean_override(parser, "random_crop", "random crop augmentation")
    _add_boolean_override(
        parser, "random_occlusion", "random occlusion augmentation"
    )
    _add_boolean_override(parser, "albumentations", "Albumentations")
    return parser.parse_args()


def _section(config, name):
    value = config.get(name) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section {name!r} must be a mapping")
    return value


def _model_path(config):
    if config.get("resume"):
        return config["resume"]
    default = "yolo26n-pose.pt" if config["task"] == "pose" else "yolo26n.pt"
    model = str(config.get("model", default))
    return model if model.endswith((".pt", ".yaml", ".yml")) else f"{model}.pt"


def _training_arguments(config):
    from augmentations import build_image_augmentations

    arguments = {
        "data": config.get("data"),
        **_section(config, "train"),
        **_section(config, "yolo_aug"),
        **_section(config, "loss"),
    }
    image_augmentations = build_image_augmentations(
        _section(config, "albumentations")
    )
    if image_augmentations is not None:
        arguments["augmentations"] = image_augmentations
    if config.get("resume"):
        arguments["resume"] = True
    return {key: value for key, value in arguments.items() if value is not None}


def _pose_trainer(config):
    augmentation_config = {
        "random_crop": _section(config, "random_crop"),
        "random_occlusion": _section(config, "random_occlusion"),
    }
    if not any(section.get("enabled") for section in augmentation_config.values()):
        return None

    from trainers import create_pose_trainer

    return create_pose_trainer(augmentation_config)


def _detect_trainer(config):
    augmentation_config = {
        name: _section(config, name)
        for name in (
            "random_crop",
            "box_crop",
            "crop_choice",
            "random_occlusion",
        )
    }
    if not any(
        augmentation_config[name].get("enabled")
        for name in ("random_crop", "box_crop", "random_occlusion")
    ):
        return None

    from trainers import create_detection_trainer

    return create_detection_trainer(augmentation_config)


def _configure_runtime():
    import cv2

    cv2.setNumThreads(0)
    try:
        cv2.ocl.setUseOpenCL(False)
    except AttributeError:
        pass
    current = os.environ.get("PYTHONPATH", "")
    paths = current.split(os.pathsep) if current else []
    for path in (str(TRAIN_DIR), str(SRC_ROOT)):
        if path not in paths:
            paths.insert(0, path)
    os.environ["PYTHONPATH"] = os.pathsep.join(paths)


def train(config):
    from ultralytics import YOLO

    task = config.get("task", "pose")
    if task not in {"pose", "detect"}:
        raise ValueError("Config field 'task' must be 'pose' or 'detect'")
    config["task"] = task

    model_path = _model_path(config)
    arguments = _training_arguments(config)
    trainer_class = _pose_trainer(config) if task == "pose" else _detect_trainer(config)

    print(f"[train] task={task}, model={model_path}")
    print(
        f"[train] epochs={arguments.get('epochs')}, batch={arguments.get('batch')}, "
        f"imgsz={arguments.get('imgsz')}"
    )
    if trainer_class is None:
        result = YOLO(model_path).train(**arguments)
    else:
        arguments["model"] = model_path
        trainer = trainer_class(overrides=arguments)
        trainer.train()
        result = trainer

    project = Path(arguments.get("project", "runs"))
    name = arguments.get("name", task)
    weights_dir = project / name / "weights"
    best_weights = weights_dir / "best.pt"
    if not best_weights.exists():
        best_weights = weights_dir / "last.pt"

    if best_weights.exists():
        print(f"[train] best checkpoint: {best_weights}")
    else:
        print(f"[train] warning: no checkpoint found in {weights_dir}")

    if config.get("export_openvino") and best_weights.exists():
        YOLO(str(best_weights)).export(
            format="openvino",
            imgsz=arguments.get("imgsz", 640),
            half=True,
            dynamic=False,
            nms=True,
        )
    return result


def main():
    _configure_runtime()
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    print(f"[train] config={config['_config_path']}")
    train(config)


if __name__ == "__main__":
    main()
