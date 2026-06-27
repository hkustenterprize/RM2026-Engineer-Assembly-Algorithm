#!/usr/bin/env python3
"""YOLO detect training entrypoint for top-down HRNet pipelines.

This script trains only bounding boxes. Keypoints are intentionally ignored.
Use pose_to_detect_dataset.py to convert YOLOPose labels to detect labels first.
"""

import argparse
import os
from pathlib import Path

from train import (
    _build_albumentations_transforms,
    _disable_opencv_threads,
    _load_config,
    _none_if_empty,
    _section,
    _set_nested,
)


DEFAULT_CONFIG = (
    Path(__file__).resolve().parent.parent / "config" / "train_detect_config.yaml"
)
_disable_opencv_threads()


def _add_yolopose_dir_to_pythonpath():
    """Ensure this directory is importable by DDP worker processes."""
    yolopose_dir = str(Path(__file__).parent.resolve())
    existing_pp = os.environ.get("PYTHONPATH", "")
    if yolopose_dir not in existing_pp.split(os.pathsep):
        os.environ["PYTHONPATH"] = (
            yolopose_dir + os.pathsep + existing_pp if existing_pp else yolopose_dir
        )


def _apply_cli_overrides(config, args):
    overrides = {
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
        "lr0": ("train", "lr0"),
        "patience": ("train", "patience"),
        "optimizer": ("train", "optimizer"),
        "export_openvino": ("export_openvino",),
        "random_crop_enabled": ("random_crop", "enabled"),
        "crop_scale": ("random_crop", "scale"),
        "crop_p": ("random_crop", "p"),
        "crop_min_bbox_area_ratio": ("random_crop", "min_bbox_area_ratio"),
        "crop_oneof_enabled": ("crop_oneof", "enabled"),
        "crop_oneof_p": ("crop_oneof", "p"),
        "box_crop_enabled": ("box_crop", "enabled"),
        "box_crop_p": ("box_crop", "p"),
        "random_mask_enabled": ("random_mask", "enabled"),
        "mask_p": ("random_mask", "p"),
        "mask_max_bbox_overlap": ("random_mask", "max_bbox_overlap"),
        "mask_max_attempts": ("random_mask", "max_attempts"),
        "albumentations_enabled": ("albumentations", "enabled"),
    }
    for attr, path in overrides.items():
        value = getattr(args, attr, None)
        if value is not None:
            _set_nested(config, path, _none_if_empty(value))
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO detect training for HRNet top-down")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="detect training YAML, default train_detect_config.yaml",
    )
    parser.add_argument("--model", type=str, default=None, help="override YAML model")
    parser.add_argument("--data", type=str, default=None, help="override YAML data")
    parser.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    parser.add_argument("--batch", type=int, default=None, help="override train.batch")
    parser.add_argument("--imgsz", type=int, default=None, help="override train.imgsz")
    parser.add_argument("--device", type=str, default=None, help="override train.device")
    parser.add_argument("--workers", type=int, default=None, help="override train.workers")
    parser.add_argument("--project", type=str, default=None, help="override train.project")
    parser.add_argument("--name", type=str, default=None, help="override train.name")
    parser.add_argument("--lr0", "--lr", dest="lr0", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--export_openvino",
        dest="export_openvino",
        action="store_true",
        default=None,
        help="override export_openvino=true",
    )
    parser.add_argument(
        "--no_export_openvino",
        dest="export_openvino",
        action="store_false",
        help="override export_openvino=false",
    )
    parser.add_argument(
        "--random_mask",
        dest="random_mask_enabled",
        action="store_true",
        default=None,
        help="override random_mask.enabled=true",
    )
    parser.add_argument(
        "--no_random_mask",
        dest="random_mask_enabled",
        action="store_false",
        help="override random_mask.enabled=false",
    )
    parser.add_argument("--mask_p", type=float, default=None, help="override random_mask.p")
    parser.add_argument(
        "--mask_max_bbox_overlap",
        type=float,
        default=None,
        help="override random_mask.max_bbox_overlap",
    )
    parser.add_argument(
        "--mask_max_attempts",
        type=int,
        default=None,
        help="override random_mask.max_attempts",
    )
    parser.add_argument(
        "--random_crop",
        dest="random_crop_enabled",
        action="store_true",
        default=None,
        help="override random_crop.enabled=true",
    )
    parser.add_argument(
        "--no_random_crop",
        dest="random_crop_enabled",
        action="store_false",
        help="override random_crop.enabled=false",
    )
    parser.add_argument(
        "--crop_scale",
        type=float,
        nargs=2,
        default=None,
        help="override random_crop.scale, e.g. --crop_scale 0.65 1.0",
    )
    parser.add_argument("--crop_p", type=float, default=None, help="override random_crop.p")
    parser.add_argument(
        "--crop_min_bbox_area_ratio",
        type=float,
        default=None,
        help="override random_crop.min_bbox_area_ratio",
    )
    parser.add_argument(
        "--crop_oneof",
        dest="crop_oneof_enabled",
        action="store_true",
        default=None,
        help="override crop_oneof.enabled=true",
    )
    parser.add_argument(
        "--no_crop_oneof",
        dest="crop_oneof_enabled",
        action="store_false",
        help="override crop_oneof.enabled=false",
    )
    parser.add_argument("--crop_oneof_p", type=float, default=None)
    parser.add_argument(
        "--box_crop",
        dest="box_crop_enabled",
        action="store_true",
        default=None,
        help="override box_crop.enabled=true",
    )
    parser.add_argument(
        "--no_box_crop",
        dest="box_crop_enabled",
        action="store_false",
        help="override box_crop.enabled=false",
    )
    parser.add_argument("--box_crop_p", type=float, default=None)
    parser.add_argument(
        "--albumentations",
        dest="albumentations_enabled",
        action="store_true",
        default=None,
        help="override albumentations.enabled=true",
    )
    parser.add_argument(
        "--no_albumentations",
        dest="albumentations_enabled",
        action="store_false",
        help="override albumentations.enabled=false",
    )
    return parser.parse_args()


def _build_train_kwargs(config):
    train_cfg = _section(config, "train")
    yolo_aug_cfg = _section(config, "yolo_aug")
    detect_loss_cfg = _section(config, "detect_loss")

    train_kwargs = {
        "data": config.get("data"),
        "epochs": train_cfg.get("epochs"),
        "batch": train_cfg.get("batch"),
        "imgsz": train_cfg.get("imgsz"),
        "device": train_cfg.get("device") or None,
        "workers": train_cfg.get("workers"),
        "project": train_cfg.get("project"),
        "name": train_cfg.get("name"),
        "lr0": train_cfg.get("lr0"),
        "patience": train_cfg.get("patience"),
        "save_period": train_cfg.get("save_period"),
        "plots": train_cfg.get("plots"),
        "verbose": train_cfg.get("verbose"),
        "optimizer": train_cfg.get("optimizer"),
    }
    train_kwargs.update(yolo_aug_cfg)
    for key in ("box", "cls", "dfl"):
        if key in detect_loss_cfg:
            train_kwargs[key] = detect_loss_cfg[key]
    return {k: v for k, v in train_kwargs.items() if v is not None}


def _model_path(config):
    resume = config.get("resume")
    if resume:
        return resume

    model = str(config.get("model", "yolo26n.pt"))
    if model.endswith((".pt", ".yaml", ".yml")):
        return model
    return f"{model}.pt"


def train_detect(config):
    from ultralytics import YOLO

    model_path = _model_path(config)
    if config.get("resume"):
        print(f"[train_detect] Resume from checkpoint: {model_path}")
    else:
        print(f"[train_detect] Load detect model: {model_path}")

    train_kwargs = _build_train_kwargs(config)

    custom_augmentations = _build_albumentations_transforms(
        _section(config, "albumentations")
    )
    if custom_augmentations is not None:
        train_kwargs["augmentations"] = custom_augmentations
        print("[train_detect] Custom Albumentations enabled")

    if config.get("resume"):
        train_kwargs["resume"] = True

    random_crop = _section(config, "random_crop")
    crop_oneof = _section(config, "crop_oneof")
    box_crop = _section(config, "box_crop")
    random_mask = _section(config, "random_mask")
    crop_enabled = bool(random_crop.get("enabled"))
    crop_oneof_enabled = bool(crop_oneof.get("enabled"))
    box_crop_enabled = bool(box_crop.get("enabled"))
    mask_enabled = bool(random_mask.get("enabled"))

    print(
        f"[train_detect] Start training: {train_kwargs.get('epochs')} epochs, "
        f"batch={train_kwargs.get('batch')}, imgsz={train_kwargs.get('imgsz')}"
    )

    if crop_enabled or box_crop_enabled or mask_enabled:
        _add_yolopose_dir_to_pythonpath()
        from detect_aug import make_detect_aug_trainer

        crop_scale = tuple(random_crop.get("scale", [0.65, 1.0]))
        crop_p = random_crop.get("p", 0.3)
        crop_min_bbox_area_ratio = random_crop.get("min_bbox_area_ratio", 0.05)
        crop_oneof_p = crop_oneof.get("p", 0.4)
        crop_oneof_weights = crop_oneof.get("weights", {}) or {}
        crop_oneof_random_weight = crop_oneof_weights.get("random_crop", 1.0)
        crop_oneof_box_weight = crop_oneof_weights.get("box_crop", 1.0)
        if crop_enabled:
            print(
                "[train_detect] RandomCrop enabled: "
                f"scale={crop_scale}, p={crop_p}, "
                f"min_bbox_area_ratio={crop_min_bbox_area_ratio}"
            )
        if crop_oneof_enabled:
            print(
                "[train_detect] Crop OneOf enabled: "
                f"p={crop_oneof_p}, random_crop_weight={crop_oneof_random_weight}, "
                f"box_crop_weight={crop_oneof_box_weight}"
            )

        box_crop_p = box_crop.get("p", 0.2)
        box_crop_target_classes = tuple(box_crop.get("target_classes", []))
        box_crop_scale = tuple(box_crop.get("crop_scale", [1.3, 3.0]))
        box_crop_center_jitter = box_crop.get("center_jitter", 0.15)
        box_crop_min_bbox_area_ratio = box_crop.get("min_bbox_area_ratio", 0.05)
        box_crop_min_crop_size_ratio = box_crop.get("min_crop_size_ratio", 0.25)
        if box_crop_enabled:
            print(
                "[train_detect] BoxCrop enabled: "
                f"p={box_crop_p}, target_classes={box_crop_target_classes}, "
                f"crop_scale={box_crop_scale}, center_jitter={box_crop_center_jitter}, "
                f"min_bbox_area_ratio={box_crop_min_bbox_area_ratio}, "
                f"min_crop_size_ratio={box_crop_min_crop_size_ratio}"
            )

        mask_holes = tuple(random_mask.get("holes", [1, 2]))
        mask_height = tuple(random_mask.get("height", [0.04, 0.15]))
        mask_width = tuple(random_mask.get("width", [0.04, 0.15]))
        mask_fill = random_mask.get("fill", 0)
        mask_random_fill_p = random_mask.get("random_fill_p", 0.0)
        mask_max_bbox_overlap = random_mask.get("max_bbox_overlap", 1.0)
        mask_max_attempts = random_mask.get("max_attempts", 20)
        mask_p = random_mask.get("p", 0.1)
        if mask_enabled:
            print(
                "[train_detect] RandomMask enabled: "
                f"p={mask_p}, holes={mask_holes}, height={mask_height}, "
                f"width={mask_width}, fill={mask_fill}, "
                f"random_fill_p={mask_random_fill_p}, "
                f"max_bbox_overlap={mask_max_bbox_overlap}, "
                f"max_attempts={mask_max_attempts}"
            )

        train_kwargs["model"] = model_path
        MaskDetectTrainer = make_detect_aug_trainer(
            crop_enabled=crop_enabled,
            crop_scale=crop_scale,
            crop_p=crop_p,
            crop_min_bbox_area_ratio=crop_min_bbox_area_ratio,
            crop_oneof_enabled=crop_oneof_enabled,
            crop_oneof_p=crop_oneof_p,
            crop_oneof_random_weight=crop_oneof_random_weight,
            crop_oneof_box_weight=crop_oneof_box_weight,
            box_crop_enabled=box_crop_enabled,
            box_crop_p=box_crop_p,
            box_crop_target_classes=box_crop_target_classes,
            box_crop_scale=box_crop_scale,
            box_crop_center_jitter=box_crop_center_jitter,
            box_crop_min_bbox_area_ratio=box_crop_min_bbox_area_ratio,
            box_crop_min_crop_size_ratio=box_crop_min_crop_size_ratio,
            mask_enabled=mask_enabled,
            mask_p=mask_p,
            mask_holes=mask_holes,
            mask_height=mask_height,
            mask_width=mask_width,
            mask_fill=mask_fill,
            mask_random_fill_p=mask_random_fill_p,
            mask_max_bbox_overlap=mask_max_bbox_overlap,
            mask_max_attempts=mask_max_attempts,
        )
        trainer = MaskDetectTrainer(overrides=train_kwargs)
        trainer.train()
        results = trainer
    else:
        model = YOLO(model_path)
        results = model.train(**train_kwargs)

    project = train_kwargs.get("project", "runs/detect")
    name = train_kwargs.get("name", "train")
    data = train_kwargs.get("data")
    imgsz = train_kwargs.get("imgsz", 640)
    best_weights = Path(project) / name / "weights" / "best.pt"
    if best_weights.exists():
        print(f"\n[train_detect] Validate best weights: {best_weights}")
        model_best = YOLO(str(best_weights))
        metrics = model_best.val(data=data, imgsz=imgsz)
        print(f"  Box mAP50:    {metrics.box.map50:.4f}")
        print(f"  Box mAP50-95: {metrics.box.map:.4f}")
    else:
        print("[train_detect] Warning: best.pt not found")
        best_weights = Path(project) / name / "weights" / "last.pt"

    if config.get("export_openvino") and best_weights.exists():
        print(f"\n[train_detect] Export OpenVINO FP16: {best_weights}")
        model_export = YOLO(str(best_weights))
        model_export.export(
            format="openvino",
            imgsz=imgsz,
            half=True,
            dynamic=False,
            nms=True,
        )

    print("\n[train_detect] Done")
    return results


if __name__ == "__main__":
    args = parse_args()
    cfg = _apply_cli_overrides(_load_config(args.config), args)
    print(f"[train_detect] Config: {cfg.get('_config_path')}")
    train_detect(cfg)
