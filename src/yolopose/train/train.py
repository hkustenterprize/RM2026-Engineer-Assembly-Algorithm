#!/usr/bin/env python3
"""
train.py — YOLO11-pose 装配柱关键点检测训练脚本

前置条件:
  1. 已运行 compose_dataset.py 生成 YOLO 格式数据集
  2. pillar_pose.yaml 已更新为正确路径
  3. pip install ultralytics albumentations pyyaml

用法:
  # 使用统一 YAML 配置
  python train.py --config train_config.yaml

  # 临时覆盖 YAML 中的少量字段
  python train.py --config train_config.yaml --epochs 10 --batch 16 --device cpu

"""

import argparse
import math
import os
from pathlib import Path

# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"


def _add_yolopose_dir_to_pythonpath():
    """Ensure this file's directory is importable by DDP worker processes."""
    yolopose_dir = str(Path(__file__).parent.resolve())
    existing_pp = os.environ.get("PYTHONPATH", "")
    if yolopose_dir not in existing_pp.split(os.pathsep):
        os.environ["PYTHONPATH"] = (
            yolopose_dir + os.pathsep + existing_pp if existing_pp else yolopose_dir
        )


DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "train_config.yaml"


def _disable_opencv_threads():
    """Avoid CPU oversubscription inside PyTorch DataLoader workers."""
    try:
        import cv2
    except ImportError:
        return

    cv2.setNumThreads(0)
    try:
        cv2.ocl.setUseOpenCL(False)
    except AttributeError:
        pass


_disable_opencv_threads()


def _load_config(path):
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing train config: {config_path}")

    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for --config. Install with: pip install pyyaml"
        ) from exc

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Train config must be a YAML mapping: {config_path}")
    config["_config_path"] = str(config_path)
    return config


def _set_nested(config, keys, value):
    current = config
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


def _none_if_empty(value):
    if isinstance(value, str) and value.lower() in {"", "none", "null"}:
        return None
    return value


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


def _section(config, key):
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{key}' must be a mapping")
    return value


def _make_transform(factory, candidates, name):
    last_error = None
    for kwargs in candidates:
        try:
            return factory(**kwargs)
        except TypeError as exc:
            last_error = exc
    print(
        f"[train] 警告: {name} 与当前 albumentations 版本不兼容，已跳过: {last_error}"
    )
    return None


def _build_albumentations_transforms(config):
    if not config.get("enabled", True):
        return None

    try:
        os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
        import albumentations as A
    except ImportError:
        print("[train] 警告: 未安装 albumentations，跳过自定义 Albumentations 增强")
        return None

    transforms = []
    transform_registry = {}
    oneof_groups = config.get("oneof_groups") or []
    use_oneof_groups = bool(oneof_groups)
    skipped = []
    pose_unsafe_spatial = {"CoarseDropout", "GridDropout", "PixelDropout"}

    def normalize_key(value):
        return str(value).replace("-", "_").replace(" ", "_").lower()

    def register_transform(name, p_key, transform):
        keys = {
            name,
            normalize_key(name),
            p_key,
            normalize_key(p_key),
            p_key[:-2] if p_key.endswith("_p") else p_key,
        }
        for key in keys:
            transform_registry[normalize_key(key)] = transform

    def add_transform(name, candidates, p_key):
        p = float(config.get(p_key, 0.0))
        if p <= 0:
            return
        if name in pose_unsafe_spatial:
            skipped.append(f"{name}(pose keypoints unsafe)")
            return

        factory = getattr(A, name, None)
        if factory is None:
            skipped.append(name)
            return

        transform = _make_transform(factory, candidates, name)
        if transform is None:
            skipped.append(name)
            return
        if use_oneof_groups:
            register_transform(name, p_key, transform)
        else:
            transforms.append(transform)

    def build_oneof_groups():
        if not isinstance(oneof_groups, list):
            raise ValueError("albumentations.oneof_groups must be a list")

        grouped_transforms = []
        for group_idx, group in enumerate(oneof_groups):
            if not isinstance(group, dict):
                raise ValueError("Each albumentations.oneof_groups item must be a mapping")

            group_name = group.get("name", f"group_{group_idx}")
            group_p = float(group.get("p", 0.0))
            if group_p <= 0:
                continue

            members = []
            for item in group.get("transforms", []):
                if isinstance(item, dict):
                    item_name = item.get("name")
                    weight = item.get("weight")
                else:
                    item_name = item
                    weight = None
                if not item_name:
                    continue

                transform = transform_registry.get(normalize_key(item_name))
                if transform is None:
                    skipped.append(f"{group_name}:{item_name}")
                    continue
                if weight is not None:
                    weight = float(weight)
                    if weight <= 0:
                        continue
                    transform.p = weight
                members.append(transform)

            if members:
                grouped_transforms.append(A.OneOf(members, p=group_p))
            else:
                skipped.append(f"{group_name}(empty)")

        return grouped_transforms

    add_transform(
        "GaussianBlur",
        [
            {
                "blur_limit": tuple(config["gaussian_blur_limit"]),
                "p": config["gaussian_blur_p"],
            }
        ],
        "gaussian_blur_p",
    )
    add_transform(
        "MotionBlur",
        [
            {
                "blur_limit": config["motion_blur_limit"],
                "p": config["motion_blur_p"],
            }
        ],
        "motion_blur_p",
    )
    add_transform(
        "MedianBlur",
        [
            {
                "blur_limit": config["median_blur_limit"],
                "p": config["median_blur_p"],
            }
        ],
        "median_blur_p",
    )
    add_transform(
        "AdvancedBlur",
        [
            {
                "blur_limit": tuple(config.get("advanced_blur_limit", [3, 7])),
                "sigma_x_limit": tuple(
                    config.get("advanced_blur_sigma_x_limit", [0.2, 1.0])
                ),
                "sigma_y_limit": tuple(
                    config.get("advanced_blur_sigma_y_limit", [0.2, 1.0])
                ),
                "rotate_limit": tuple(
                    config.get("advanced_blur_rotate_limit", [-90, 90])
                ),
                "beta_limit": tuple(
                    config.get("advanced_blur_beta_limit", [0.5, 8.0])
                ),
                "noise_limit": tuple(
                    config.get("advanced_blur_noise_limit", [0.9, 1.1])
                ),
                "p": config.get("advanced_blur_p", 0.0),
            }
        ],
        "advanced_blur_p",
    )
    add_transform(
        "Defocus",
        [
            {
                "radius": tuple(config.get("defocus_radius", [3, 10])),
                "alias_blur": tuple(config.get("defocus_alias_blur", [0.1, 0.5])),
                "p": config.get("defocus_p", 0.0),
            }
        ],
        "defocus_p",
    )
    add_transform(
        "ChromaticAberration",
        [
            {
                "primary_distortion_limit": tuple(
                    config.get("chromatic_primary_distortion_limit", [-0.02, 0.02])
                ),
                "secondary_distortion_limit": tuple(
                    config.get("chromatic_secondary_distortion_limit", [-0.05, 0.05])
                ),
                "mode": config.get("chromatic_mode", "red_blue"),
                "p": config.get("chromatic_aberration_p", 0.0),
            }
        ],
        "chromatic_aberration_p",
    )
    add_transform(
        "RandomShadow",
        [
            {
                "num_shadows_limit": tuple(config["shadow_num_shadows"]),
                "p": config["shadow_p"],
            },
            {
                "num_shadows_lower": config["shadow_num_shadows"][0],
                "num_shadows_upper": config["shadow_num_shadows"][1],
                "p": config["shadow_p"],
            }
        ],
        "shadow_p",
    )
    add_transform(
        "PlasmaShadow",
        [
            {
                "shadow_intensity_range": tuple(
                    config.get("plasma_shadow_intensity_range", [0.3, 0.7])
                ),
                "plasma_size": int(config.get("plasma_shadow_size", 256)),
                "roughness": float(config.get("plasma_shadow_roughness", 3.0)),
                "p": config.get("plasma_shadow_p", 0.0),
            }
        ],
        "plasma_shadow_p",
    )
    add_transform(
        "RandomBrightnessContrast",
        [
            {
                "brightness_limit": tuple(config["brightness_limit"]),
                "contrast_limit": tuple(config["contrast_limit"]),
                "p": config["brightness_contrast_p"],
            }
        ],
        "brightness_contrast_p",
    )
    add_transform(
        "AutoContrast",
        [
            {
                "cutoff": float(config.get("autocontrast_cutoff", 0.0)),
                "method": config.get("autocontrast_method", "cdf"),
                "p": config.get("autocontrast_p", 0.0),
            }
        ],
        "autocontrast_p",
    )
    add_transform(
        "Equalize",
        [
            {
                "mode": config.get("equalize_mode", "cv"),
                "by_channels": config.get("equalize_by_channels", True),
                "p": config.get("equalize_p", 0.0),
            }
        ],
        "equalize_p",
    )
    add_transform(
        "Downscale",
        [
            {
                "scale_range": tuple(config["downscale_range"]),
                "p": config["downscale_p"],
            },
            {
                "scale_min": config["downscale_range"][0],
                "scale_max": config["downscale_range"][1],
                "p": config["downscale_p"],
            },
        ],
        "downscale_p",
    )
    add_transform(
        "RandomFog",
        [
            {
                "fog_coef_range": tuple(config["fog_coef_range"]),
                "p": config["fog_p"],
            },
            {
                "fog_coef_lower": config["fog_coef_range"][0],
                "fog_coef_upper": config["fog_coef_range"][1],
                "p": config["fog_p"],
            },
        ],
        "fog_p",
    )
    add_transform(
        "CLAHE",
        [
            {
                "clip_limit": tuple(config["clahe_clip_limit"]),
                "tile_grid_size": tuple(config["clahe_grid"]),
                "p": config["clahe_p"],
            }
        ],
        "clahe_p",
    )
    add_transform(
        "RandomGamma",
        [
            {
                "gamma_limit": tuple(config["gamma_limit"]),
                "p": config["gamma_p"],
            }
        ],
        "gamma_p",
    )
    add_transform(
        "PlanckianJitter",
        [
            {
                "mode": config.get("planckian_mode", "blackbody"),
                "temperature_limit": tuple(
                    config.get("planckian_temperature_limit", [3000, 15000])
                ),
                "sampling_method": config.get("planckian_sampling_method", "uniform"),
                "p": config.get("planckian_jitter_p", 0.0),
            }
        ],
        "planckian_jitter_p",
    )
    add_transform(
        "PixelDropout",
        [
            {
                "dropout_prob": config["pixel_dropout_prob"],
                "per_channel": config["pixel_dropout_per_channel"],
                "p": config["pixel_dropout_p"],
            }
        ],
        "pixel_dropout_p",
    )
    add_transform(
        "GridDropout",
        [
            {
                "ratio": config["grid_dropout_ratio"],
                "unit_size_range": tuple(config["grid_dropout_unit_size"]),
                "p": config["grid_dropout_p"],
            },
            {
                "ratio": config["grid_dropout_ratio"],
                "unit_size_min": config["grid_dropout_unit_size"][0],
                "unit_size_max": config["grid_dropout_unit_size"][1],
                "p": config["grid_dropout_p"],
            },
        ],
        "grid_dropout_p",
    )
    add_transform(
        "ChannelDropout",
        [
            {
                "channel_drop_range": tuple(config["channel_drop_range"]),
                "fill": config["channel_drop_fill"],
                "p": config["channel_dropout_p"],
            },
            {
                "channel_drop_range": tuple(config["channel_drop_range"]),
                "fill_value": config["channel_drop_fill"],
                "p": config["channel_dropout_p"],
            },
        ],
        "channel_dropout_p",
    )

    gauss_noise_var_min = float(config["gauss_noise_var_limit"][0])
    gauss_noise_var_max = float(config["gauss_noise_var_limit"][1])
    gauss_noise_std_range = (
        math.sqrt(max(gauss_noise_var_min, 0.0)) / 255.0,
        math.sqrt(max(gauss_noise_var_max, 0.0)) / 255.0,
    )
    gauss_noise_mean = float(config["gauss_noise_mean"])
    add_transform(
        "GaussNoise",
        [
            {
                "std_range": gauss_noise_std_range,
                "mean_range": (gauss_noise_mean / 255.0, gauss_noise_mean / 255.0),
                "p": config["gauss_noise_p"],
            },
            {
                "var_limit": tuple(config["gauss_noise_var_limit"]),
                "mean": gauss_noise_mean,
                "p": config["gauss_noise_p"],
            },
        ],
        "gauss_noise_p",
    )
    add_transform(
        "ISONoise",
        [
            {
                "color_shift": tuple(config["iso_noise_color_shift"]),
                "intensity": tuple(config["iso_noise_intensity"]),
                "p": config["iso_noise_p"],
            }
        ],
        "iso_noise_p",
    )
    add_transform(
        "ShotNoise",
        [
            {
                "scale_range": tuple(config.get("shot_noise_scale_range", [0.1, 0.3])),
                "p": config.get("shot_noise_p", 0.0),
            }
        ],
        "shot_noise_p",
    )
    add_transform(
        "SaltAndPepper",
        [
            {
                "amount": tuple(config.get("salt_pepper_amount", [0.01, 0.06])),
                "salt_vs_pepper": tuple(
                    config.get("salt_pepper_salt_vs_pepper", [0.4, 0.6])
                ),
                "p": config.get("salt_pepper_p", 0.0),
            }
        ],
        "salt_pepper_p",
    )
    add_transform(
        "MultiplicativeNoise",
        [
            {
                "multiplier": tuple(
                    config.get("multiplicative_noise_multiplier", [0.9, 1.1])
                ),
                "per_channel": config.get("multiplicative_noise_per_channel", False),
                "elementwise": config.get("multiplicative_noise_elementwise", False),
                "p": config.get("multiplicative_noise_p", 0.0),
            }
        ],
        "multiplicative_noise_p",
    )

    add_transform(
        "CoarseDropout",
        [
            {
                "num_holes_range": tuple(config["coarse_dropout_holes"]),
                "hole_height_range": tuple(config["coarse_dropout_height"]),
                "hole_width_range": tuple(config["coarse_dropout_width"]),
                "fill": config["coarse_dropout_fill"],
                "p": config["coarse_dropout_p"],
            },
            {
                "min_holes": config["coarse_dropout_holes"][0],
                "max_holes": config["coarse_dropout_holes"][1],
                "min_height": config["coarse_dropout_height"][0],
                "max_height": config["coarse_dropout_height"][1],
                "min_width": config["coarse_dropout_width"][0],
                "max_width": config["coarse_dropout_width"][1],
                "fill_value": config["coarse_dropout_fill"],
                "p": config["coarse_dropout_p"],
            },
        ],
        "coarse_dropout_p",
    )

    quality_min = int(config["image_compression_quality"][0])
    quality_max = int(config["image_compression_quality"][1])
    add_transform(
        "ImageCompression",
        [
            {
                "quality_range": (quality_min, quality_max),
                "p": config["image_compression_p"],
            },
            {
                "quality_lower": quality_min,
                "quality_upper": quality_max,
                "p": config["image_compression_p"],
            },
        ],
        "image_compression_p",
    )
    add_transform(
        "RingingOvershoot",
        [
            {
                "blur_limit": tuple(config.get("ringing_blur_limit", [7, 15])),
                "cutoff": tuple(config.get("ringing_cutoff", [0.785, 1.571])),
                "p": config.get("ringing_overshoot_p", 0.0),
            }
        ],
        "ringing_overshoot_p",
    )
    add_transform(
        "Posterize",
        [
            {
                "num_bits": int(config["posterize_num_bits"]),
                "p": config["posterize_p"],
            }
        ],
        "posterize_p",
    )
    add_transform(
        "RandomToneCurve",
        [
            {
                "scale": config["tone_curve_scale"],
                "per_channel": config["tone_curve_per_channel"],
                "p": config["tone_curve_p"],
            },
            {
                "scale": config["tone_curve_scale"],
                "p": config["tone_curve_p"],
            },
        ],
        "tone_curve_p",
    )
    add_transform(
        "ColorJitter",
        [
            {
                "brightness": config["color_jitter_brightness"],
                "contrast": config["color_jitter_contrast"],
                "saturation": config["color_jitter_saturation"],
                "hue": config["color_jitter_hue"],
                "p": config["color_jitter_p"],
            }
        ],
        "color_jitter_p",
    )
    add_transform(
        "HueSaturationValue",
        [
            {
                "hue_shift_limit": config["hsv_hue_shift_limit"],
                "sat_shift_limit": config["hsv_sat_shift_limit"],
                "val_shift_limit": config["hsv_val_shift_limit"],
                "p": config["hsv_p"],
            }
        ],
        "hsv_p",
    )
    add_transform(
        "RGBShift",
        [
            {
                "r_shift_limit": config["rgb_shift_limit"][0],
                "g_shift_limit": config["rgb_shift_limit"][1],
                "b_shift_limit": config["rgb_shift_limit"][2],
                "p": config["rgb_shift_p"],
            }
        ],
        "rgb_shift_p",
    )
    add_transform(
        "ChannelShuffle",
        [{"p": config["channel_shuffle_p"]}],
        "channel_shuffle_p",
    )
    add_transform(
        "Sharpen",
        [
            {
                "alpha": tuple(config["sharpen_alpha"]),
                "lightness": tuple(config["sharpen_lightness"]),
                "p": config["sharpen_p"],
            }
        ],
        "sharpen_p",
    )
    add_transform(
        "UnsharpMask",
        [
            {
                "blur_limit": tuple(config["unsharp_mask_blur_limit"]),
                "sigma_limit": tuple(config["unsharp_mask_sigma_limit"]),
                "alpha": tuple(config["unsharp_mask_alpha"]),
                "threshold": int(config["unsharp_mask_threshold"]),
                "p": config["unsharp_mask_p"],
            }
        ],
        "unsharp_mask_p",
    )

    if use_oneof_groups:
        transforms = build_oneof_groups()

    if skipped:
        print(
            "[train] 提示: 以下 Albumentations 变换不可用，已自动跳过: "
            + ", ".join(skipped)
        )
    if not transforms:
        print("[train] 警告: 未构建出任何 Albumentations 变换，继续使用原始训练流程")
        return None

    def describe_transform(tf):
        if tf.__class__.__name__ == "OneOf":
            members = "/".join(t.__class__.__name__ for t in tf.transforms)
            return f"OneOf(p={tf.p}: {members})"
        return f"{tf.__class__.__name__}(p={tf.p})"

    active = ", ".join(describe_transform(tf) for tf in transforms)
    print(f"[train] Albumentations transforms: {active}")
    return transforms


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO11-pose 装配柱关键点训练")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="统一训练配置 YAML，默认 train_config.yaml",
    )
    parser.add_argument("--model", type=str, default=None, help="覆盖 YAML model")
    parser.add_argument("--data", type=str, default=None, help="覆盖 YAML data")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖 YAML train.epochs")
    parser.add_argument("--batch", type=int, default=None, help="覆盖 YAML train.batch")
    parser.add_argument("--imgsz", type=int, default=None, help="覆盖 YAML train.imgsz")
    parser.add_argument("--device", type=str, default=None, help="覆盖 YAML train.device")
    parser.add_argument("--workers", type=int, default=None, help="覆盖 YAML train.workers")
    parser.add_argument("--project", type=str, default=None, help="覆盖 YAML train.project")
    parser.add_argument("--name", type=str, default=None, help="覆盖 YAML train.name")
    parser.add_argument("--lr0", "--lr", dest="lr0", type=float, default=None, help="覆盖 YAML train.lr0")
    parser.add_argument("--patience", type=int, default=None, help="覆盖 YAML train.patience")
    parser.add_argument("--optimizer", type=str, default=None, help="覆盖 YAML train.optimizer")
    parser.add_argument("--resume", type=str, default=None, help="覆盖 YAML resume")
    parser.add_argument(
        "--export_openvino",
        dest="export_openvino",
        action="store_true",
        default=None,
        help="覆盖 YAML export_openvino=true",
    )
    parser.add_argument(
        "--no_export_openvino",
        dest="export_openvino",
        action="store_false",
        help="覆盖 YAML export_openvino=false",
    )
    parser.add_argument(
        "--random_crop",
        dest="random_crop_enabled",
        action="store_true",
        default=None,
        help="覆盖 YAML random_crop.enabled=true",
    )
    parser.add_argument(
        "--no_random_crop",
        dest="random_crop_enabled",
        action="store_false",
        help="覆盖 YAML random_crop.enabled=false",
    )
    parser.add_argument(
        "--crop_scale",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="覆盖 YAML random_crop.scale",
    )
    parser.add_argument("--crop_p", type=float, default=None, help="覆盖 YAML random_crop.p")
    parser.add_argument(
        "--crop_min_bbox_area_ratio",
        type=float,
        default=None,
        help="覆盖 YAML random_crop.min_bbox_area_ratio",
    )
    parser.add_argument(
        "--random_mask",
        dest="random_mask_enabled",
        action="store_true",
        default=None,
        help="覆盖 YAML random_mask.enabled=true",
    )
    parser.add_argument(
        "--no_random_mask",
        dest="random_mask_enabled",
        action="store_false",
        help="覆盖 YAML random_mask.enabled=false",
    )
    parser.add_argument("--mask_p", type=float, default=None, help="覆盖 YAML random_mask.p")
    parser.add_argument(
        "--mask_max_bbox_overlap",
        type=float,
        default=None,
        help="覆盖 YAML random_mask.max_bbox_overlap",
    )
    parser.add_argument(
        "--mask_max_attempts",
        type=int,
        default=None,
        help="覆盖 YAML random_mask.max_attempts",
    )
    parser.add_argument(
        "--albumentations",
        dest="albumentations_enabled",
        action="store_true",
        default=None,
        help="覆盖 YAML albumentations.enabled=true",
    )
    parser.add_argument(
        "--no_albumentations",
        dest="albumentations_enabled",
        action="store_false",
        help="覆盖 YAML albumentations.enabled=false",
    )
    return parser.parse_args()


def _build_train_kwargs(config):
    train_cfg = _section(config, "train")
    yolo_aug_cfg = _section(config, "yolo_aug")
    loss_cfg = _section(config, "loss")

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
    train_kwargs.update(loss_cfg)
    return {k: v for k, v in train_kwargs.items() if v is not None}


def _model_path(config):
    resume = config.get("resume")
    if resume:
        return resume

    model = str(config.get("model", "yolo11n-pose"))
    if model.endswith((".pt", ".yaml", ".yml")):
        return model
    return f"{model}.pt"


def train(config):
    from ultralytics import YOLO

    # ─── 加载模型 ─────────────────────────────────────────────────────────────
    # resume 时必须把 last.pt 作为模型加载，optimizer 状态才会被恢复
    if config.get("resume"):
        ckpt = config["resume"]
        print(f"[train] Resume from checkpoint: {ckpt}")
        model = YOLO(ckpt)
    else:
        model_name = _model_path(config)
        print(f"[train] 加载模型: {model_name}")
        model = YOLO(model_name)

    # ─── 训练超参数 ───────────────────────────────────────────────────────────
    # 几何增强继续走 Ultralytics 内置管线；自定义 Albumentations 只做 image-only 变化
    train_kwargs = _build_train_kwargs(config)

    custom_augmentations = _build_albumentations_transforms(
        _section(config, "albumentations")
    )
    if custom_augmentations is not None:
        train_kwargs["augmentations"] = custom_augmentations
        print("[train] 自定义 Albumentations 已启用，保留 YOLO 自带几何增强")

    if config.get("resume"):
        # resume 时仍显式回传 augmentations，避免官方恢复流程丢失自定义增强
        train_kwargs["resume"] = True

    print(
        f"[train] 开始训练: {train_kwargs.get('epochs')} epoch, "
        f"batch={train_kwargs.get('batch')}, imgsz={train_kwargs.get('imgsz')}"
    )

    random_crop = _section(config, "random_crop")
    random_mask = _section(config, "random_mask")
    crop_enabled = bool(random_crop.get("enabled"))
    mask_enabled = bool(random_mask.get("enabled"))
    if crop_enabled or mask_enabled:
        _add_yolopose_dir_to_pythonpath()
        from random_crop_aug import make_pose_aug_trainer

        crop_scale = tuple(random_crop.get("scale", [0.5, 1.0]))
        crop_p = random_crop.get("p", 0.5)
        crop_min_bbox_area_ratio = random_crop.get("min_bbox_area_ratio", 0.1)
        if crop_enabled:
            print(
                f"[train] RandomCrop 已启用: scale={crop_scale}, p={crop_p}, "
                f"min_bbox_area_ratio={crop_min_bbox_area_ratio}"
            )

        mask_holes = tuple(random_mask.get("holes", [1, 2]))
        mask_height = tuple(random_mask.get("height", [0.1, 0.3]))
        mask_width = tuple(random_mask.get("width", [0.1, 0.3]))
        mask_fill = random_mask.get("fill", 0)
        mask_random_fill_p = random_mask.get("random_fill_p", 0.0)
        mask_max_bbox_overlap = random_mask.get("max_bbox_overlap", 1.0)
        mask_max_attempts = random_mask.get("max_attempts", 20)
        mask_p = random_mask.get("p", 0.2)
        if mask_enabled:
            print(
                "[train] RandomMask 已启用: "
                f"p={mask_p}, holes={mask_holes}, height={mask_height}, "
                f"width={mask_width}, fill={mask_fill}, "
                f"random_fill_p={mask_random_fill_p}, "
                f"max_bbox_overlap={mask_max_bbox_overlap}, "
                f"max_attempts={mask_max_attempts}"
            )

        # PoseTrainer 需要在 overrides 中传入 model 路径（字符串），而不是 YOLO 对象
        train_kwargs["model"] = _model_path(config)
        PoseAugTrainer = make_pose_aug_trainer(
            crop_enabled=crop_enabled,
            crop_scale=crop_scale,
            crop_p=crop_p,
            crop_min_bbox_area_ratio=crop_min_bbox_area_ratio,
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
        trainer = PoseAugTrainer(overrides=train_kwargs)
        trainer.train()
        results = trainer
    else:
        results = model.train(**train_kwargs)

    # ─── 验证 ─────────────────────────────────────────────────────────────────
    project = train_kwargs.get("project", "train_files/yoloposev11")
    name = train_kwargs.get("name", "pillar")
    data = train_kwargs.get("data")
    imgsz = train_kwargs.get("imgsz", 640)
    best_weights = Path(project) / name / "weights" / "best.pt"
    if best_weights.exists():
        print(f"\n[train] 验证最优权重: {best_weights}")
        model_best = YOLO(str(best_weights))
        metrics = model_best.val(data=data, imgsz=imgsz)
        print(f"  Pose mAP50:    {metrics.pose.map50:.4f}")
        print(f"  Pose mAP50-95: {metrics.pose.map:.4f}")
        print(f"  Box  mAP50:    {metrics.box.map50:.4f}")
    else:
        print("[train] 警告: 未找到 best.pt")
        best_weights = Path(project) / name / "weights" / "last.pt"

    # ─── 导出 OpenVINO ────────────────────────────────────────────────────────
    if config.get("export_openvino") and best_weights.exists():
        print(f"\n[train] 导出 OpenVINO FP16: {best_weights}")
        model_export = YOLO(str(best_weights))
        model_export.export(
            format="openvino",
            imgsz=imgsz,
            half=True,  # FP16
            dynamic=False,
            nms=True,  # 内置 NMS
        )
        ov_dir = best_weights.parent / f"best_openvino_model"
        print(f"  → {ov_dir}")

    print("\n[train] 完成!")
    return results


if __name__ == "__main__":
    args = parse_args()
    cfg = _apply_cli_overrides(_load_config(args.config), args)
    print(f"[train] 使用配置: {cfg.get('_config_path')}")
    train(cfg)
