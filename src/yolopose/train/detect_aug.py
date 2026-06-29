"""Custom bbox-safe augmentations for YOLO detect training."""

from random_crop_aug import (
    build_transforms_with_box_crop,
    build_transforms_with_crop,
    build_transforms_with_crop_oneof,
    build_transforms_with_mask,
)


try:
    from ultralytics.models.yolo.detect import DetectionTrainer as _DetectionTrainer

    class MaskDetectTrainer(_DetectionTrainer):
        """DetectionTrainer with optional bbox crop and image-only mask augmentation."""

        CROP_ENABLED: bool = False
        CROP_SCALE: tuple = (0.65, 1.0)
        CROP_P: float = 0.3
        CROP_MIN_BBOX_AREA_RATIO: float = 0.05
        CROP_ONEOF_ENABLED: bool = False
        CROP_ONEOF_P: float = 0.4
        CROP_ONEOF_RANDOM_WEIGHT: float = 1.0
        CROP_ONEOF_BOX_WEIGHT: float = 1.0
        BOX_CROP_ENABLED: bool = False
        BOX_CROP_P: float = 0.2
        BOX_CROP_TARGET_CLASSES: tuple = ()
        BOX_CROP_SCALE: tuple = (1.3, 3.0)
        BOX_CROP_CENTER_JITTER: float = 0.15
        BOX_CROP_MIN_BBOX_AREA_RATIO: float = 0.05
        BOX_CROP_MIN_CROP_SIZE_RATIO: float = 0.25
        MASK_ENABLED: bool = False
        MASK_P: float = 0.1
        MASK_HOLES: tuple = (1, 2)
        MASK_HEIGHT: tuple = (0.04, 0.15)
        MASK_WIDTH: tuple = (0.04, 0.15)
        MASK_FILL = 0
        MASK_RANDOM_FILL_P: float = 0.0
        MASK_MAX_BBOX_OVERLAP: float = 1.0
        MASK_MAX_ATTEMPTS: int = 20

        def build_dataset(self, img_path, mode="train", batch=None):
            dataset = super().build_dataset(img_path, mode, batch)
            if mode == "train":
                import os as _os

                crop_enabled = _env_bool("YOLO_CROP_ENABLED", self.__class__.CROP_ENABLED)
                box_crop_enabled = _env_bool(
                    "YOLO_BOX_CROP_ENABLED", self.__class__.BOX_CROP_ENABLED
                )
                crop_scale = _env_tuple("YOLO_CROP_SCALE", self.__class__.CROP_SCALE)
                crop_p = float(_os.environ.get("YOLO_CROP_P", str(self.__class__.CROP_P)))
                crop_min_bbox_area_ratio = float(
                    _os.environ.get(
                        "YOLO_CROP_MIN_BBOX_AREA_RATIO",
                        str(self.__class__.CROP_MIN_BBOX_AREA_RATIO),
                    )
                )
                box_crop_p = float(
                    _os.environ.get("YOLO_BOX_CROP_P", str(self.__class__.BOX_CROP_P))
                )
                box_crop_target_classes = tuple(
                    _env_json(
                        "YOLO_BOX_CROP_TARGET_CLASSES",
                        self.__class__.BOX_CROP_TARGET_CLASSES,
                    )
                )
                box_crop_scale = _env_tuple(
                    "YOLO_BOX_CROP_SCALE", self.__class__.BOX_CROP_SCALE
                )
                box_crop_center_jitter = float(
                    _os.environ.get(
                        "YOLO_BOX_CROP_CENTER_JITTER",
                        str(self.__class__.BOX_CROP_CENTER_JITTER),
                    )
                )
                box_crop_min_bbox_area_ratio = float(
                    _os.environ.get(
                        "YOLO_BOX_CROP_MIN_BBOX_AREA_RATIO",
                        str(self.__class__.BOX_CROP_MIN_BBOX_AREA_RATIO),
                    )
                )
                box_crop_min_crop_size_ratio = float(
                    _os.environ.get(
                        "YOLO_BOX_CROP_MIN_CROP_SIZE_RATIO",
                        str(self.__class__.BOX_CROP_MIN_CROP_SIZE_RATIO),
                    )
                )

                if _env_bool("YOLO_CROP_ONEOF_ENABLED", self.__class__.CROP_ONEOF_ENABLED):
                    oneof_p = float(
                        _os.environ.get(
                            "YOLO_CROP_ONEOF_P", str(self.__class__.CROP_ONEOF_P)
                        )
                    )
                    random_weight = float(
                        _os.environ.get(
                            "YOLO_CROP_ONEOF_RANDOM_WEIGHT",
                            str(self.__class__.CROP_ONEOF_RANDOM_WEIGHT),
                        )
                    )
                    box_weight = float(
                        _os.environ.get(
                            "YOLO_CROP_ONEOF_BOX_WEIGHT",
                            str(self.__class__.CROP_ONEOF_BOX_WEIGHT),
                        )
                    )
                    dataset.transforms = build_transforms_with_crop_oneof(
                        dataset,
                        p=oneof_p,
                        random_crop_enabled=crop_enabled,
                        random_crop_weight=random_weight,
                        random_crop_scale=crop_scale,
                        random_crop_min_bbox_area_ratio=crop_min_bbox_area_ratio,
                        box_crop_enabled=box_crop_enabled,
                        box_crop_weight=box_weight,
                        box_crop_target_classes=box_crop_target_classes,
                        box_crop_scale=box_crop_scale,
                        box_crop_center_jitter=box_crop_center_jitter,
                        box_crop_min_bbox_area_ratio=box_crop_min_bbox_area_ratio,
                        box_crop_min_crop_size_ratio=box_crop_min_crop_size_ratio,
                    )
                    print(
                        f"[MaskDetectTrainer] Injected Crop OneOf "
                        f"(p={oneof_p}, random_weight={random_weight}, "
                        f"box_weight={box_weight})"
                    )
                else:
                    if crop_enabled:
                        dataset.transforms = build_transforms_with_crop(
                            dataset,
                            crop_scale=crop_scale,
                            p=crop_p,
                            min_bbox_area_ratio=crop_min_bbox_area_ratio,
                        )
                        print(
                            f"[MaskDetectTrainer] Injected RandomCropPose "
                            f"(crop_scale={crop_scale}, p={crop_p}, "
                            f"min_bbox_area_ratio={crop_min_bbox_area_ratio})"
                        )
                    if box_crop_enabled:
                        dataset.transforms = build_transforms_with_box_crop(
                            dataset,
                            p=box_crop_p,
                            target_classes=box_crop_target_classes,
                            crop_scale=box_crop_scale,
                            center_jitter=box_crop_center_jitter,
                            min_bbox_area_ratio=box_crop_min_bbox_area_ratio,
                            min_crop_size_ratio=box_crop_min_crop_size_ratio,
                        )
                        print(
                            f"[MaskDetectTrainer] Injected RandomBoxCrop "
                            f"(p={box_crop_p}, target_classes={box_crop_target_classes}, "
                            f"crop_scale={box_crop_scale}, "
                            f"center_jitter={box_crop_center_jitter}, "
                            f"min_bbox_area_ratio={box_crop_min_bbox_area_ratio}, "
                            f"min_crop_size_ratio={box_crop_min_crop_size_ratio})"
                        )

                if not _env_bool("YOLO_MASK_ENABLED", self.__class__.MASK_ENABLED):
                    return dataset

                mask_p = float(_os.environ.get("YOLO_MASK_P", str(self.__class__.MASK_P)))
                holes = tuple(_env_json("YOLO_MASK_HOLES", self.__class__.MASK_HOLES))
                height = tuple(_env_json("YOLO_MASK_HEIGHT", self.__class__.MASK_HEIGHT))
                width = tuple(_env_json("YOLO_MASK_WIDTH", self.__class__.MASK_WIDTH))
                fill = _env_json("YOLO_MASK_FILL", self.__class__.MASK_FILL)
                random_fill_p = float(
                    _os.environ.get(
                        "YOLO_MASK_RANDOM_FILL_P",
                        str(self.__class__.MASK_RANDOM_FILL_P),
                    )
                )
                max_bbox_overlap = float(
                    _os.environ.get(
                        "YOLO_MASK_MAX_BBOX_OVERLAP",
                        str(self.__class__.MASK_MAX_BBOX_OVERLAP),
                    )
                )
                max_attempts = int(
                    _os.environ.get(
                        "YOLO_MASK_MAX_ATTEMPTS",
                        str(self.__class__.MASK_MAX_ATTEMPTS),
                    )
                )
                dataset.transforms = build_transforms_with_mask(
                    dataset,
                    p=mask_p,
                    holes=holes,
                    height=height,
                    width=width,
                    fill=fill,
                    random_fill_p=random_fill_p,
                    max_bbox_overlap=max_bbox_overlap,
                    max_attempts=max_attempts,
                )
                print(
                    f"[MaskDetectTrainer] Injected RandomMaskImageOnly "
                    f"(p={mask_p}, holes={holes}, height={height}, "
                    f"width={width}, fill={fill}, random_fill_p={random_fill_p}, "
                    f"max_bbox_overlap={max_bbox_overlap}, "
                    f"max_attempts={max_attempts})"
                )
            return dataset

    # Ultralytics DDP serializes the trainer class by module path, then imports
    # it in fresh worker processes. Keep the path stable and importable.
    MaskDetectTrainer.__module__ = "detect_aug"

except ImportError:
    MaskDetectTrainer = None  # type: ignore[assignment,misc]


def _env_bool(name, default=False):
    import os as _os

    value = _os.environ.get(name)
    if value is None:
        return bool(default)
    return value.lower() in {"1", "true", "yes", "on"}


def _env_json(name, default):
    import json
    import os as _os

    value = _os.environ.get(name)
    if value is None:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _env_tuple(name, default):
    return tuple(_env_json(name, default))


def _set_env_json(name, value):
    import json
    import os as _os

    _os.environ[name] = json.dumps(value)


def make_mask_detect_trainer(
    crop_enabled=False,
    crop_scale=(0.65, 1.0),
    crop_p=0.3,
    crop_min_bbox_area_ratio=0.05,
    crop_oneof_enabled=False,
    crop_oneof_p=0.4,
    crop_oneof_random_weight=1.0,
    crop_oneof_box_weight=1.0,
    box_crop_enabled=False,
    box_crop_p=0.2,
    box_crop_target_classes=(),
    box_crop_scale=(1.3, 3.0),
    box_crop_center_jitter=0.15,
    box_crop_min_bbox_area_ratio=0.05,
    box_crop_min_crop_size_ratio=0.25,
    mask_enabled=False,
    mask_p=0.1,
    mask_holes=(1, 2),
    mask_height=(0.04, 0.15),
    mask_width=(0.04, 0.15),
    mask_fill=0,
    mask_random_fill_p=0.0,
    mask_max_bbox_overlap=1.0,
    mask_max_attempts=20,
):
    """Configure and return the module-level MaskDetectTrainer class."""
    import os as _os

    if MaskDetectTrainer is None:
        raise ImportError("ultralytics is required for MaskDetectTrainer")

    MaskDetectTrainer.CROP_ENABLED = bool(crop_enabled)
    MaskDetectTrainer.CROP_SCALE = tuple(crop_scale)
    MaskDetectTrainer.CROP_P = float(crop_p)
    MaskDetectTrainer.CROP_MIN_BBOX_AREA_RATIO = float(crop_min_bbox_area_ratio)
    MaskDetectTrainer.CROP_ONEOF_ENABLED = bool(crop_oneof_enabled)
    MaskDetectTrainer.CROP_ONEOF_P = float(crop_oneof_p)
    MaskDetectTrainer.CROP_ONEOF_RANDOM_WEIGHT = float(crop_oneof_random_weight)
    MaskDetectTrainer.CROP_ONEOF_BOX_WEIGHT = float(crop_oneof_box_weight)
    MaskDetectTrainer.BOX_CROP_ENABLED = bool(box_crop_enabled)
    MaskDetectTrainer.BOX_CROP_P = float(box_crop_p)
    MaskDetectTrainer.BOX_CROP_TARGET_CLASSES = tuple(box_crop_target_classes or ())
    MaskDetectTrainer.BOX_CROP_SCALE = tuple(box_crop_scale)
    MaskDetectTrainer.BOX_CROP_CENTER_JITTER = float(box_crop_center_jitter)
    MaskDetectTrainer.BOX_CROP_MIN_BBOX_AREA_RATIO = float(box_crop_min_bbox_area_ratio)
    MaskDetectTrainer.BOX_CROP_MIN_CROP_SIZE_RATIO = float(box_crop_min_crop_size_ratio)
    MaskDetectTrainer.MASK_ENABLED = bool(mask_enabled)
    MaskDetectTrainer.MASK_P = float(mask_p)
    MaskDetectTrainer.MASK_HOLES = tuple(mask_holes)
    MaskDetectTrainer.MASK_HEIGHT = tuple(mask_height)
    MaskDetectTrainer.MASK_WIDTH = tuple(mask_width)
    MaskDetectTrainer.MASK_FILL = mask_fill
    MaskDetectTrainer.MASK_RANDOM_FILL_P = float(mask_random_fill_p)
    MaskDetectTrainer.MASK_MAX_BBOX_OVERLAP = float(mask_max_bbox_overlap)
    MaskDetectTrainer.MASK_MAX_ATTEMPTS = int(mask_max_attempts)

    _os.environ["YOLO_CROP_ENABLED"] = "1" if crop_enabled else "0"
    _set_env_json("YOLO_CROP_SCALE", list(crop_scale))
    _os.environ["YOLO_CROP_P"] = str(crop_p)
    _os.environ["YOLO_CROP_MIN_BBOX_AREA_RATIO"] = str(crop_min_bbox_area_ratio)
    _os.environ["YOLO_CROP_ONEOF_ENABLED"] = "1" if crop_oneof_enabled else "0"
    _os.environ["YOLO_CROP_ONEOF_P"] = str(crop_oneof_p)
    _os.environ["YOLO_CROP_ONEOF_RANDOM_WEIGHT"] = str(crop_oneof_random_weight)
    _os.environ["YOLO_CROP_ONEOF_BOX_WEIGHT"] = str(crop_oneof_box_weight)

    _os.environ["YOLO_BOX_CROP_ENABLED"] = "1" if box_crop_enabled else "0"
    _os.environ["YOLO_BOX_CROP_P"] = str(box_crop_p)
    _set_env_json("YOLO_BOX_CROP_TARGET_CLASSES", list(box_crop_target_classes or ()))
    _set_env_json("YOLO_BOX_CROP_SCALE", list(box_crop_scale))
    _os.environ["YOLO_BOX_CROP_CENTER_JITTER"] = str(box_crop_center_jitter)
    _os.environ["YOLO_BOX_CROP_MIN_BBOX_AREA_RATIO"] = str(
        box_crop_min_bbox_area_ratio
    )
    _os.environ["YOLO_BOX_CROP_MIN_CROP_SIZE_RATIO"] = str(
        box_crop_min_crop_size_ratio
    )

    _os.environ["YOLO_MASK_ENABLED"] = "1" if mask_enabled else "0"
    _os.environ["YOLO_MASK_P"] = str(mask_p)
    _set_env_json("YOLO_MASK_HOLES", list(mask_holes))
    _set_env_json("YOLO_MASK_HEIGHT", list(mask_height))
    _set_env_json("YOLO_MASK_WIDTH", list(mask_width))
    _set_env_json("YOLO_MASK_FILL", mask_fill)
    _os.environ["YOLO_MASK_RANDOM_FILL_P"] = str(mask_random_fill_p)
    _os.environ["YOLO_MASK_MAX_BBOX_OVERLAP"] = str(mask_max_bbox_overlap)
    _os.environ["YOLO_MASK_MAX_ATTEMPTS"] = str(mask_max_attempts)

    return MaskDetectTrainer


make_detect_aug_trainer = make_mask_detect_trainer
