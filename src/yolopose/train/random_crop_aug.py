"""
Custom image/pose augmentations for YOLO Pose training.

Usage — inject into training via a custom PoseTrainer:

    from ultralytics.models.yolo.pose import PoseTrainer
    from random_crop_aug import RandomCropPose, build_transforms_with_crop

    class CropPoseTrainer(PoseTrainer):
        def build_dataset(self, img_path, mode="train", batch=None):
            dataset = super().build_dataset(img_path, mode, batch)
            if mode == "train":
                dataset.transforms = build_transforms_with_crop(
                    dataset, crop_scale=(0.6, 1.0), p=0.5
                )
            return dataset

    trainer = CropPoseTrainer(overrides={...})
    trainer.train()
"""

import json
import random

import numpy as np

from ultralytics.data.augment import (
    BaseTransform,
    Compose,
    v8_transforms,
    LetterBox,
    Format,
)


class RandomCropPose(BaseTransform):
    """
    Random crop augmentation for YOLO Pose datasets.

    After cropping, keypoints that fall outside the crop region have their
    visibility set to 0 (OOV). Bounding boxes are clipped to the crop bounds;
    instances whose clipped bbox area falls below `min_bbox_area_ratio` are
    removed together with their keypoints.

    Args:
        crop_scale (tuple[float, float]): Min/max fraction of image size to crop.
            Both width and height are sampled independently from this range.
            Default: (0.5, 1.0).
        p (float): Probability of applying the crop. Default: 0.5.
        min_bbox_area_ratio (float): Minimum ratio of clipped-bbox area to
            original-bbox area to keep an instance. Default: 0.1.
    """

    def __init__(self, crop_scale=(0.5, 1.0), p=0.5, min_bbox_area_ratio=0.1):
        super().__init__()
        assert 0 < crop_scale[0] <= crop_scale[1] <= 1.0, "crop_scale must be in (0, 1]"
        self.crop_scale = crop_scale
        self.p = p
        self.min_bbox_area_ratio = min_bbox_area_ratio

    def __call__(self, labels):
        """Apply random crop. Overrides BaseTransform.__call__ directly."""
        if random.random() > self.p:
            return labels

        img = labels["img"]
        h, w = img.shape[:2]

        # Sample crop size
        scale_h = random.uniform(*self.crop_scale)
        scale_w = random.uniform(*self.crop_scale)
        crop_h = int(h * scale_h)
        crop_w = int(w * scale_w)

        # Sample crop origin
        y1 = random.randint(0, max(0, h - crop_h))
        x1 = random.randint(0, max(0, w - crop_w))
        y2 = y1 + crop_h
        x2 = x1 + crop_w

        return _crop_labels(
            labels,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            min_bbox_area_ratio=self.min_bbox_area_ratio,
        )

    def apply_image(self, labels):
        return labels  # handled in __call__

    def apply_instances(self, labels):
        return labels  # handled in __call__


def _crop_labels(labels, x1, y1, x2, y2, min_bbox_area_ratio=0.1):
    """Crop image and synchronously clip/filter YOLO Instances."""
    img = labels["img"]
    h, w = img.shape[:2]
    crop_w = max(1, int(x2 - x1))
    crop_h = max(1, int(y2 - y1))

    labels["img"] = img[y1:y2, x1:x2]
    labels["resized_shape"] = (crop_h, crop_w)

    instances = labels.get("instances")
    if instances is None or len(instances) == 0:
        return labels

    # Force xyxy absolute-pixel format before any math.
    if instances.normalized:
        instances.denormalize(w, h)
    instances.convert_bbox(format="xyxy")
    bboxes = instances.bboxes.copy()

    orig_area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    orig_area = np.maximum(orig_area, 1e-6)

    bboxes_clipped = bboxes.copy()
    bboxes_clipped[:, [0, 2]] = bboxes[:, [0, 2]].clip(x1, x2) - x1
    bboxes_clipped[:, [1, 3]] = bboxes[:, [1, 3]].clip(y1, y2) - y1

    clipped_area = (bboxes_clipped[:, 2] - bboxes_clipped[:, 0]) * (
        bboxes_clipped[:, 3] - bboxes_clipped[:, 1]
    )
    keep = (clipped_area / orig_area) >= float(min_bbox_area_ratio)

    kpts_new = None
    if instances.keypoints is not None:
        kpts = instances.keypoints.copy()
        kpts_shifted = kpts.copy()
        kpts_shifted[:, :, 0] = kpts[:, :, 0] - x1
        kpts_shifted[:, :, 1] = kpts[:, :, 1] - y1

        originally_visible = kpts[:, :, 2] > 0
        out_of_crop = (
            (kpts_shifted[:, :, 0] < 0)
            | (kpts_shifted[:, :, 0] >= crop_w)
            | (kpts_shifted[:, :, 1] < 0)
            | (kpts_shifted[:, :, 1] >= crop_h)
        )
        kpts_shifted[:, :, 2] = np.where(
            originally_visible & out_of_crop,
            0,
            kpts[:, :, 2],
        )
        oov = kpts_shifted[:, :, 2] == 0
        kpts_shifted[oov, 0] = 0
        kpts_shifted[oov, 1] = 0
        kpts_new = kpts_shifted[keep]

    try:
        from ultralytics.utils.instance import Instances as _Instances
    except ImportError:
        _Instances = None

    orig_segments = instances.segments
    if orig_segments is not None and len(orig_segments) > 0:
        segs_kept = orig_segments[keep]
    else:
        segs_kept = orig_segments

    if _Instances is not None:
        new_instances = _Instances(
            bboxes=bboxes_clipped[keep],
            segments=segs_kept,
            keypoints=kpts_new,
            bbox_format="xyxy",
            normalized=False,
        )
    else:
        new_instances = instances
        new_instances.bboxes = bboxes_clipped[keep]
        new_instances.keypoints = kpts_new

    if "cls" in labels and labels["cls"] is not None:
        labels["cls"] = labels["cls"][keep]

    labels["instances"] = new_instances
    return labels


class RandomBoxCrop(BaseTransform):
    """
    Crop around an existing bbox to synthesize closer/larger objects.

    This is bbox-safe: all labels are clipped to the crop and small remnants are
    filtered by `min_bbox_area_ratio`.
    """

    def __init__(
        self,
        p=0.2,
        target_classes=None,
        crop_scale=(1.3, 3.0),
        center_jitter=0.15,
        min_bbox_area_ratio=0.05,
        min_crop_size_ratio=0.25,
    ):
        super().__init__()
        self.p = float(p)
        self.target_classes = None if not target_classes else {int(c) for c in target_classes}
        self.crop_scale = self._as_float_range(crop_scale)
        self.center_jitter = float(center_jitter)
        self.min_bbox_area_ratio = float(min_bbox_area_ratio)
        self.min_crop_size_ratio = float(min_crop_size_ratio)

    @staticmethod
    def _as_float_range(value):
        if isinstance(value, (int, float)):
            value = (value, value)
        lo, hi = float(value[0]), float(value[1])
        if lo > hi:
            lo, hi = hi, lo
        return max(0.0, lo), max(0.0, hi)

    def __call__(self, labels):
        if random.random() > self.p:
            return labels

        img = labels.get("img")
        instances = labels.get("instances")
        if img is None or instances is None or len(instances) == 0:
            return labels

        h, w = img.shape[:2]
        if instances.normalized:
            instances.denormalize(w, h)
        instances.convert_bbox(format="xyxy")
        bboxes = instances.bboxes.copy()
        if len(bboxes) == 0:
            return labels

        cls = labels.get("cls")
        eligible = np.arange(len(bboxes))
        if self.target_classes is not None and cls is not None:
            cls_flat = np.asarray(cls).reshape(-1).astype(int)
            eligible = np.asarray(
                [i for i, class_id in enumerate(cls_flat) if class_id in self.target_classes],
                dtype=np.int64,
            )
        if len(eligible) == 0:
            return labels

        wh = bboxes[:, 2:4] - bboxes[:, 0:2]
        valid = eligible[(wh[eligible, 0] > 1) & (wh[eligible, 1] > 1)]
        if len(valid) == 0:
            return labels

        idx = int(random.choice(valid.tolist()))
        bx1, by1, bx2, by2 = bboxes[idx]
        bw = max(1.0, float(bx2 - bx1))
        bh = max(1.0, float(by2 - by1))
        cx = float((bx1 + bx2) * 0.5)
        cy = float((by1 + by2) * 0.5)

        scale = random.uniform(*self.crop_scale)
        crop_w = max(bw * scale, w * self.min_crop_size_ratio)
        crop_h = max(bh * scale, h * self.min_crop_size_ratio)
        crop_w = min(float(w), crop_w)
        crop_h = min(float(h), crop_h)

        cx += random.uniform(-self.center_jitter, self.center_jitter) * bw
        cy += random.uniform(-self.center_jitter, self.center_jitter) * bh

        x1 = int(round(cx - crop_w * 0.5))
        y1 = int(round(cy - crop_h * 0.5))
        x1 = max(0, min(x1, int(w - crop_w)))
        y1 = max(0, min(y1, int(h - crop_h)))
        x2 = min(w, x1 + int(round(crop_w)))
        y2 = min(h, y1 + int(round(crop_h)))

        if x2 <= x1 or y2 <= y1:
            return labels

        return _crop_labels(
            labels,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            min_bbox_area_ratio=self.min_bbox_area_ratio,
        )

    def apply_image(self, labels):
        return labels

    def apply_instances(self, labels):
        return labels


class RandomOneOfTransforms(BaseTransform):
    """Apply at most one transform from a weighted candidate list."""

    def __init__(self, transforms, weights=None, p=0.5):
        super().__init__()
        self.transforms = list(transforms)
        self.weights = list(weights) if weights is not None else [1.0] * len(self.transforms)
        self.p = float(p)

    def __call__(self, labels):
        if not self.transforms or random.random() > self.p:
            return labels
        transform = random.choices(self.transforms, weights=self.weights, k=1)[0]
        return transform(labels)

    def apply_image(self, labels):
        return labels

    def apply_instances(self, labels):
        return labels


class RandomMaskImageOnly(BaseTransform):
    """
    Random rectangular mask augmentation that only edits image pixels.

    This is intentionally image-only for YOLO Pose. Albumentations dropout
    transforms are treated as spatial transforms by Ultralytics and can desync
    bboxes/keypoints, so this transform never touches labels["instances"].
    """

    def __init__(
        self,
        p=0.2,
        holes=(1, 2),
        height=(0.1, 0.3),
        width=(0.1, 0.3),
        fill=0,
        random_fill_p=0.0,
        max_bbox_overlap=1.0,
        max_attempts=20,
    ):
        super().__init__()
        self.p = float(p)
        self.holes = self._as_int_range(holes)
        self.height = self._as_float_range(height)
        self.width = self._as_float_range(width)
        self.fill = fill
        self.random_fill_p = float(random_fill_p)
        self.max_bbox_overlap = min(1.0, max(0.0, float(max_bbox_overlap)))
        self.max_attempts = max(1, int(max_attempts))

    @staticmethod
    def _as_int_range(value):
        if isinstance(value, (int, float)):
            value = (value, value)
        lo, hi = int(value[0]), int(value[1])
        if lo > hi:
            lo, hi = hi, lo
        return max(0, lo), max(0, hi)

    @staticmethod
    def _as_float_range(value):
        if isinstance(value, (int, float)):
            value = (value, value)
        lo, hi = float(value[0]), float(value[1])
        if lo > hi:
            lo, hi = hi, lo
        return max(0.0, lo), max(0.0, hi)

    @staticmethod
    def _sample_dim(value_range, full_size):
        value = random.uniform(*value_range)
        pixels = value * full_size if 0.0 <= value <= 1.0 else value
        return max(1, min(full_size, int(round(pixels))))

    @staticmethod
    def _random_fill(shape, dtype):
        if np.issubdtype(dtype, np.integer):
            return np.random.randint(0, 256, size=shape, dtype=dtype)
        return np.random.random(size=shape).astype(dtype)

    def _fill_value(self, shape, dtype):
        if random.random() < self.random_fill_p:
            return self._random_fill(shape, dtype)
        if isinstance(self.fill, str) and self.fill.lower() == "random":
            return self._random_fill(shape, dtype)
        return np.asarray(self.fill, dtype=dtype)

    @staticmethod
    def _bboxes_xyxy(labels, w, h):
        instances = labels.get("instances")
        if instances is None or len(instances) == 0:
            return None

        bboxes = np.asarray(instances.bboxes, dtype=np.float32).copy()
        if bboxes.size == 0:
            return None

        if getattr(instances, "normalized", False):
            bboxes[:, [0, 2]] *= w
            bboxes[:, [1, 3]] *= h

        bbox_format = getattr(getattr(instances, "_bboxes", None), "format", "xyxy")
        if bbox_format == "xywh":
            cx, cy, bw, bh = bboxes.T
            bboxes = np.stack(
                [cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5],
                axis=1,
            )
        elif bbox_format == "ltwh":
            x, y, bw, bh = bboxes.T
            bboxes = np.stack([x, y, x + bw, y + bh], axis=1)
        elif bbox_format != "xyxy":
            return None

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clip(0, w)
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clip(0, h)
        areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        return bboxes[areas > 1e-6]

    def _overlap_allowed(self, mask_xyxy, bboxes_xyxy):
        if self.max_bbox_overlap >= 1.0 or bboxes_xyxy is None or len(bboxes_xyxy) == 0:
            return True

        x1, y1, x2, y2 = mask_xyxy
        inter_w = np.maximum(0.0, np.minimum(x2, bboxes_xyxy[:, 2]) - np.maximum(x1, bboxes_xyxy[:, 0]))
        inter_h = np.maximum(0.0, np.minimum(y2, bboxes_xyxy[:, 3]) - np.maximum(y1, bboxes_xyxy[:, 1]))
        inter_area = inter_w * inter_h
        bbox_area = np.maximum(
            (bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0])
            * (bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]),
            1e-6,
        )
        return float(np.max(inter_area / bbox_area)) <= self.max_bbox_overlap

    def __call__(self, labels):
        """Apply random image masks without modifying pose annotations."""
        if random.random() > self.p:
            return labels

        img = labels.get("img")
        if img is None or img.size == 0:
            return labels

        h, w = img.shape[:2]
        holes_min, holes_max = self.holes
        if holes_max <= 0:
            return labels

        bboxes_xyxy = self._bboxes_xyxy(labels, w, h)
        num_holes = random.randint(holes_min, holes_max)
        for _ in range(num_holes):
            for _attempt in range(self.max_attempts):
                mask_h = self._sample_dim(self.height, h)
                mask_w = self._sample_dim(self.width, w)
                y1 = random.randint(0, max(0, h - mask_h))
                x1 = random.randint(0, max(0, w - mask_w))
                y2 = y1 + mask_h
                x2 = x1 + mask_w
                if self._overlap_allowed((x1, y1, x2, y2), bboxes_xyxy):
                    break
            else:
                continue

            fill = self._fill_value(img[y1:y2, x1:x2].shape, img.dtype)
            img[y1:y2, x1:x2] = fill

        labels["img"] = img
        return labels

    def apply_image(self, labels):
        return labels  # handled in __call__

    def apply_instances(self, labels):
        return labels  # image-only transform


# ─────────────────────────────────────────────────────────────────────────────
# Helper: rebuild transform pipeline with custom transforms injected
# ─────────────────────────────────────────────────────────────────────────────


def _patch_copypaste_none_segments(transforms):
    """
    Workaround for ultralytics bug: CopyPaste.__call__ calls len(segments) before
    checking self.p == 0, causing TypeError when segments=None (pose datasets).

    Recursively patches CopyPaste instances in the transform pipeline and inside
    any nested Mosaic pre_transforms.
    """
    try:
        from ultralytics.data.augment import CopyPaste, BaseMixTransform
    except ImportError:
        return

    if not hasattr(transforms, "transforms"):
        return

    for tf in transforms.transforms:
        if isinstance(tf, CopyPaste):
            _orig = tf.__call__

            def _safe_call(labels, _orig=_orig):
                inst = labels.get("instances")
                if inst is not None and inst.segments is None:
                    # Provide empty segments array so len() returns 0
                    inst._segments = np.zeros((0, 0, 2), dtype=np.float32)
                return _orig(labels)

            tf.__call__ = _safe_call

        # Recurse into Mosaic / any BaseMixTransform pre_transform
        pre = getattr(tf, "pre_transform", None)
        if pre is not None:
            _patch_copypaste_none_segments(pre)


def _insert_before_letterbox(dataset, transform):
    from ultralytics.data.augment import LetterBox  # lazy import

    transforms = dataset.transforms
    if transforms is None:
        return transforms

    insert_idx = None
    for i, tf in enumerate(transforms.transforms):
        if isinstance(tf, LetterBox):
            insert_idx = i
            break

    if insert_idx is not None:
        transforms.insert(insert_idx, transform)
    else:
        transforms.insert(0, transform)

    _patch_copypaste_none_segments(transforms)
    return transforms


def build_transforms_with_crop(
    dataset,
    crop_scale=(0.5, 1.0),
    p=0.5,
    imgsz=None,
    min_bbox_area_ratio=0.1,
):
    """
    Inject RandomCropPose into the dataset's existing transform pipeline (in-place).

    The dataset's transforms are already built by the parent YOLODataset.
    We find the first LetterBox in the pipeline and insert RandomCropPose before it.
    If no LetterBox is found, insert at position 0.

    Args:
        dataset: YOLODataset instance with .transforms already set.
        crop_scale (tuple): (min_scale, max_scale) for RandomCropPose.
        p (float): Probability of applying the crop.
        imgsz: Unused, kept for API compatibility.
        min_bbox_area_ratio: Minimum visible bbox area ratio to keep an instance.

    Returns:
        Compose: The modified transform pipeline.
    """
    crop_tf = RandomCropPose(
        crop_scale=crop_scale,
        p=p,
        min_bbox_area_ratio=min_bbox_area_ratio,
    )
    return _insert_before_letterbox(dataset, crop_tf)


def build_transforms_with_box_crop(
    dataset,
    p=0.2,
    target_classes=None,
    crop_scale=(1.3, 3.0),
    center_jitter=0.15,
    min_bbox_area_ratio=0.05,
    min_crop_size_ratio=0.25,
):
    """Inject RandomBoxCrop into the dataset's existing transform pipeline."""
    crop_tf = RandomBoxCrop(
        p=p,
        target_classes=target_classes,
        crop_scale=crop_scale,
        center_jitter=center_jitter,
        min_bbox_area_ratio=min_bbox_area_ratio,
        min_crop_size_ratio=min_crop_size_ratio,
    )
    return _insert_before_letterbox(dataset, crop_tf)


def build_transforms_with_crop_oneof(
    dataset,
    p=0.4,
    random_crop_enabled=True,
    random_crop_weight=1.0,
    random_crop_scale=(0.65, 1.0),
    random_crop_min_bbox_area_ratio=0.03,
    box_crop_enabled=True,
    box_crop_weight=1.0,
    box_crop_target_classes=None,
    box_crop_scale=(1.3, 3.0),
    box_crop_center_jitter=0.15,
    box_crop_min_bbox_area_ratio=0.05,
    box_crop_min_crop_size_ratio=0.25,
):
    """Inject a crop OneOf transform: at most one crop transform is applied."""
    transforms = []
    weights = []
    if random_crop_enabled:
        transforms.append(
            RandomCropPose(
                crop_scale=random_crop_scale,
                p=1.0,
                min_bbox_area_ratio=random_crop_min_bbox_area_ratio,
            )
        )
        weights.append(float(random_crop_weight))
    if box_crop_enabled:
        transforms.append(
            RandomBoxCrop(
                p=1.0,
                target_classes=box_crop_target_classes,
                crop_scale=box_crop_scale,
                center_jitter=box_crop_center_jitter,
                min_bbox_area_ratio=box_crop_min_bbox_area_ratio,
                min_crop_size_ratio=box_crop_min_crop_size_ratio,
            )
        )
        weights.append(float(box_crop_weight))
    if not transforms or sum(weights) <= 0:
        return dataset.transforms

    return _insert_before_letterbox(
        dataset,
        RandomOneOfTransforms(transforms=transforms, weights=weights, p=p),
    )


def build_transforms_with_mask(
    dataset,
    p=0.2,
    holes=(1, 2),
    height=(0.1, 0.3),
    width=(0.1, 0.3),
    fill=0,
    random_fill_p=0.0,
    max_bbox_overlap=1.0,
    max_attempts=20,
):
    """
    Inject RandomMaskImageOnly into the dataset's transform pipeline (in-place).

    The mask is inserted before Format so it edits the final numpy image while
    leaving pose annotations untouched.
    """
    transforms = dataset.transforms
    if transforms is None:
        return transforms

    mask_tf = RandomMaskImageOnly(
        p=p,
        holes=holes,
        height=height,
        width=width,
        fill=fill,
        random_fill_p=random_fill_p,
        max_bbox_overlap=max_bbox_overlap,
        max_attempts=max_attempts,
    )

    insert_idx = None
    for i, tf in enumerate(transforms.transforms):
        if isinstance(tf, Format):
            insert_idx = i
            break

    if insert_idx is not None:
        transforms.insert(insert_idx, mask_tf)
    else:
        transforms.insert(len(transforms.transforms), mask_tf)

    _patch_copypaste_none_segments(transforms)
    return transforms


# ─────────────────────────────────────────────────────────────────────────────
# Custom PoseTrainer that injects custom transforms at training time.
# Must be defined at MODULE LEVEL so ultralytics DDP can import it as:
#   from random_crop_aug import CropPoseTrainer
# ─────────────────────────────────────────────────────────────────────────────

try:
    from ultralytics.models.yolo.pose import PoseTrainer as _PoseTrainer

    class CropPoseTrainer(_PoseTrainer):
        """PoseTrainer with optional RandomCropPose and RandomMaskImageOnly."""

        # Class-level parameters — configured by make_crop_pose_trainer() before use
        CROP_ENABLED: bool = True
        CROP_SCALE: tuple = (0.5, 1.0)
        CROP_P: float = 0.5
        CROP_MIN_BBOX_AREA_RATIO: float = 0.1
        MASK_ENABLED: bool = False
        MASK_P: float = 0.2
        MASK_HOLES: tuple = (1, 2)
        MASK_HEIGHT: tuple = (0.1, 0.3)
        MASK_WIDTH: tuple = (0.1, 0.3)
        MASK_FILL = 0
        MASK_RANDOM_FILL_P: float = 0.0
        MASK_MAX_BBOX_OVERLAP: float = 1.0
        MASK_MAX_ATTEMPTS: int = 20

        def build_dataset(self, img_path, mode="train", batch=None):
            dataset = super().build_dataset(img_path, mode, batch)
            if mode == "train":
                # Read params from env vars so DDP workers get the right values
                import os as _os

                crop_enabled = _env_bool(
                    "YOLO_CROP_ENABLED", self.__class__.CROP_ENABLED
                )
                mask_enabled = _env_bool(
                    "YOLO_MASK_ENABLED", self.__class__.MASK_ENABLED
                )

                if crop_enabled:
                    crop_scale = (
                        float(
                            _os.environ.get(
                                "YOLO_CROP_SCALE_MIN",
                                str(self.__class__.CROP_SCALE[0]),
                            )
                        ),
                        float(
                            _os.environ.get(
                                "YOLO_CROP_SCALE_MAX",
                                str(self.__class__.CROP_SCALE[1]),
                            )
                        ),
                    )
                    crop_p = float(
                        _os.environ.get("YOLO_CROP_P", str(self.__class__.CROP_P))
                    )
                    crop_min_bbox_area_ratio = float(
                        _os.environ.get(
                            "YOLO_CROP_MIN_BBOX_AREA_RATIO",
                            str(self.__class__.CROP_MIN_BBOX_AREA_RATIO),
                        )
                    )
                    dataset.transforms = build_transforms_with_crop(
                        dataset,
                        crop_scale=crop_scale,
                        p=crop_p,
                        min_bbox_area_ratio=crop_min_bbox_area_ratio,
                    )
                    print(
                        f"[CropPoseTrainer] Injected RandomCropPose "
                        f"(crop_scale={crop_scale}, p={crop_p}, "
                        f"min_bbox_area_ratio={crop_min_bbox_area_ratio}) "
                        f"into train transforms"
                    )

                if mask_enabled:
                    mask_p = float(
                        _os.environ.get("YOLO_MASK_P", str(self.__class__.MASK_P))
                    )
                    holes = tuple(
                        _env_json("YOLO_MASK_HOLES", self.__class__.MASK_HOLES)
                    )
                    height = tuple(
                        _env_json("YOLO_MASK_HEIGHT", self.__class__.MASK_HEIGHT)
                    )
                    width = tuple(
                        _env_json("YOLO_MASK_WIDTH", self.__class__.MASK_WIDTH)
                    )
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
                        f"[CropPoseTrainer] Injected RandomMaskImageOnly "
                        f"(p={mask_p}, holes={holes}, height={height}, "
                        f"width={width}, fill={fill}, "
                        f"random_fill_p={random_fill_p}, "
                        f"max_bbox_overlap={max_bbox_overlap}, "
                        f"max_attempts={max_attempts}) "
                        f"into train transforms"
                    )
            return dataset

except ImportError:
    CropPoseTrainer = None  # type: ignore[assignment,misc]


def _env_bool(name, default=False):
    import os as _os

    value = _os.environ.get(name)
    if value is None:
        return bool(default)
    return value.lower() in {"1", "true", "yes", "on"}


def _env_json(name, default):
    import os as _os

    value = _os.environ.get(name)
    if value is None:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _set_env_json(name, value):
    import os as _os

    _os.environ[name] = json.dumps(value)


def make_pose_aug_trainer(
    crop_enabled=False,
    crop_scale=(0.5, 1.0),
    crop_p=0.5,
    crop_min_bbox_area_ratio=0.1,
    mask_enabled=False,
    mask_p=0.2,
    mask_holes=(1, 2),
    mask_height=(0.1, 0.3),
    mask_width=(0.1, 0.3),
    mask_fill=0,
    mask_random_fill_p=0.0,
    mask_max_bbox_overlap=1.0,
    mask_max_attempts=20,
):
    """Configure and return the module-level custom PoseTrainer class."""
    import os as _os

    if CropPoseTrainer is None:
        raise ImportError("ultralytics is required for CropPoseTrainer")

    CropPoseTrainer.CROP_ENABLED = bool(crop_enabled)
    CropPoseTrainer.CROP_SCALE = tuple(crop_scale)
    CropPoseTrainer.CROP_P = float(crop_p)
    CropPoseTrainer.CROP_MIN_BBOX_AREA_RATIO = float(crop_min_bbox_area_ratio)
    CropPoseTrainer.MASK_ENABLED = bool(mask_enabled)
    CropPoseTrainer.MASK_P = float(mask_p)
    CropPoseTrainer.MASK_HOLES = tuple(mask_holes)
    CropPoseTrainer.MASK_HEIGHT = tuple(mask_height)
    CropPoseTrainer.MASK_WIDTH = tuple(mask_width)
    CropPoseTrainer.MASK_FILL = mask_fill
    CropPoseTrainer.MASK_RANDOM_FILL_P = float(mask_random_fill_p)
    CropPoseTrainer.MASK_MAX_BBOX_OVERLAP = float(mask_max_bbox_overlap)
    CropPoseTrainer.MASK_MAX_ATTEMPTS = int(mask_max_attempts)

    # Persist params in env so DDP workers inherit them.
    _os.environ["YOLO_CROP_ENABLED"] = "1" if crop_enabled else "0"
    _os.environ["YOLO_CROP_SCALE_MIN"] = str(crop_scale[0])
    _os.environ["YOLO_CROP_SCALE_MAX"] = str(crop_scale[1])
    _os.environ["YOLO_CROP_P"] = str(crop_p)
    _os.environ["YOLO_CROP_MIN_BBOX_AREA_RATIO"] = str(crop_min_bbox_area_ratio)

    _os.environ["YOLO_MASK_ENABLED"] = "1" if mask_enabled else "0"
    _os.environ["YOLO_MASK_P"] = str(mask_p)
    _set_env_json("YOLO_MASK_HOLES", list(mask_holes))
    _set_env_json("YOLO_MASK_HEIGHT", list(mask_height))
    _set_env_json("YOLO_MASK_WIDTH", list(mask_width))
    _set_env_json("YOLO_MASK_FILL", mask_fill)
    _os.environ["YOLO_MASK_RANDOM_FILL_P"] = str(mask_random_fill_p)
    _os.environ["YOLO_MASK_MAX_BBOX_OVERLAP"] = str(mask_max_bbox_overlap)
    _os.environ["YOLO_MASK_MAX_ATTEMPTS"] = str(mask_max_attempts)

    return CropPoseTrainer


def make_crop_pose_trainer(crop_scale=(0.5, 1.0), crop_p=0.5):
    """
    Configure and return the module-level CropPoseTrainer class.

    Writes YOLO_CROP_SCALE_MIN/MAX and YOLO_CROP_P to os.environ so that
    DDP worker processes (which re-import this class from the module) pick up
    the same parameters.

    Example:
        CropPoseTrainer = make_crop_pose_trainer(crop_scale=(0.6, 1.0), crop_p=0.5)
        trainer = CropPoseTrainer(overrides={...})
        trainer.train()
    """
    return make_pose_aug_trainer(
        crop_enabled=True,
        crop_scale=crop_scale,
        crop_p=crop_p,
        crop_min_bbox_area_ratio=0.1,
        mask_enabled=False,
    )
