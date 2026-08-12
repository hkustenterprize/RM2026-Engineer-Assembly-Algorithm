"""Configuration-driven image and label-aware augmentations."""

import os
import random

import numpy as np
from ultralytics.data.augment import BaseTransform, Format, LetterBox

from nn.utils import make_object_from_config


def _ordered_range(value, cast=float, lower=0):
    if isinstance(value, (int, float)):
        value = (value, value)
    low, high = cast(value[0]), cast(value[1])
    low, high = min(low, high), max(low, high)
    return max(lower, low), max(lower, high)


def _crop_labels(labels, x1, y1, x2, y2, min_bbox_area_ratio):
    """Crop an image and synchronously update its Ultralytics instances."""
    image = labels["img"]
    image_height, image_width = image.shape[:2]
    crop_width = max(1, int(x2 - x1))
    crop_height = max(1, int(y2 - y1))
    labels["img"] = image[y1:y2, x1:x2]
    labels["resized_shape"] = (crop_height, crop_width)

    instances = labels.get("instances")
    if instances is None or len(instances) == 0:
        return labels

    if instances.normalized:
        instances.denormalize(image_width, image_height)
    instances.convert_bbox(format="xyxy")
    boxes = instances.bboxes.copy()
    original_area = np.maximum(
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
        1e-6,
    )

    clipped_boxes = boxes.copy()
    clipped_boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(x1, x2) - x1
    clipped_boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(y1, y2) - y1
    clipped_area = (clipped_boxes[:, 2] - clipped_boxes[:, 0]) * (
        clipped_boxes[:, 3] - clipped_boxes[:, 1]
    )
    keep = clipped_area / original_area >= float(min_bbox_area_ratio)

    keypoints = None
    if instances.keypoints is not None:
        keypoints = instances.keypoints.copy()
        visible = keypoints[:, :, 2] > 0
        keypoints[:, :, 0] -= x1
        keypoints[:, :, 1] -= y1
        outside = (
            (keypoints[:, :, 0] < 0)
            | (keypoints[:, :, 0] >= crop_width)
            | (keypoints[:, :, 1] < 0)
            | (keypoints[:, :, 1] >= crop_height)
        )
        keypoints[:, :, 2][visible & outside] = 0
        keypoints[keypoints[:, :, 2] == 0, :2] = 0
        keypoints = keypoints[keep]

    segments = instances.segments
    if segments is not None and len(segments) > 0:
        segments = segments[keep]

    from ultralytics.utils.instance import Instances

    labels["instances"] = Instances(
        bboxes=clipped_boxes[keep],
        segments=segments,
        keypoints=keypoints,
        bbox_format="xyxy",
        normalized=False,
    )
    if labels.get("cls") is not None:
        labels["cls"] = labels["cls"][keep]
    return labels


class RandomCrop(BaseTransform):
    """Randomly crop an image while keeping boxes and keypoints synchronized."""

    def __init__(self, scale=(0.5, 1.0), p=0.5, min_bbox_area_ratio=0.1):
        super().__init__()
        self.scale = _ordered_range(scale)
        if not 0 < self.scale[0] <= self.scale[1] <= 1:
            raise ValueError("scale must lie in (0, 1]")
        self.p = float(p)
        self.min_bbox_area_ratio = float(min_bbox_area_ratio)

    def __call__(self, labels):
        if random.random() > self.p:
            return labels
        height, width = labels["img"].shape[:2]
        crop_height = max(1, int(height * random.uniform(*self.scale)))
        crop_width = max(1, int(width * random.uniform(*self.scale)))
        y1 = random.randint(0, max(0, height - crop_height))
        x1 = random.randint(0, max(0, width - crop_width))
        return _crop_labels(
            labels,
            x1,
            y1,
            x1 + crop_width,
            y1 + crop_height,
            self.min_bbox_area_ratio,
        )

    def apply_image(self, labels):
        return labels

    def apply_instances(self, labels):
        return labels


class RandomBoxCrop(BaseTransform):
    """Crop around a selected box to synthesize a closer target view."""

    def __init__(
        self,
        p=0.2,
        target_classes=None,
        scale=(1.3, 3.0),
        center_jitter=0.15,
        min_bbox_area_ratio=0.05,
        min_crop_size_ratio=0.25,
    ):
        super().__init__()
        self.p = float(p)
        self.target_classes = (
            None if not target_classes else {int(value) for value in target_classes}
        )
        self.scale = _ordered_range(scale)
        self.center_jitter = float(center_jitter)
        self.min_bbox_area_ratio = float(min_bbox_area_ratio)
        self.min_crop_size_ratio = float(min_crop_size_ratio)

    def __call__(self, labels):
        if random.random() > self.p:
            return labels
        image = labels.get("img")
        instances = labels.get("instances")
        if image is None or instances is None or len(instances) == 0:
            return labels

        height, width = image.shape[:2]
        if instances.normalized:
            instances.denormalize(width, height)
        instances.convert_bbox(format="xyxy")
        boxes = instances.bboxes.copy()
        eligible = np.arange(len(boxes))
        classes = labels.get("cls")
        if self.target_classes is not None and classes is not None:
            class_ids = np.asarray(classes).reshape(-1).astype(int)
            eligible = np.flatnonzero(
                np.isin(class_ids, tuple(self.target_classes))
            )
        sizes = boxes[:, 2:4] - boxes[:, 0:2]
        eligible = eligible[(sizes[eligible, 0] > 1) & (sizes[eligible, 1] > 1)]
        if len(eligible) == 0:
            return labels

        box = boxes[int(random.choice(eligible.tolist()))]
        box_width, box_height = box[2] - box[0], box[3] - box[1]
        center_x, center_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        factor = random.uniform(*self.scale)
        crop_width = min(width, max(box_width * factor, width * self.min_crop_size_ratio))
        crop_height = min(
            height, max(box_height * factor, height * self.min_crop_size_ratio)
        )
        center_x += random.uniform(-self.center_jitter, self.center_jitter) * box_width
        center_y += random.uniform(-self.center_jitter, self.center_jitter) * box_height
        x1 = max(0, min(int(round(center_x - crop_width / 2)), int(width - crop_width)))
        y1 = max(
            0, min(int(round(center_y - crop_height / 2)), int(height - crop_height))
        )
        x2 = min(width, x1 + int(round(crop_width)))
        y2 = min(height, y1 + int(round(crop_height)))
        if x2 <= x1 or y2 <= y1:
            return labels
        return _crop_labels(
            labels, x1, y1, x2, y2, self.min_bbox_area_ratio
        )

    def apply_image(self, labels):
        return labels

    def apply_instances(self, labels):
        return labels


class RandomChoice(BaseTransform):
    """Apply at most one transform from a weighted list."""

    def __init__(self, transforms, weights=None, p=0.5):
        super().__init__()
        self.transforms = list(transforms)
        self.weights = list(weights or [1.0] * len(self.transforms))
        if len(self.weights) != len(self.transforms):
            raise ValueError("weights must match the number of transforms")
        if any(weight < 0 for weight in self.weights) or not sum(self.weights) > 0:
            raise ValueError("weights must be non-negative with a positive sum")
        self.p = float(p)

    def __call__(self, labels):
        if not self.transforms or random.random() > self.p:
            return labels
        return random.choices(self.transforms, weights=self.weights, k=1)[0](labels)

    def apply_image(self, labels):
        return labels

    def apply_instances(self, labels):
        return labels


class RandomOcclusion(BaseTransform):
    """Apply rectangular image occlusions without changing annotations."""

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
        self.holes = _ordered_range(holes, int)
        self.height = _ordered_range(height)
        self.width = _ordered_range(width)
        self.fill = fill
        self.random_fill_p = float(random_fill_p)
        self.max_bbox_overlap = min(1.0, max(0.0, float(max_bbox_overlap)))
        self.max_attempts = max(1, int(max_attempts))

    @staticmethod
    def _sample_size(value_range, full_size):
        value = random.uniform(*value_range)
        pixels = value * full_size if 0 <= value <= 1 else value
        return max(1, min(full_size, int(round(pixels))))

    @staticmethod
    def _random_fill(shape, dtype):
        if np.issubdtype(dtype, np.integer):
            return np.random.randint(0, 256, size=shape, dtype=dtype)
        return np.random.random(size=shape).astype(dtype)

    def _fill_value(self, shape, dtype):
        if self.fill == "random" or random.random() < self.random_fill_p:
            return self._random_fill(shape, dtype)
        return np.asarray(self.fill, dtype=dtype)

    @staticmethod
    def _boxes_xyxy(labels, width, height):
        instances = labels.get("instances")
        if instances is None or len(instances) == 0:
            return None
        boxes = np.asarray(instances.bboxes, dtype=np.float32).copy()
        if boxes.size == 0:
            return None
        if instances.normalized:
            boxes[:, [0, 2]] *= width
            boxes[:, [1, 3]] *= height
        box_format = getattr(getattr(instances, "_bboxes", None), "format", "xyxy")
        if box_format == "xywh":
            center_x, center_y, box_width, box_height = boxes.T
            boxes = np.stack(
                [
                    center_x - box_width / 2,
                    center_y - box_height / 2,
                    center_x + box_width / 2,
                    center_y + box_height / 2,
                ],
                axis=1,
            )
        elif box_format == "ltwh":
            x, y, box_width, box_height = boxes.T
            boxes = np.stack([x, y, x + box_width, y + box_height], axis=1)
        elif box_format != "xyxy":
            return None
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return boxes[area > 1e-6]

    def _is_allowed(self, rectangle, boxes):
        if self.max_bbox_overlap >= 1 or boxes is None or len(boxes) == 0:
            return True
        x1, y1, x2, y2 = rectangle
        intersection = np.maximum(
            0, np.minimum(x2, boxes[:, 2]) - np.maximum(x1, boxes[:, 0])
        ) * np.maximum(
            0, np.minimum(y2, boxes[:, 3]) - np.maximum(y1, boxes[:, 1])
        )
        box_area = np.maximum(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-6
        )
        return float(np.max(intersection / box_area)) <= self.max_bbox_overlap

    def __call__(self, labels):
        if random.random() > self.p:
            return labels
        image = labels.get("img")
        if image is None or image.size == 0 or self.holes[1] <= 0:
            return labels
        height, width = image.shape[:2]
        boxes = self._boxes_xyxy(labels, width, height)
        for _ in range(random.randint(*self.holes)):
            for _ in range(self.max_attempts):
                mask_height = self._sample_size(self.height, height)
                mask_width = self._sample_size(self.width, width)
                y1 = random.randint(0, max(0, height - mask_height))
                x1 = random.randint(0, max(0, width - mask_width))
                rectangle = (x1, y1, x1 + mask_width, y1 + mask_height)
                if self._is_allowed(rectangle, boxes):
                    break
            else:
                continue
            x1, y1, x2, y2 = rectangle
            image[y1:y2, x1:x2] = self._fill_value(
                image[y1:y2, x1:x2].shape, image.dtype
            )
        labels["img"] = image
        return labels

    def apply_image(self, labels):
        return labels

    def apply_instances(self, labels):
        return labels


def build_image_augmentations(config):
    """Build image-only Albumentations transforms for ``YOLO.train``."""
    if not config.get("enabled", True):
        return None

    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    import albumentations as A

    definitions = config.get("transforms") or {}
    if not isinstance(definitions, dict):
        raise ValueError("albumentations.transforms must be a mapping")
    transforms = {
        name: make_object_from_config(definition)
        for name, definition in definitions.items()
    }

    groups = config.get("oneof_groups") or []
    if groups:
        result = []
        for group in groups:
            names = group.get("transforms") or []
            missing = [name for name in names if name not in transforms]
            if missing:
                group_name = group.get("name", "<unnamed>")
                raise ValueError(f"Unknown transforms in group {group_name}: {missing}")
            result.append(A.OneOf([transforms[name] for name in names], p=group["p"]))
    else:
        result = list(transforms.values())

    if not result:
        return None
    names = ", ".join(type(transform).__name__ for transform in result)
    print(f"[train] image augmentations: {names}")
    return result


def _make_transform(config, **overrides):
    definition = {
        key: value
        for key, value in (config or {}).items()
        if key != "enabled"
    }
    definition.update(overrides)
    if "_class_name" not in definition:
        raise ValueError("Custom augmentation config requires _class_name")
    return make_object_from_config(definition)


def _insert_before(dataset, transform, transform_type, fallback_index):
    pipeline = dataset.transforms
    if pipeline is None:
        return
    index = next(
        (
            index
            for index, existing in enumerate(pipeline.transforms)
            if isinstance(existing, transform_type)
        ),
        fallback_index(pipeline.transforms),
    )
    pipeline.insert(index, transform)


def _insert_spatial(dataset, transform):
    _insert_before(dataset, transform, LetterBox, lambda _: 0)


def _insert_occlusion(dataset, transform):
    _insert_before(dataset, transform, Format, len)


def apply_pose_augmentations(dataset, config):
    """Inject the configured pose-safe spatial and occlusion transforms."""
    crop = config.get("random_crop") or {}
    occlusion = config.get("random_occlusion") or {}
    if crop.get("enabled"):
        _insert_spatial(dataset, _make_transform(crop))
    if occlusion.get("enabled"):
        _insert_occlusion(dataset, _make_transform(occlusion))
    return dataset


def apply_detection_augmentations(dataset, config):
    """Inject the configured detection-safe crop and occlusion transforms."""
    random_crop = config.get("random_crop") or {}
    box_crop = config.get("box_crop") or {}
    choice = config.get("crop_choice") or {}
    enabled_crops = {
        name: section
        for name, section in {
            "random_crop": random_crop,
            "box_crop": box_crop,
        }.items()
        if section.get("enabled")
    }

    if choice.get("enabled") and enabled_crops:
        weights = choice.get("weights") or {}
        _insert_spatial(
            dataset,
            RandomChoice(
                [_make_transform(section, p=1.0) for section in enabled_crops.values()],
                weights=[float(weights.get(name, 1.0)) for name in enabled_crops],
                p=float(choice.get("p", 0.5)),
            ),
        )
    else:
        for section in enabled_crops.values():
            _insert_spatial(dataset, _make_transform(section))

    occlusion = config.get("random_occlusion") or {}
    if occlusion.get("enabled"):
        _insert_occlusion(dataset, _make_transform(occlusion))
    return dataset
