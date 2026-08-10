"""Ultralytics trainer adapters and DDP configuration handoff."""

import json
import os

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.pose import PoseTrainer

from nn.yolo.train.augmentations import (
    apply_detection_augmentations,
    apply_pose_augmentations,
)


def _store_worker_config(variable, config):
    os.environ[variable] = json.dumps(config)


def _load_worker_config(variable):
    value = os.environ.get(variable)
    return json.loads(value) if value else {}


class _AugmentedDatasetMixin:
    """Apply task-specific transforms after Ultralytics builds a dataset."""

    augmentation_config_env = ""
    apply_augmentations = None

    def build_dataset(self, img_path, mode="train", batch=None):
        dataset = super().build_dataset(img_path, mode, batch)
        if mode == "train":
            config = _load_worker_config(self.augmentation_config_env)
            self.apply_augmentations(dataset, config)
        return dataset


class AugmentedPoseTrainer(_AugmentedDatasetMixin, PoseTrainer):
    augmentation_config_env = "RM26_POSE_AUGMENTATIONS"
    apply_augmentations = staticmethod(apply_pose_augmentations)


class AugmentedDetectionTrainer(_AugmentedDatasetMixin, DetectionTrainer):
    augmentation_config_env = "RM26_DETECT_AUGMENTATIONS"
    apply_augmentations = staticmethod(apply_detection_augmentations)


def create_pose_trainer(config):
    """Configure DDP workers and return the pose trainer class."""
    _store_worker_config(AugmentedPoseTrainer.augmentation_config_env, config)
    return AugmentedPoseTrainer


def create_detection_trainer(config):
    """Configure DDP workers and return the detection trainer class."""
    _store_worker_config(AugmentedDetectionTrainer.augmentation_config_env, config)
    return AugmentedDetectionTrainer
