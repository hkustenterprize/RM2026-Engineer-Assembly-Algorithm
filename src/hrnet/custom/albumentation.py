from __future__ import annotations

from copy import deepcopy

import albumentations as A
from mmcv.transforms import BaseTransform
from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PixelAlbumentation(BaseTransform):
    """Apply Albumentations 2 pixel transforms to an MMPose image."""

    def __init__(self, transforms: list[dict]) -> None:
        self.transforms = deepcopy(transforms)
        self.augmentation = A.Compose(
            [self._build_transform(config) for config in self.transforms]
        )

    def _build_transform(self, config: dict):
        args = deepcopy(config)
        transform_type = args.pop("type")
        if "transforms" in args:
            args["transforms"] = [
                self._build_transform(child) for child in args["transforms"]
            ]
        return getattr(A, transform_type)(**args)

    def transform(self, results: dict) -> dict:
        results["img"] = self.augmentation(image=results["img"])["image"]
        return results
