#!/usr/bin/env python3
from __future__ import annotations

import importlib
import os
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import torch
from mmcv.ops import nms
from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmpose.registry import MODELS, TRANSFORMS
from mmpose.utils import register_all_modules
from pycocotools import mask as _pycocotools_mask
from xtcocotools.coco import COCO as _XTCoco

HRNET_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    versions = {}
    for name in (
        "torch",
        "torchvision",
        "numpy",
        "mmcv",
        "mmengine",
        "mmdet",
        "mmpose",
        "albumentations",
        "ultralytics",
    ):
        module = importlib.import_module(name)
        versions[name] = getattr(module, "__version__", "unknown")

    register_all_modules()
    for config_path in sorted((HRNET_ROOT / "configs").glob("*.py")):
        cfg = Config.fromfile(config_path)
        import_modules_from_strings(**cfg.custom_imports)
        model = MODELS.build(cfg.model)
        augmentation = next(
            step
            for step in cfg.train_pipeline
            if step["type"] == "PixelAlbumentation"
        )
        TRANSFORMS.build(augmentation)
        print(f"model: {config_path.name} -> {type(model).__name__}")

    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0]])
    scores = torch.tensor([0.9, 0.8])
    _, keep = nms(boxes, scores, 0.5)
    if keep.tolist() != [0]:
        raise RuntimeError(f"Unexpected MMCV NMS result: {keep.tolist()}")

    print(f"xtcocotools: {_XTCoco.__module__}")
    print(f"pycocotools: {_pycocotools_mask.__name__}")
    print("mmcv CPU ops: available")
    print(f"CUDA available: {torch.cuda.is_available()}")

    for name, version in versions.items():
        print(f"{name}: {version}")


if __name__ == "__main__":
    main()
