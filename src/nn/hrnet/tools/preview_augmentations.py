#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2
import numpy as np
from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmpose.registry import DATASETS
from mmpose.utils import register_all_modules

HRNET_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview transformed training samples")
    parser.add_argument("config", help="MMPose configuration file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-items", type=int, default=80)
    return parser.parse_args()


def tensor_to_bgr(inputs) -> np.ndarray:
    image = inputs.detach().cpu().numpy().transpose(1, 2, 0)
    image = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def main() -> None:
    args = parse_args()
    if args.max_items < 1:
        raise ValueError("--max-items must be positive")

    register_all_modules()
    cfg = Config.fromfile(Path(args.config).expanduser().resolve())
    import_modules_from_strings(**cfg.custom_imports)
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    count = min(args.max_items, len(dataset))
    for index in range(count):
        sample = dataset[index]
        image = tensor_to_bgr(sample["inputs"])
        data_sample = sample["data_samples"]
        keypoints = np.asarray(data_sample.gt_instances.keypoints)[0]
        visible = np.asarray(data_sample.gt_instances.keypoints_visible)[0]
        for point, is_visible in zip(keypoints, visible):
            if float(np.asarray(is_visible).reshape(-1)[0]) <= 0:
                continue
            x, y = np.round(point).astype(int)
            cv2.circle(image, (x, y), 4, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.imwrite(str(output_dir / f"{index:04d}.jpg"), image)

    print(f"Saved {count} transformed samples to: {output_dir}")


if __name__ == "__main__":
    main()
