#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings
from mmpose.utils import register_all_modules

# Runner resume uses MMEngine's legacy torch.load call. Only resume from
# checkpoints obtained from trusted sources.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

HRNET_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a LiteHRNet MMPose model")
    parser.add_argument("config", help="MMPose configuration file")
    parser.add_argument("--work-dir", help="directory for logs and checkpoints")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="resume from the latest checkpoint or from a specified checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    register_all_modules()
    cfg = Config.fromfile(config_path)
    import_modules_from_strings(**cfg.custom_imports)
    if args.work_dir:
        cfg.work_dir = str(Path(args.work_dir).expanduser().resolve())
    elif not cfg.get("work_dir"):
        cfg.work_dir = str(HRNET_ROOT / "runs" / config_path.stem)

    if args.resume:
        cfg.resume = True
        if args.resume != "auto":
            cfg.load_from = str(Path(args.resume).expanduser().resolve())

    Runner.from_cfg(cfg).train()


if __name__ == "__main__":
    main()
