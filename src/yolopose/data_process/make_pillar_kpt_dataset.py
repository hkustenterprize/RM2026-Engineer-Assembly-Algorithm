#!/usr/bin/env python3
"""Convert a 2-class 12-keypoint YOLO pose dataset into a 2-class 5-keypoint format.

Differences from make_pillar_only_dataset.py:
  - All label lines are kept (not just pillar / class-0).
  - Only pillar (class 0) instances carry the 5 pillar keypoints (TL/TR/BL/BR/ring).
  - Non-pillar instances keep their bounding box but have all keypoints zeroed out.

Input dataset layout:
    dataset_root/
        images/
            train/
            val/
        labels/
            train/
            val/
        dataset.yaml          # must contain a 'names' mapping

Output dataset layout:
    output_root/
        images/
            train/
            val/
        labels/
            train/
            val/
        dataset.yaml

Images are symlinked by default to avoid duplication.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


SPLITS = ("train", "val")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
EXPECTED_LABEL_LEN = 5 + 12 * 3  # 41  (source: 12-kpt format)
PILLAR_LABEL_LEN = 5 + 5 * 3  # 20  (output: 5-kpt format)
NUM_OUT_KPT = 5
ZERO_KPT_BLOCK = " ".join(["0 0 0"] * NUM_OUT_KPT)  # placeholder for non-pillar kps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a 5-keypoint YOLO pose dataset that keeps all class bounding boxes "
            "but only includes pillar keypoints for pillar (class-0) instances."
        )
    )
    parser.add_argument("--source", help="source dataset root")
    parser.add_argument("--output", help="output dataset root")
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="copy images instead of creating symlinks",
    )
    return parser.parse_args()


def ensure_layout(root: Path) -> None:
    for split in SPLITS:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def convert_label_line(line: str) -> str | None:
    """Convert one label line.

    - class 0 (pillar):     keep bbox + first 5 keypoints.
    - other classes:        keep bbox, zero out all 5 output keypoints.
    - empty / blank lines:  return None (skip).
    """
    vals = line.strip().split()
    if not vals:
        return None

    cls_id = int(float(vals[0]))

    if len(vals) != EXPECTED_LABEL_LEN:
        raise ValueError(
            f"Unexpected label length {len(vals)}; expected {EXPECTED_LABEL_LEN}"
        )

    bbox = " ".join(vals[1:5])  # cx cy w h

    if cls_id == 0:
        # pillar: keep first NUM_OUT_KPT keypoints
        kpts = " ".join(vals[5:PILLAR_LABEL_LEN])
    else:
        # non-pillar: zero out keypoints
        kpts = ZERO_KPT_BLOCK

    return f"{cls_id} {bbox} {kpts}"


def transfer_image(src: Path, dst: Path, copy_images: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_images:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def read_source_names(source_root: Path) -> dict:
    """Read class names from source dataset.yaml."""
    yaml_path = source_root / "dataset.yaml"
    with yaml_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", {0: "pillar"})
    # yaml.safe_load may return a list; normalise to dict
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    return names


def write_dataset_yaml(output_root: Path, names: dict) -> None:
    cfg = {
        "path": str(output_root),
        "train": "images/train",
        "val": "images/val",
        "kpt_shape": [NUM_OUT_KPT, 3],
        "flip_idx": [1, 0, 3, 2, 4],
        "names": names,
        "kpt_names": {
            0: "TL",
            1: "TR",
            2: "BL",
            3: "BR",
            4: "ring",
        },
    }
    with (output_root / "dataset.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def validate_source(source_root: Path) -> None:
    if not (source_root / "dataset.yaml").is_file():
        raise FileNotFoundError(f"Missing dataset.yaml: {source_root / 'dataset.yaml'}")
    for split in SPLITS:
        if not (source_root / "images" / split).is_dir():
            raise FileNotFoundError(
                f"Missing images split directory: {source_root / 'images' / split}"
            )
        if not (source_root / "labels" / split).is_dir():
            raise FileNotFoundError(
                f"Missing labels split directory: {source_root / 'labels' / split}"
            )


def convert_split(
    source_root: Path,
    output_root: Path,
    split: str,
    copy_images: bool,
) -> tuple[int, int, int]:
    src_img_dir = source_root / "images" / split
    src_lbl_dir = source_root / "labels" / split
    dst_img_dir = output_root / "images" / split
    dst_lbl_dir = output_root / "labels" / split

    n_images = 0
    n_pillar = 0
    n_other = 0

    for image_path in sorted(src_img_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue

        label_path = src_lbl_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label for image: {image_path}")

        dst_image = dst_img_dir / image_path.name
        transfer_image(image_path, dst_image, copy_images=copy_images)

        converted_lines: list[str] = []
        with label_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                converted = convert_label_line(raw_line)
                if converted is not None:
                    converted_lines.append(converted)
                    cls_id = int(converted.split()[0])
                    if cls_id == 0:
                        n_pillar += 1
                    else:
                        n_other += 1

        dst_label = dst_lbl_dir / label_path.name
        with dst_label.open("w", encoding="utf-8") as f:
            if converted_lines:
                f.write("\n".join(converted_lines) + "\n")

        n_images += 1

    return n_images, n_pillar, n_other


def main() -> None:
    args = parse_args()
    source_root = Path(args.source).resolve()
    output_root = Path(args.output).resolve()

    validate_source(source_root)
    names = read_source_names(source_root)
    ensure_layout(output_root)

    totals: dict[str, tuple[int, int, int]] = {}
    for split in SPLITS:
        totals[split] = convert_split(
            source_root=source_root,
            output_root=output_root,
            split=split,
            copy_images=args.copy_images,
        )

    write_dataset_yaml(output_root, names)

    print(f"Created dataset: {output_root}")
    for split in SPLITS:
        n_images, n_pillar, n_other = totals[split]
        print(
            f"  {split}: {n_images} images, "
            f"{n_pillar} pillar instances (with kpts), "
            f"{n_other} other instances (bbox only)"
        )


if __name__ == "__main__":
    main()
