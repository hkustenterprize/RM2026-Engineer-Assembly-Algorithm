#!/usr/bin/env python3
"""Convert a 2-class 12-keypoint YOLO pose dataset into pillar-only 1-class 5-keypoint format.

Input dataset layout:
    dataset_root/
        images/
            train/
            val/
        labels/
            train/
            val/
        dataset.yaml

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
EXPECTED_LABEL_LEN = 5 + 12 * 3
PILLAR_LABEL_LEN = 5 + 5 * 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create pillar-only YOLO pose dataset from full dataset"
    )
    parser.add_argument("source", help="source dataset root")
    parser.add_argument("output", help="output dataset root")
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
    vals = line.strip().split()
    if not vals:
        return None

    cls_id = int(float(vals[0]))
    if cls_id != 0:
        return None

    if len(vals) != EXPECTED_LABEL_LEN:
        raise ValueError(
            f"Unexpected label length {len(vals)}; expected {EXPECTED_LABEL_LEN}"
        )

    pillar_vals = vals[:PILLAR_LABEL_LEN]
    pillar_vals[0] = "0"
    return " ".join(pillar_vals)


def transfer_image(src: Path, dst: Path, copy_images: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_images:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def write_dataset_yaml(output_root: Path) -> None:
    cfg = {
        "path": str(output_root),
        "train": "images/train",
        "val": "images/val",
        "kpt_shape": [5, 3],
        "flip_idx": [1, 0, 3, 2, 4],
        "names": {0: "pillar"},
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


def convert_split(source_root: Path, output_root: Path, split: str, copy_images: bool) -> tuple[int, int]:
    src_img_dir = source_root / "images" / split
    src_lbl_dir = source_root / "labels" / split
    dst_img_dir = output_root / "images" / split
    dst_lbl_dir = output_root / "labels" / split

    n_images = 0
    n_pillar_instances = 0

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

        dst_label = dst_lbl_dir / label_path.name
        with dst_label.open("w", encoding="utf-8") as f:
            if converted_lines:
                f.write("\n".join(converted_lines) + "\n")

        n_images += 1
        n_pillar_instances += len(converted_lines)

    return n_images, n_pillar_instances


def main() -> None:
    args = parse_args()
    source_root = Path(args.source).resolve()
    output_root = Path(args.output).resolve()

    validate_source(source_root)
    ensure_layout(output_root)

    totals = {}
    for split in SPLITS:
        totals[split] = convert_split(
            source_root=source_root,
            output_root=output_root,
            split=split,
            copy_images=args.copy_images,
        )

    write_dataset_yaml(output_root)

    print(f"Created pillar-only dataset: {output_root}")
    for split in SPLITS:
        n_images, n_instances = totals[split]
        print(f"  {split}: {n_images} images, {n_instances} pillar instances")


if __name__ == "__main__":
    main()
