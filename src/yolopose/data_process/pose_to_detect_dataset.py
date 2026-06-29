#!/usr/bin/env python3
"""Convert a YOLOPose dataset into a YOLO detect-only dataset.

Input labels may contain keypoints:
    cls cx cy w h kpt_x kpt_y kpt_v ...

Output labels contain only boxes:
    cls cx cy w h
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert YOLOPose dataset to YOLO detect dataset")
    parser.add_argument("--source", required=True, help="source YOLOPose dataset root")
    parser.add_argument("--output", required=True, help="output detect dataset root")
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="copy images instead of symlinking them",
    )
    parser.add_argument(
        "--include-classes",
        type=int,
        nargs="+",
        default=None,
        help="optional source class ids to keep; kept classes are remapped to 0..N-1",
    )
    parser.add_argument(
        "--allow-missing-label",
        action="store_true",
        help="create empty labels when an image has no source label file",
    )
    return parser.parse_args()


def read_dataset_yaml(source_root: Path) -> dict:
    yaml_path = source_root / "dataset.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Missing dataset.yaml: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"dataset.yaml must be a mapping: {yaml_path}")
    return data


def normalize_names(names) -> dict[int, str]:
    if isinstance(names, list):
        return {i: str(name) for i, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {0: "object"}


def build_class_map(source_names: dict[int, str], include_classes: list[int] | None):
    if include_classes is None:
        kept = sorted(source_names)
    else:
        missing = [cls_id for cls_id in include_classes if cls_id not in source_names]
        if missing:
            raise ValueError(f"include-classes not found in source names: {missing}")
        kept = include_classes
    class_map = {src_cls: dst_cls for dst_cls, src_cls in enumerate(kept)}
    output_names = {dst_cls: source_names[src_cls] for src_cls, dst_cls in class_map.items()}
    return class_map, output_names


def validate_layout(source_root: Path) -> None:
    if not (source_root / "images").is_dir():
        raise FileNotFoundError(f"Missing images dir: {source_root / 'images'}")
    if not (source_root / "labels").is_dir():
        raise FileNotFoundError(f"Missing labels dir: {source_root / 'labels'}")


def transfer_image(src: Path, dst: Path, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy_images:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def convert_label_line(line: str, class_map: dict[int, int]) -> str | None:
    parts = line.strip().split()
    if not parts:
        return None
    if len(parts) < 5:
        raise ValueError(f"Expected at least 5 label columns, got {len(parts)}: {line!r}")

    src_cls = int(float(parts[0]))
    if src_cls not in class_map:
        return None

    # Validate bbox values while preserving source precision in output.
    for value in parts[1:5]:
        float(value)
    return " ".join([str(class_map[src_cls]), *parts[1:5]])


def convert_label_file(src: Path | None, dst: Path, class_map: dict[int, int]) -> tuple[int, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    converted: list[str] = []
    skipped = 0

    if src is not None and src.is_file():
        with src.open("r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                try:
                    converted_line = convert_label_line(raw_line, class_map)
                except Exception as exc:
                    raise ValueError(f"{src}:{line_no}: {exc}") from exc
                if converted_line is None:
                    skipped += 1
                else:
                    converted.append(converted_line)

    dst.write_text(("\n".join(converted) + "\n") if converted else "", encoding="utf-8")
    return len(converted), skipped


def collect_images(source_root: Path, split: str) -> list[Path]:
    image_dir = source_root / "images" / split
    if not image_dir.is_dir():
        return []
    return [
        path
        for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]


def write_dataset_yaml(output_root: Path, names: dict[int, str], has_test: bool) -> None:
    cfg = {
        "path": str(output_root),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    if has_test:
        cfg["test"] = "images/test"
    with (output_root / "dataset.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def main() -> None:
    args = parse_args()
    source_root = Path(args.source).resolve()
    output_root = Path(args.output).resolve()

    validate_layout(source_root)
    source_yaml = read_dataset_yaml(source_root)
    source_names = normalize_names(source_yaml.get("names"))
    class_map, output_names = build_class_map(source_names, args.include_classes)

    for split in SPLITS:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_boxes = 0
    total_skipped = 0
    per_split: dict[str, tuple[int, int]] = {}

    for split in SPLITS:
        images = collect_images(source_root, split)
        if not images:
            continue

        split_boxes = 0
        for image_path in images:
            src_label = source_root / "labels" / split / f"{image_path.stem}.txt"
            if not src_label.is_file():
                if not args.allow_missing_label:
                    raise FileNotFoundError(f"Missing label for image: {image_path}")
                src_label_arg: Path | None = None
            else:
                src_label_arg = src_label

            dst_image = output_root / "images" / split / image_path.name
            dst_label = output_root / "labels" / split / f"{image_path.stem}.txt"
            transfer_image(image_path, dst_image, copy_images=args.copy_images)
            n_boxes, n_skipped = convert_label_file(src_label_arg, dst_label, class_map)
            split_boxes += n_boxes
            total_skipped += n_skipped

        per_split[split] = (len(images), split_boxes)
        total_images += len(images)
        total_boxes += split_boxes

    write_dataset_yaml(output_root, output_names, has_test="test" in per_split)

    mode = "copy" if args.copy_images else "symlink"
    print(f"Created detect dataset: {output_root}")
    print(f"  image mode: {mode}")
    print(f"  classes: {output_names}")
    for split, (n_images, n_boxes) in per_split.items():
        print(f"  {split}: {n_images} images, {n_boxes} boxes")
    print(f"  total: {total_images} images, {total_boxes} boxes, skipped labels={total_skipped}")


if __name__ == "__main__":
    main()
