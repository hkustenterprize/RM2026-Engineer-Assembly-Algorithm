#!/usr/bin/env python3
"""Merge multiple YOLO-style datasets into one dataset.

Expected input layout for each dataset:
    dataset_root/
        images/
            train/
            val/
        labels/
            train/
            val/
        dataset.yaml

The script copies or symlinks images and labels into a single output dataset.
To avoid filename collisions, each sample is renamed with a dataset prefix:
    <dataset_name>__<original_stem>.<ext>

Example:

    python /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/yolopose/data_process/merge_yolo_datasets.py \
        --datasets /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/dataset_v10.0 \
        /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/dataset_v9.1 /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/dataset_v9.2/data_front_v9.2_compose \
        --output /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/dataset_v11.0 

    python merge_yolo_datasets.py \
        --datasets /path/to/ds_a /path/to/ds_b \
        --output /path/to/merged

    python merge_yolo_datasets.py \
        --datasets /path/to/ds_a /path/to/ds_b \
        --output /path/to/merged \
        --symlink
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "val")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge YOLO-format datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="dataset roots to merge",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="output merged dataset root",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="use symbolic links instead of copying files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="number of parallel file transfer workers",
    )
    return parser.parse_args()


def read_dataset_yaml_lines(dataset_root: Path) -> list[str]:
    yaml_path = dataset_root / "dataset.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing dataset.yaml: {yaml_path}")
    return yaml_path.read_text(encoding="utf-8").splitlines()


def read_dataset_yaml(dataset_root: Path) -> dict:
    yaml_path = dataset_root / "dataset.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing dataset.yaml: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"dataset.yaml must be a mapping: {yaml_path}")
    return data


def extract_metadata_block(lines: list[str]) -> list[str]:
    """Keep dataset metadata while dropping path/train/val fields."""
    kept: list[str] = []
    skip_prefixes = ("path:", "train:", "val:", "test:")
    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(prefix) for prefix in skip_prefixes):
            continue
        kept.append(line)
    return kept


def extract_schema(dataset_yaml: dict) -> dict:
    """Extract only the fields that define label/keypoint compatibility."""
    return {
        "kpt_shape": dataset_yaml.get("kpt_shape"),
        "flip_idx": dataset_yaml.get("flip_idx"),
        "names": dataset_yaml.get("names"),
        "kpt_names": dataset_yaml.get("kpt_names"),
    }


def validate_layout(dataset_root: Path) -> None:
    if not (dataset_root / "images").is_dir():
        raise FileNotFoundError(f"Missing images dir: {dataset_root / 'images'}")
    if not (dataset_root / "labels").is_dir():
        raise FileNotFoundError(f"Missing labels dir: {dataset_root / 'labels'}")
    if not (dataset_root / "dataset.yaml").is_file():
        raise FileNotFoundError(f"Missing dataset.yaml: {dataset_root / 'dataset.yaml'}")


def transfer_file(src: Path, dst: Path, symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def collect_pairs(dataset_root: Path, split: str) -> list[tuple[Path, Path]]:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split

    if not image_dir.is_dir() or not label_dir.is_dir():
        return []

    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTS or not image_path.is_file():
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")
        pairs.append((image_path, label_path))
    return pairs


def build_dataset_prefixes(dataset_roots: list[Path]) -> dict[Path, str]:
    """Create unique, readable filename prefixes for merged samples."""
    name_counts = Counter(dataset_root.name for dataset_root in dataset_roots)
    used_prefixes: set[str] = set()
    prefixes: dict[Path, str] = {}

    for dataset_root in dataset_roots:
        parts = [dataset_root.name]
        ancestor = dataset_root.parent

        if name_counts[dataset_root.name] > 1:
            parts.insert(0, ancestor.name)
            ancestor = ancestor.parent

        prefix = "__".join(parts)
        while prefix in used_prefixes:
            parts.insert(0, ancestor.name)
            ancestor = ancestor.parent
            prefix = "__".join(parts)

        used_prefixes.add(prefix)
        prefixes[dataset_root] = prefix

    return prefixes


def write_output_yaml(output_root: Path, metadata_lines: list[str]) -> None:
    yaml_lines = [
        f"path: {output_root}",
        "train: images/train",
        "val: images/val",
        "",
        *metadata_lines,
        "",
    ]
    (output_root / "dataset.yaml").write_text(
        "\n".join(yaml_lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()

    dataset_roots = [Path(p).resolve() for p in args.datasets]
    output_root = Path(args.output).resolve()

    if not dataset_roots:
        raise ValueError("No datasets provided")

    for dataset_root in dataset_roots:
        validate_layout(dataset_root)
    dataset_prefixes = build_dataset_prefixes(dataset_roots)

    first_yaml_lines = read_dataset_yaml_lines(dataset_roots[0])
    metadata_lines = extract_metadata_block(first_yaml_lines)
    reference_schema = extract_schema(read_dataset_yaml(dataset_roots[0]))

    output_root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    transfer_jobs: list[tuple[Path, Path]] = []
    total_per_split = {split: 0 for split in SPLITS}

    for dataset_root in dataset_roots:
        dataset_name = dataset_prefixes[dataset_root]

        current_schema = extract_schema(read_dataset_yaml(dataset_root))
        if current_schema != reference_schema:
            raise ValueError(
                f"dataset.yaml metadata mismatch: {dataset_root}\n"
                "All merged datasets must share the same class/keypoint schema."
            )

        for split in SPLITS:
            pairs = collect_pairs(dataset_root, split)
            total_per_split[split] += len(pairs)

            for image_path, label_path in pairs:
                new_stem = f"{dataset_name}__{image_path.stem}"
                dst_image = output_root / "images" / split / f"{new_stem}{image_path.suffix.lower()}"
                dst_label = output_root / "labels" / split / f"{new_stem}.txt"
                transfer_jobs.append((image_path, dst_image))
                transfer_jobs.append((label_path, dst_label))

    mode = "symlink" if args.symlink else "copy"
    print(
        f"Merging {len(dataset_roots)} datasets -> {output_root} "
        f"({mode}, workers={args.workers})"
    )
    for dataset_root in dataset_roots:
        print(f"Prefix: {dataset_prefixes[dataset_root]} <- {dataset_root}")
    print(
        f"Samples: train={total_per_split['train']}, val={total_per_split['val']}"
    )

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [
            executor.submit(transfer_file, src, dst, args.symlink)
            for src, dst in transfer_jobs
        ]
        for future in as_completed(futures):
            future.result()

    write_output_yaml(output_root, metadata_lines)
    print("Done.")
    print(f"dataset.yaml: {output_root / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
