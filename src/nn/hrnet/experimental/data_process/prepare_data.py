"""
Convert YOLO Pose dataset to COCO keypoint format for MMPose.

Supported modes:

1. pillar5
   Keep the pillar crop and only export the first 5 keypoints:
   ['TL', 'TR', 'BL', 'BR', 'ring'].

2. exchange12
   Keep the exchange crop and export all 12 keypoints:
   5 pillar keypoints + 7 exchange keypoints.
   This mode pairs the pillar row and exchange row from the same YOLO label file.

Expected dataset layout:
    dataset_root/
        images/
            train/
            val/
        labels/
            train/
            val/
        dataset.yaml

Usage:
    python prepare_data.py /path/to/dataset_root --output annotations --mode both
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from glob import glob

import cv2


PILLAR_KPT_NAMES = ["TL", "TR", "BL", "BR", "ring"]
EXCHANGE_KPT_NAMES = [
    "light_BR",
    "light_TR",
    "shell_R",
    "shell_M",
    "shell_L",
    "light_TL",
    "light_BL",
]
FULL_KPT_NAMES = PILLAR_KPT_NAMES + EXCHANGE_KPT_NAMES

PILLAR_SKELETON = [[1, 2], [2, 4], [4, 3], [3, 1]]
EXCHANGE_SKELETON = [[6, 7], [7, 11], [11, 12], [12, 6], [7, 8], [8, 9], [9, 10], [10, 11]]
FULL_SKELETON = PILLAR_SKELETON + EXCHANGE_SKELETON

PILLAR_FLIP_IDX = [1, 0, 3, 2, 4]
FULL_FLIP_IDX = [1, 0, 3, 2, 4, 11, 10, 9, 8, 7, 6, 5]


@dataclass
class LabelRecord:
    cls_id: int
    bbox_cx: float
    bbox_cy: float
    bbox_w: float
    bbox_h: float
    all_kpts: list[tuple[float, float, int]]


def parse_label_line(line: str, total_kpts: int = 12) -> LabelRecord:
    vals = list(map(float, line.strip().split()))
    expected_len = 5 + total_kpts * 3
    if len(vals) != expected_len:
        raise ValueError(f"Bad label length: got {len(vals)}, expected {expected_len}")

    cls_id = int(vals[0])
    bbox_cx, bbox_cy, bbox_w, bbox_h = vals[1], vals[2], vals[3], vals[4]

    all_kpts = []
    for i in range(total_kpts):
        base = 5 + i * 3
        all_kpts.append((vals[base], vals[base + 1], int(vals[base + 2])))

    return LabelRecord(
        cls_id=cls_id,
        bbox_cx=bbox_cx,
        bbox_cy=bbox_cy,
        bbox_w=bbox_w,
        bbox_h=bbox_h,
        all_kpts=all_kpts,
    )


def find_image(image_dir: str, stem: str) -> str | None:
    for ext in (".jpg", ".png", ".jpeg", ".bmp", ".webp"):
        candidate = os.path.join(image_dir, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def clamp_bbox_xywh(x: float, y: float, w: float, h: float, img_w: int, img_h: int):
    x = max(0.0, x)
    y = max(0.0, y)
    w = max(0.0, min(w, img_w - x))
    h = max(0.0, min(h, img_h - y))
    return x, y, w, h


def yolo_bbox_to_coco_xywh(
    rec: LabelRecord,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    x = (rec.bbox_cx - rec.bbox_w / 2.0) * img_w
    y = (rec.bbox_cy - rec.bbox_h / 2.0) * img_h
    w = rec.bbox_w * img_w
    h = rec.bbox_h * img_h
    return clamp_bbox_xywh(x, y, w, h, img_w, img_h)


def keypoints_to_coco(
    keypoints_norm: list[tuple[float, float, int]],
    img_w: int,
    img_h: int,
) -> tuple[list[float], int]:
    coco_kpts: list[float] = []
    valid_count = 0

    for kx, ky, kv in keypoints_norm:
        px = kx * img_w
        py = ky * img_h
        v = int(kv)
        if v > 0:
            valid_count += 1
        coco_kpts.extend([float(px), float(py), v])

    return coco_kpts, valid_count


def build_pillar5_instances(
    records: list[LabelRecord],
    img_w: int,
    img_h: int,
    pillar_class_id: int,
    keep_kpts: int,
) -> list[dict]:
    out = []
    for rec in records:
        if rec.cls_id != pillar_class_id:
            continue
        x, y, w, h = yolo_bbox_to_coco_xywh(rec, img_w, img_h)
        if w < 2 or h < 2:
            continue
        coco_kpts, valid_count = keypoints_to_coco(rec.all_kpts[:keep_kpts], img_w, img_h)
        out.append(
            {
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w * h),
                "num_keypoints": valid_count,
                "keypoints": coco_kpts,
            }
        )
    return out


def build_exchange12_instances(
    records: list[LabelRecord],
    img_w: int,
    img_h: int,
    pillar_class_id: int,
    exchange_class_id: int,
    pillar_kpts: int,
) -> tuple[list[dict], int]:
    out = []
    pending_pillars: list[LabelRecord] = []
    skipped_exchange = 0

    for rec in records:
        if rec.cls_id == pillar_class_id:
            pending_pillars.append(rec)
            continue

        if rec.cls_id != exchange_class_id:
            continue

        if not pending_pillars:
            skipped_exchange += 1
            continue

        pillar_rec = pending_pillars.pop(0)
        merged_kpts = pillar_rec.all_kpts[:pillar_kpts] + rec.all_kpts[pillar_kpts:]
        x, y, w, h = yolo_bbox_to_coco_xywh(rec, img_w, img_h)
        if w < 2 or h < 2:
            continue
        coco_kpts, valid_count = keypoints_to_coco(merged_kpts, img_w, img_h)
        out.append(
            {
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w * h),
                "num_keypoints": valid_count,
                "keypoints": coco_kpts,
            }
        )

    skipped_exchange += len(pending_pillars)
    return out, skipped_exchange


def process_split(
    image_dir: str,
    label_dir: str,
    output_json: str,
    mode: str,
    total_kpts: int = 12,
    pillar_class_id: int = 0,
    exchange_class_id: int = 1,
    keep_kpts: int = 5,
) -> None:
    images = []
    annotations = []
    skipped_pairs = 0

    if mode == "pillar5":
        categories = [
            {
                "id": 1,
                "name": "pillar",
                "supercategory": "object",
                "keypoints": PILLAR_KPT_NAMES,
                "skeleton": PILLAR_SKELETON,
            }
        ]
    elif mode == "exchange12":
        categories = [
            {
                "id": 1,
                "name": "exchange_with_pillar",
                "supercategory": "object",
                "keypoints": FULL_KPT_NAMES,
                "skeleton": FULL_SKELETON,
            }
        ]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    label_files = sorted(glob(os.path.join(label_dir, "*.txt")))
    img_id = 1
    ann_id = 1

    for label_path in label_files:
        stem = os.path.splitext(os.path.basename(label_path))[0]
        img_path = find_image(image_dir, stem)
        if img_path is None:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        images.append(
            {
                "id": img_id,
                "file_name": os.path.basename(img_path),
                "width": img_w,
                "height": img_h,
            }
        )

        with open(label_path, "r") as f:
            raw_lines = [line.strip() for line in f if line.strip()]

        records: list[LabelRecord] = []
        for line in raw_lines:
            try:
                records.append(parse_label_line(line, total_kpts=total_kpts))
            except ValueError as exc:
                print(f"Warning: {label_path}: {exc}")

        if mode == "pillar5":
            instances = build_pillar5_instances(
                records=records,
                img_w=img_w,
                img_h=img_h,
                pillar_class_id=pillar_class_id,
                keep_kpts=keep_kpts,
            )
        else:
            instances, skipped = build_exchange12_instances(
                records=records,
                img_w=img_w,
                img_h=img_h,
                pillar_class_id=pillar_class_id,
                exchange_class_id=exchange_class_id,
                pillar_kpts=keep_kpts,
            )
            skipped_pairs += skipped

        for inst in instances:
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": inst["bbox"],
                    "area": inst["area"],
                    "iscrowd": 0,
                    "num_keypoints": inst["num_keypoints"],
                    "keypoints": inst["keypoints"],
                }
            )
            ann_id += 1

        img_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(coco, f)

    print(f"Saved: {output_json}")
    print(f"  mode: {mode}")
    print(f"  images: {len(images)}")
    print(f"  annotations: {len(annotations)}")
    if mode == "exchange12":
        print(f"  skipped_unpaired_rows: {skipped_pairs}")


def run_mode(
    source: str,
    out_dir: str,
    split: str,
    mode: str,
    total_kpts: int,
    pillar_class_id: int,
    exchange_class_id: int,
    keep_kpts: int,
) -> None:
    prefix = "pillar" if mode == "pillar5" else "exchange12"
    process_split(
        image_dir=os.path.join(source, "images", split),
        label_dir=os.path.join(source, "labels", split),
        output_json=os.path.join(out_dir, f"{prefix}_{split}.json"),
        mode=mode,
        total_kpts=total_kpts,
        pillar_class_id=pillar_class_id,
        exchange_class_id=exchange_class_id,
        keep_kpts=keep_kpts,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLO Pose -> COCO keypoint json for MMPose"
    )
    parser.add_argument("source", help="YOLO Pose dataset root")
    parser.add_argument(
        "--output",
        default="annotations",
        help="output directory relative to source, or absolute path",
    )
    parser.add_argument(
        "--mode",
        choices=["pillar5", "exchange12", "both"],
        default="pillar5",
        help="pillar5 keeps current 5-kpt pillar crop; exchange12 uses exchange crop and all 12 keypoints",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="pillar class id (default 0)",
    )
    parser.add_argument(
        "--exchange-class-id",
        type=int,
        default=1,
        help="exchange class id (default 1)",
    )
    parser.add_argument(
        "--total-kpts",
        type=int,
        default=12,
        help="total keypoints per YOLO label row",
    )
    parser.add_argument(
        "--keep-kpts",
        type=int,
        default=5,
        help="pillar keypoint count; used by pillar5 and as the split point in exchange12",
    )
    args = parser.parse_args()

    if args.keep_kpts != len(PILLAR_KPT_NAMES):
        raise ValueError(
            f"--keep-kpts must be {len(PILLAR_KPT_NAMES)} for the current pillar definition"
        )
    if args.total_kpts != len(FULL_KPT_NAMES):
        raise ValueError(
            f"--total-kpts must be {len(FULL_KPT_NAMES)} for pillar5/exchange12 conversion"
        )

    source = os.path.abspath(args.source)
    out_dir = args.output if os.path.isabs(args.output) else os.path.join(source, args.output)
    os.makedirs(out_dir, exist_ok=True)

    modes = ["pillar5", "exchange12"] if args.mode == "both" else [args.mode]
    for mode in modes:
        for split in ("train", "val"):
            run_mode(
                source=source,
                out_dir=out_dir,
                split=split,
                mode=mode,
                total_kpts=args.total_kpts,
                pillar_class_id=args.class_id,
                exchange_class_id=args.exchange_class_id,
                keep_kpts=args.keep_kpts,
            )

    print("\nDone.")
    if "pillar5" in modes:
        print(f"pillar5 flip_idx: {PILLAR_FLIP_IDX}")
    if "exchange12" in modes:
        print(f"exchange12 flip_idx: {FULL_FLIP_IDX}")


if __name__ == "__main__":
    main()
