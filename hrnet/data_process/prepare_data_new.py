"""
Convert a YOLOPose-style dataset into COCO keypoint annotations for MMPose.

Dataset layout:
    dataset_root/
        images/
            train/
            val/
        labels/
            train/
            val/

YOLOPose label semantics:
    v = 0: keypoint is fully out of frame
    v = 1: keypoint is in frame but occluded
    v = 2: keypoint is visible

The COCO json written by this script preserves the raw 0/1/2 values in the
standard ``keypoints`` field. MMPose's default COCO parser will then collapse
this into ``keypoints_visible = (v > 0)``, which is exactly the supervision
needed for an "in-frame" classification head.

Compared with ``prepare_data.py``:
  - keeps the same pillar5 / exchange12 conversion modes
  - explicitly records raw visibility and in-frame flags in extra fields
  - defaults to ``exchange12`` because this is the main target for the new
    HRNet in-frame head
  - can optionally export visualization previews for the converted samples
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from glob import glob

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
EXCHANGE_SKELETON = [
    [6, 7],
    [7, 11],
    [11, 12],
    [12, 6],
    [7, 8],
    [8, 9],
    [9, 10],
    [10, 11],
]
FULL_SKELETON = PILLAR_SKELETON + EXCHANGE_SKELETON


def get_mode_spec(mode: str) -> tuple[str, list[str], list[list[int]]]:
    if mode == "pillar5":
        return "pillar", PILLAR_KPT_NAMES, PILLAR_SKELETON
    if mode == "exchange12":
        return "exchange_with_pillar", FULL_KPT_NAMES, FULL_SKELETON
    raise ValueError(f"Unsupported mode: {mode}")


def read_image_shape(path: str) -> tuple[int, int] | None:
    """Return image shape as (height, width) with minimal runtime assumptions."""
    try:
        import cv2  # type: ignore
    except ImportError:
        cv2 = None

    if cv2 is not None:
        img = cv2.imread(path)
        if img is not None:
            h, w = img.shape[:2]
            return int(h), int(w)

    try:
        from PIL import Image
    except ImportError:
        return None

    with Image.open(path) as img:
        width, height = img.size
    return int(height), int(width)


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
    for idx in range(total_kpts):
        base = 5 + idx * 3
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
) -> tuple[list[float], list[int], list[int], int, int]:
    coco_kpts: list[float] = []
    raw_visibility: list[int] = []
    in_frame_flags: list[int] = []
    visible_or_occluded = 0
    in_frame_count = 0

    for kx, ky, kv in keypoints_norm:
        px = kx * img_w
        py = ky * img_h
        v = int(kv)
        in_frame = int(v > 0)
        if v > 0:
            visible_or_occluded += 1
        if in_frame:
            in_frame_count += 1

        raw_visibility.append(v)
        in_frame_flags.append(in_frame)
        coco_kpts.extend([float(px), float(py), v])

    return coco_kpts, raw_visibility, in_frame_flags, visible_or_occluded, in_frame_count


def build_instance(
    rec: LabelRecord,
    keypoints_norm: list[tuple[float, float, int]],
    img_w: int,
    img_h: int,
) -> dict | None:
    x, y, w, h = yolo_bbox_to_coco_xywh(rec, img_w, img_h)
    if w < 2 or h < 2:
        return None

    (coco_kpts, raw_visibility, in_frame_flags, num_keypoints,
     num_keypoints_in_frame) = keypoints_to_coco(keypoints_norm, img_w, img_h)

    return {
        "bbox": [float(x), float(y), float(w), float(h)],
        "area": float(w * h),
        "num_keypoints": num_keypoints,
        "num_keypoints_in_frame": num_keypoints_in_frame,
        "keypoints": coco_kpts,
        "keypoints_raw_visibility": raw_visibility,
        "keypoints_in_frame": in_frame_flags,
    }


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
        instance = build_instance(rec, rec.all_kpts[:keep_kpts], img_w, img_h)
        if instance is not None:
            out.append(instance)
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
        instance = build_instance(rec, merged_kpts, img_w, img_h)
        if instance is not None:
            out.append(instance)

    skipped_exchange += len(pending_pillars)
    return out, skipped_exchange


def export_visualization(
    img_path: str,
    out_path: str,
    mode: str,
    instances: list[dict],
) -> bool:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("Warning: PIL is not installed, skipping visualization export")
        return False

    _, keypoint_names, skeleton = get_mode_spec(mode)
    try:
        with Image.open(img_path) as img:
            canvas = img.convert("RGB")
    except OSError as exc:
        print(f"Warning: failed to open {img_path} for visualization: {exc}")
        return False

    draw = ImageDraw.Draw(canvas, "RGBA")
    point_radius = max(4, round(min(canvas.size) * 0.007))
    bbox_width = max(2, point_radius // 2)
    color_by_visibility = {
        0: (160, 160, 160),
        1: (255, 170, 0),
        2: (0, 220, 120),
    }
    bbox_color = (255, 64, 64)
    edge_color = (80, 180, 255)

    legend_lines = [
        f"mode={mode}",
        "gray=v0 out-of-frame  orange=v1 occluded  green=v2 visible",
    ]
    legend_w = max(280, max(len(line) for line in legend_lines) * 7)
    legend_h = 14 + 18 * len(legend_lines)
    draw.rectangle([8, 8, 8 + legend_w, 8 + legend_h], fill=(0, 0, 0, 160))
    text_y = 14
    for line in legend_lines:
        draw.text((14, text_y), line, fill=(255, 255, 255))
        text_y += 18

    for inst_idx, inst in enumerate(instances, 1):
        x, y, w, h = inst["bbox"]
        draw.rectangle(
            [x, y, x + w, y + h],
            outline=bbox_color,
            width=bbox_width,
        )

        label = f"#{inst_idx} in_frame={inst['num_keypoints_in_frame']}/{len(keypoint_names)}"
        label_w = max(120, len(label) * 7)
        label_y = max(0.0, y - 20)
        draw.rectangle(
            [x, label_y, x + label_w, label_y + 18],
            fill=(0, 0, 0, 160),
        )
        draw.text((x + 4, label_y + 3), label, fill=(255, 255, 255))

        pts = []
        for idx in range(0, len(inst["keypoints"]), 3):
            pts.append(
                (
                    float(inst["keypoints"][idx]),
                    float(inst["keypoints"][idx + 1]),
                    int(inst["keypoints"][idx + 2]),
                )
            )

        for start, end in skeleton:
            sx, sy, sv = pts[start - 1]
            ex, ey, ev = pts[end - 1]
            if sv > 0 and ev > 0:
                draw.line([sx, sy, ex, ey], fill=edge_color, width=bbox_width)

        for name, (px, py, vis) in zip(keypoint_names, pts):
            color = color_by_visibility.get(vis, (255, 255, 255))
            ellipse_box = [px - point_radius, py - point_radius, px + point_radius, py + point_radius]
            if vis == 0:
                draw.ellipse(ellipse_box, outline=color, width=2)
            else:
                draw.ellipse(ellipse_box, fill=color, outline=(0, 0, 0), width=1)
            draw.text((px + point_radius + 2, py - point_radius - 2), name, fill=color)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)
    return True


def process_split(
    image_dir: str,
    label_dir: str,
    output_json: str,
    mode: str,
    total_kpts: int = 12,
    pillar_class_id: int = 0,
    exchange_class_id: int = 1,
    keep_kpts: int = 5,
    vis_dir: str | None = None,
    vis_num: int = 0,
) -> None:
    images = []
    annotations = []
    skipped_pairs = 0
    vis_written = 0
    category_name, keypoint_names, skeleton = get_mode_spec(mode)
    categories = [
        {
            "id": 1,
            "name": category_name,
            "supercategory": "object",
            "keypoints": keypoint_names,
            "skeleton": skeleton,
        }
    ]

    label_files = sorted(glob(os.path.join(label_dir, "*.txt")))
    img_id = 1
    ann_id = 1

    for label_path in label_files:
        stem = os.path.splitext(os.path.basename(label_path))[0]
        img_path = find_image(image_dir, stem)
        if img_path is None:
            continue

        image_shape = read_image_shape(img_path)
        if image_shape is None:
            continue
        img_h, img_w = image_shape

        images.append(
            {
                "id": img_id,
                "file_name": os.path.basename(img_path),
                "width": img_w,
                "height": img_h,
            }
        )

        with open(label_path, "r", encoding="utf-8") as f:
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
                    "num_keypoints_in_frame": inst["num_keypoints_in_frame"],
                    "keypoints": inst["keypoints"],
                    "keypoints_raw_visibility": inst["keypoints_raw_visibility"],
                    "keypoints_in_frame": inst["keypoints_in_frame"],
                }
            )
            ann_id += 1

        if vis_dir and vis_num > 0 and instances and vis_written < vis_num:
            out_name = f"{vis_written + 1:04d}_{os.path.basename(img_path)}"
            out_path = os.path.join(vis_dir, out_name)
            if export_visualization(
                img_path=img_path,
                out_path=out_path,
                mode=mode,
                instances=instances,
            ):
                vis_written += 1

        img_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco, f)

    print(f"Saved: {output_json}")
    print(f"  mode: {mode}")
    print(f"  images: {len(images)}")
    print(f"  annotations: {len(annotations)}")
    if mode == "exchange12":
        print(f"  skipped_unpaired_rows: {skipped_pairs}")
    if vis_dir and vis_num > 0:
        print(f"  visualizations: {vis_written}")
        print(f"  visualization_dir: {vis_dir}")


def run_mode(
    source: str,
    out_dir: str,
    split: str,
    mode: str,
    total_kpts: int,
    pillar_class_id: int,
    exchange_class_id: int,
    keep_kpts: int,
    vis_root: str | None,
    vis_num: int,
) -> None:
    prefix = "pillar" if mode == "pillar5" else "exchange12"
    vis_dir = None
    if vis_root:
        vis_dir = os.path.join(vis_root, prefix, split)
    process_split(
        image_dir=os.path.join(source, "images", split),
        label_dir=os.path.join(source, "labels", split),
        output_json=os.path.join(out_dir, f"{prefix}_{split}.json"),
        mode=mode,
        total_kpts=total_kpts,
        pillar_class_id=pillar_class_id,
        exchange_class_id=exchange_class_id,
        keep_kpts=keep_kpts,
        vis_dir=vis_dir,
        vis_num=vis_num,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLOPose -> COCO keypoint json for MMPose in-frame training"
    )
    parser.add_argument("source", help="YOLOPose dataset root")
    parser.add_argument(
        "--output",
        default="annotations",
        help="output directory relative to source, or absolute path",
    )
    parser.add_argument(
        "--mode",
        choices=["pillar5", "exchange12", "both"],
        default="exchange12",
        help="conversion target; exchange12 is the default for the new HRNet head",
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
    parser.add_argument(
        "--vis-dir",
        default=None,
        help="optional visualization directory; relative paths are resolved under --output",
    )
    parser.add_argument(
        "--vis-num",
        type=int,
        default=8,
        help="maximum preview images to export for each mode/split when --vis-dir is set",
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
    if args.vis_num < 0:
        raise ValueError("--vis-num must be non-negative")

    source = os.path.abspath(args.source)
    out_dir = (
        args.output
        if os.path.isabs(args.output)
        else os.path.join(source, args.output)
    )
    os.makedirs(out_dir, exist_ok=True)
    vis_root = None
    if args.vis_dir:
        vis_root = (
            args.vis_dir
            if os.path.isabs(args.vis_dir)
            else os.path.join(out_dir, args.vis_dir)
        )
        os.makedirs(vis_root, exist_ok=True)

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
                vis_root=vis_root,
                vis_num=args.vis_num,
            )


if __name__ == "__main__":
    main()
