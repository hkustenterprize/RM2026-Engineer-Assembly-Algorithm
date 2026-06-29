import argparse
import json
import os
import random
from pathlib import Path

import cv2

'''
python /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/data_process/viz_data.py \
  /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/data/dataset_v7.0 \
  --ann /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/data/dataset_v7.0_annotations/pillar_train.json \
  --img-dir images/train \
  --out-dir /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/vis_coco_v7.0_train \
  --num-samples 10
  
  
python /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/data_process/viz_data.py \
  /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/dataset_v7.0 \
  --ann /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/dataset_v7.0_annotations/pillar_val.json \
  --img-dir images/val \
  --out-dir /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/vis_coco_v7.1_val \
  --num-samples 10
'''



KPT_NAMES = ["TL", "TR", "BL", "BR", "ring"]
FULL_KPT_NAMES = [
    "TL",
    "TR",
    "BL",
    "BR",
    "ring",
    "light_BR",
    "light_TR",
    "shell_R",
    "shell_M",
    "shell_L",
    "light_TL",
    "light_BL",
]

# BGR colors for OpenCV
KPT_COLORS = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 128, 0),
    (255, 200, 0),
    (128, 255, 0),
    (0, 255, 128),
    (0, 200, 255),
    (128, 0, 255),
    (255, 0, 128),
]

SKELETON = [(0, 1), (1, 3), (3, 2), (2, 0)]
FULL_SKELETON = SKELETON + [(5, 6), (10, 11), (11, 5), (6, 7), (7, 8), (8, 9), (9, 10)]
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_coco(json_path):
    with open(json_path, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    return coco, images, anns_by_img


def shorten_kpt_label(name):
    label = str(name).strip()
    lower = label.lower()
    if lower == "ring":
        return "RG"
    if lower.startswith("light_"):
        return "L" + label.split("_", 1)[1].upper()
    if lower.startswith("shell_"):
        return "S" + label.split("_", 1)[1].upper()
    return label.upper()


def draw_outlined_text(canvas, text, org, color, font_scale=0.85, thickness=2):
    cv2.putText(
        canvas,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        max(3, thickness + 2),
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_inference_style_overlay(
    img,
    bbox_xyxy,
    keypoints_xy,
    visible=None,
    keypoint_names=None,
    skeleton=None,
    keypoint_colors=None,
    show_labels=True,
):
    out = img.copy()
    keypoint_names = keypoint_names or KPT_NAMES
    skeleton = skeleton or SKELETON
    keypoint_colors = keypoint_colors or KPT_COLORS

    x0, y0, x1, y1 = [int(round(v)) for v in bbox_xyxy]
    cv2.rectangle(out, (x0, y0), (x1, y1), (80, 220, 80), 10, cv2.LINE_AA)

    pts = [(float(x), float(y)) for x, y in keypoints_xy]
    vis = list(visible) if visible is not None else [2] * len(pts)

    for a, b in skeleton:
        if a >= len(pts) or b >= len(pts):
            continue
        if vis[a] > 0 and vis[b] > 0:
            ax, ay = [int(round(v)) for v in pts[a]]
            bx, by = [int(round(v)) for v in pts[b]]
            cv2.line(out, (ax, ay), (bx, by), (220, 220, 220), 5, cv2.LINE_AA)

    for k, (x_f, y_f) in enumerate(pts):
        if vis[k] <= 0:
            continue
        x, y = int(round(x_f)), int(round(y_f))
        color = keypoint_colors[k % len(keypoint_colors)]
        cv2.circle(out, (x, y), 16, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, (x, y), 12, color, -1, cv2.LINE_AA)
        if show_labels:
            raw_label = keypoint_names[k] if k < len(keypoint_names) else f"kp{k}"
            draw_outlined_text(out, shorten_kpt_label(raw_label), (x + 22, y - 22), color)

    return out


def draw_annotation(img, ann, show_labels=True):
    out = img.copy()

    x, y, w, h = ann["bbox"]
    kpts = ann["keypoints"]
    n_kpts = len(kpts) // 3
    keypoint_names = KPT_NAMES if n_kpts == 5 else FULL_KPT_NAMES[:n_kpts]
    skeleton = SKELETON if n_kpts == 5 else FULL_SKELETON

    pts = [(float(kpts[i * 3]), float(kpts[i * 3 + 1])) for i in range(n_kpts)]
    visible = [int(kpts[i * 3 + 2]) for i in range(n_kpts)]
    return draw_inference_style_overlay(
        out,
        bbox_xyxy=[x, y, x + w, y + h],
        keypoints_xy=pts,
        visible=visible,
        keypoint_names=keypoint_names,
        skeleton=skeleton,
        show_labels=show_labels,
    )


def load_dataset_yaml(yaml_path):
    import yaml

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    root = Path(cfg.get("path") or Path(yaml_path).parent).expanduser()
    return cfg, root


def normalize_mapping(mapping):
    if isinstance(mapping, dict):
        return {int(k): str(v) for k, v in mapping.items()}
    if isinstance(mapping, list):
        return {i: str(v) for i, v in enumerate(mapping)}
    return {}


def find_image(image_dir, stem):
    for ext in IMAGE_EXTS:
        path = Path(image_dir) / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def parse_yolo_label_line(line, img_w, img_h, total_kpts):
    vals = [float(v) for v in line.strip().split()]
    if len(vals) < 5:
        return None
    cls_id = int(vals[0])
    cx, cy, bw, bh = vals[1:5]
    x0 = (cx - bw / 2.0) * img_w
    y0 = (cy - bh / 2.0) * img_h
    x1 = (cx + bw / 2.0) * img_w
    y1 = (cy + bh / 2.0) * img_h

    keypoints = []
    visible = []
    for i in range(total_kpts):
        base = 5 + i * 3
        if base + 2 >= len(vals):
            break
        kx, ky, kv = vals[base : base + 3]
        keypoints.append((kx * img_w, ky * img_h))
        visible.append(int(kv))
    return cls_id, [x0, y0, x1, y1], keypoints, visible


def visualize_yolo_dataset(dataset_yaml, split, out_dir, num_samples, seed, show_labels=True, class_id=None):
    cfg, root = load_dataset_yaml(dataset_yaml)
    image_dir = root / cfg.get(split, f"images/{split}")
    label_dir = root / "labels" / split
    kpt_shape = cfg.get("kpt_shape", [5, 3])
    total_kpts = int(kpt_shape[0])
    names = normalize_mapping(cfg.get("kpt_names"))
    keypoint_names = [names.get(i, f"kp{i}") for i in range(total_kpts)]
    skeleton = SKELETON if total_kpts == 5 else FULL_SKELETON

    label_files = sorted(label_dir.glob("*.txt"))
    random.seed(seed)
    random.shuffle(label_files)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for label_path in label_files:
        if saved >= num_samples:
            break
        img_path = find_image(image_dir, label_path.stem)
        if img_path is None:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        vis = img.copy()
        drew = False
        for raw in label_path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            parsed = parse_yolo_label_line(raw, w, h, total_kpts)
            if parsed is None:
                continue
            cls_id, bbox, keypoints, visible = parsed
            if class_id is not None and cls_id != class_id:
                continue
            # Inference HRNet is pillar-only, so class 0 uses the first 5 keypoints.
            if cls_id == 0 and total_kpts >= 5:
                kp_names = keypoint_names[:5]
                kp_xy = keypoints[:5]
                kp_vis = visible[:5]
                skel = SKELETON
            else:
                kp_names = keypoint_names
                kp_xy = keypoints
                kp_vis = visible
                skel = skeleton
            vis = draw_inference_style_overlay(
                vis,
                bbox_xyxy=bbox,
                keypoints_xy=kp_xy,
                visible=kp_vis,
                keypoint_names=kp_names,
                skeleton=skel,
                show_labels=show_labels,
            )
            drew = True

        if not drew:
            continue
        save_path = out_dir / img_path.name
        cv2.imwrite(str(save_path), vis)
        saved += 1

    print(f"Saved {saved} YOLO training visualizations to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize pillar COCO keypoint annotations")
    parser.add_argument("data_root", help="dataset root, e.g. /path/to/blender3_2400_v6.0_cropped")
    parser.add_argument("--ann", default="pillar_train.json", help="annotation file name under data_root")
    parser.add_argument("--img-dir", default="images/train", help="image dir under data_root")
    parser.add_argument("--out-dir", default="vis_pillar", help="output dir")
    parser.add_argument("--num-samples", type=int, default=50, help="number of images to visualize")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--yolo", action="store_true", help="read data_root as YOLO dataset.yaml")
    parser.add_argument("--split", default="train", help="YOLO split to visualize")
    parser.add_argument("--class-id", type=int, default=None, help="optional YOLO class id filter")
    parser.add_argument("--hide-labels", action="store_true")
    args = parser.parse_args()

    if args.yolo:
        visualize_yolo_dataset(
            dataset_yaml=args.data_root,
            split=args.split,
            out_dir=args.out_dir,
            num_samples=args.num_samples,
            seed=args.seed,
            show_labels=not args.hide_labels,
            class_id=args.class_id,
        )
        return

    json_path = os.path.join(args.data_root, args.ann)
    img_dir = os.path.join(args.data_root, args.img_dir)
    out_dir = os.path.join(args.data_root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    _, images, anns_by_img = load_coco(json_path)

    image_ids = list(anns_by_img.keys())
    if len(image_ids) == 0:
        print("No annotations found.")
        return

    random.seed(args.seed)
    random.shuffle(image_ids)
    image_ids = image_ids[: min(args.num_samples, len(image_ids))]

    saved = 0
    for image_id in image_ids:
        img_info = images[image_id]
        file_name = img_info["file_name"]
        img_path = os.path.join(img_dir, file_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: failed to read {img_path}")
            continue

        vis = img.copy()
        anns = anns_by_img[image_id]

        for ann in anns:
            vis = draw_annotation(vis, ann, show_labels=True)

        save_path = os.path.join(out_dir, file_name)
        cv2.imwrite(save_path, vis)
        saved += 1

    print(f"Saved {saved} visualizations to: {out_dir}")


if __name__ == "__main__":
    main()