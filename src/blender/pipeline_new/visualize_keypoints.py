"""
visualize_keypoints.py — 在渲染图片上叠加关键点，验证投影精度
用法:
  conda run -n mujoco-sim python visualize_keypoints.py \
      --dataset_dir ../../data/装配站_v1.0 --n 20 --out_dir ../../data/装配站_v1.0/vis_kp

python /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/blender/scripts/visualize_keypoints.py --dataset_dir /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/data/new_new_rende_blue_test --n 5 --out_dir /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/data/new_new_rende_blue_test_vis

vis 值含义:
  2 = 可见  → 实心彩色圆 + 名称
  1 = 遮挡  → 橙色空心圆 + × + 名称(occ)
  0 = 视野外 → 不绘制

依赖: opencv-python, numpy  (conda run -n mujoco-sim python visualize_keypoints.py ...)
"""
import argparse
import json
import os
import cv2
import numpy as np

# 关键点颜色 (BGR) — vis=2 时使用
KP_COLORS = {
    # pillar 关键点
    "TL":   (0,   255, 0  ),   # 绿
    "TR":   (0,   200, 0  ),   # 深绿
    "BL":   (255, 0,   0  ),   # 蓝
    "BR":   (200, 0,   0  ),   # 深蓝
    "ring": (0,   0,   255),   # 红
    # exchange 关键点
    "light_BR": (255, 255, 0),   # 青
    "light_TR": (200, 255, 0),   # 浅青
    "shell_R":  (255, 0, 255),   # 品红
    "shell_M":  (200, 0, 200),   # 暗品红
    "shell_L":  (255, 100, 255), # 浅品红
    "light_TL": (200, 255, 0),   # 浅青
    "light_BL": (255, 255, 0),   # 青
}
OCC_COLOR   = (0, 165, 255)    # 橙色 — vis=1 遮挡专用颜色
LINE_COLOR  = (200, 200, 200)  # 四边形连线颜色
QUAD_ORDER  = ["TL", "TR", "BR", "BL"]  # 顺次连线顺序

# targets bbox 颜色表 (BGR): 最多支持 8 个 target
_TARGET_COLORS = [
    (0, 210, 50),     # pillar      — 绿
    (255, 100, 0),    # exchange    — 蓝
    (0, 200, 255),    # target_3    — 黄
    (200, 0, 255),    # target_4    — 紫
    (0, 100, 255),    # target_5    — 橙
    (255, 0, 100),    # target_6    — 玫红
    (100, 255, 100),  # target_7    — 浅绿
    (255, 200, 0),    # target_8    — 青
]


OOV_COLOR = (100, 100, 100)    # 灰色 — vis=0 视野外专用颜色


def _draw_target_bboxes(out, ann, targets_ann):
    """绘制各 target 的 bbox."""
    if targets_ann:
        for ti, (tgt_name, tgt_data) in enumerate(targets_ann.items()):
            color = _TARGET_COLORS[ti % len(_TARGET_COLORS)]
            bbox_t = tgt_data.get("bbox_2d")
            if bbox_t is not None:
                x0, y0, x1, y1 = bbox_t
                cv2.rectangle(out, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
                cv2.putText(out, tgt_name, (x0 + 2, y0 + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
    else:
        # 旧格式向后兼容
        bbox_ex = ann.get("bbox_2d_exchange")
        if bbox_ex is not None:
            x0, y0, x1, y1 = bbox_ex
            cv2.rectangle(out, (x0, y0), (x1, y1), _TARGET_COLORS[1], 1, cv2.LINE_AA)
            cv2.putText(out, "exch", (x0 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, _TARGET_COLORS[1], 1, cv2.LINE_AA)
        bbox = ann.get("bbox_2d")
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            cv2.rectangle(out, (x0, y0), (x1, y1), _TARGET_COLORS[0], 1, cv2.LINE_AA)
            cv2.putText(out, "pillar", (x0 + 2, y0 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, _TARGET_COLORS[0], 1, cv2.LINE_AA)


def _draw_quad_lines(out, kps, kp_names):
    """绘制四边形连线 (仅 vis=2 的角点)."""
    quad_pts_vis = []
    for name in QUAD_ORDER:
        if name not in kp_names:
            continue
        idx = kp_names.index(name)
        u, v, vis = kps[idx]
        quad_pts_vis.append((int(u), int(v), vis))
    for i in range(len(quad_pts_vis)):
        p1 = quad_pts_vis[i]
        p2 = quad_pts_vis[(i + 1) % len(quad_pts_vis)]
        if p1[2] == 2 and p2[2] == 2:
            cv2.line(out, (p1[0], p1[1]), (p2[0], p2[1]), LINE_COLOR, 1, cv2.LINE_AA)


def _draw_kps(out, kps, names, r):
    """绘制关键点 (vis=2 实心圆, vis=1 橙色×, vis=0 不绘制)."""
    for idx, name in enumerate(names):
        if idx >= len(kps):
            break
        u, v, vis = kps[idx]
        u_i, v_i = int(u), int(v)
        if vis == 2:
            color = KP_COLORS.get(name, (255, 255, 255))
            cv2.circle(out, (u_i, v_i), r, color, -1, cv2.LINE_AA)
            cv2.putText(out, name, (u_i + r + 2, v_i - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        elif vis == 1:
            cv2.circle(out, (u_i, v_i), r, OCC_COLOR, 2, cv2.LINE_AA)
            d = r // 2
            cv2.line(out, (u_i - d, v_i - d), (u_i + d, v_i + d), OCC_COLOR, 2)
            cv2.line(out, (u_i + d, v_i - d), (u_i - d, v_i + d), OCC_COLOR, 2)
            cv2.putText(out, f"{name}(occ)", (u_i + r + 2, v_i - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, OCC_COLOR, 1, cv2.LINE_AA)


def _fmt(val, fmt=".2f"):
    """Safe numeric format: returns '?' when *val* is missing/non-numeric."""
    if val is None or isinstance(val, str):
        return "?"
    try:
        return format(val, fmt)
    except (ValueError, TypeError):
        return str(val)


def _draw_meta_text(out, ann):
    """绘制元信息文字条 (兼容 render_dataset / render_dataset_new)."""
    meta = ann.get("meta", {})
    n_vis = meta.get("n_visible", "?")
    n_occ = meta.get("n_occluded", "?")

    dist = meta.get("dist_m") or meta.get("distance_m")
    elev = meta.get("elev_deg") or meta.get("pitch_deg")

    text = (f"dist={_fmt(dist)}m  "
            f"elev={_fmt(elev, '.1f')}deg  "
            f"light={meta.get('light_type', '?')}  "
            f"vis={n_vis} occ={n_occ}  "
            f"{ann.get('width', '?')}x{ann.get('height', '?')}")
    cv2.putText(out, text, (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA)


def draw_ann(img_bgr: np.ndarray, ann: dict, kp_names: list) -> np.ndarray:
    out = img_bgr.copy()
    kps = ann.get("keypoints_2d", [])
    h, w = out.shape[:2]
    r = max(5, w // 80)

    targets_ann = ann.get("targets", {})
    _draw_target_bboxes(out, ann, targets_ann)
    _draw_quad_lines(out, kps, kp_names)
    _draw_kps(out, kps, kp_names, r)

    # 副 target 关键点
    for tgt_name, tgt_data in targets_ann.items():
        sec_kps = tgt_data.get("keypoints_2d", [])
        sec_names = tgt_data.get("keypoint_names", [])
        if not sec_kps or not sec_names or sec_names == kp_names:
            continue
        _draw_kps(out, sec_kps, sec_names, r)

    # 收集所有 vis=0 (视野外) 的关键点, 以文字列表形式标注在图片左下角
    oov_names = []
    for i, name in enumerate(kp_names):
        if i < len(kps) and kps[i][2] == 0:
            oov_names.append(name)
    for tgt_name, tgt_data in targets_ann.items():
        sec_kps = tgt_data.get("keypoints_2d", [])
        sec_names = tgt_data.get("keypoint_names", [])
        if not sec_kps or not sec_names or sec_names == kp_names:
            continue
        for i, sname in enumerate(sec_names):
            if i < len(sec_kps) and sec_kps[i][2] == 0:
                oov_names.append(sname)
    if oov_names:
        oov_text = "OOV: " + ", ".join(oov_names)
        cv2.putText(out, oov_text, (8, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, OOV_COLOR, 1, cv2.LINE_AA)

    _draw_meta_text(out, ann)
    return out


def _scale_ann(ann, scale):
    """按 scale 倍率放大标注中的所有坐标."""
    scaled = dict(ann)
    scaled["keypoints_2d"] = [
        [u * scale, v * scale, vis] for u, v, vis in ann.get("keypoints_2d", [])]
    if ann.get("targets"):
        scaled_targets = {}
        for tgt_name, tgt_data in ann["targets"].items():
            st = dict(tgt_data)
            if tgt_data.get("bbox_2d"):
                x0, y0, x1, y1 = tgt_data["bbox_2d"]
                st["bbox_2d"] = [x0*scale, y0*scale, x1*scale, y1*scale]
            if tgt_data.get("keypoints_2d"):
                st["keypoints_2d"] = [
                    [u*scale, v*scale, vis] for u, v, vis in tgt_data["keypoints_2d"]]
            scaled_targets[tgt_name] = st
        scaled["targets"] = scaled_targets
    for key in ("bbox_2d", "bbox_2d_exchange"):
        if ann.get(key):
            x0, y0, x1, y1 = ann[key]
            scaled[key] = [x0*scale, y0*scale, x1*scale, y1*scale]
    return scaled


def main():
    parser = argparse.ArgumentParser(
        description="可视化 annotations.json 中的关键点投影（含遮挡状态）")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="含 annotations.json 和 images/ 子目录的数据集路径")
    parser.add_argument("--n",           type=int, default=10,
                        help="最多可视化多少张 (默认 10)")
    parser.add_argument("--out_dir",     type=str, default=None,
                        help="输出目录 (默认 <dataset_dir>/vis_kp/)")
    parser.add_argument("--scale",       type=int, default=0,
                        help="放大倍数 (0=自动: 短边<300 则放大, 否则不放大)")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.dataset_dir, "vis_kp")
    os.makedirs(out_dir, exist_ok=True)

    ann_path = os.path.join(args.dataset_dir, "annotations.json")
    with open(ann_path, encoding="utf-8") as f:
        data = json.load(f)

    kp_names = data["keypoint_names"]
    images   = data["images"][:args.n]

    for ann in images:
        img_path = os.path.join(args.dataset_dir, "images", ann["file_name"])
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[warn] 无法读取 {img_path}")
            continue

        # RGBA → BGR + 暗底
        if img.ndim == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3:4].astype(float) / 255.0
            bgr   = img[:, :, :3].astype(float)
            img   = (bgr * alpha + 30.0 * (1 - alpha)).clip(0, 255).astype(np.uint8)
        elif img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # 放大
        h, w = img.shape[:2]
        scale = args.scale if args.scale > 0 else max(1, 300 // min(w, h))
        if scale > 1:
            img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)
            ann_scaled = _scale_ann(ann, scale)
        else:
            ann_scaled = ann

        vis_img = draw_ann(img, ann_scaled, kp_names)
        out_path = os.path.join(out_dir, ann["file_name"])
        cv2.imwrite(out_path, vis_img)
        n_vis = ann["meta"].get("n_visible", "?")
        n_occ = ann["meta"].get("n_occluded", "?")
        print(f"  [{ann['id']:05d}] {ann['width']}x{ann['height']}  vis={n_vis} occ={n_occ}  → {out_path}")

    print(f"\n[visualize_keypoints] 完成, 共写出 {len(images)} 张至 {out_dir}")


if __name__ == "__main__":
    main()
