#!/usr/bin/env python3
"""
compose_dataset.py — Blender RGBA 合成图 → YOLO Pose 格式数据集

流程:
  1. 读取 Blender 生成的 annotations.json + RGBA 图像
  2. 每张输出图从 crop 池中随机选取 1~2 个装配柱合成到背景图上
  3. Luma 匹配: 按材质前景亮度选择亮度相近的背景图 (非强行拉亮)
  4. 双重遮挡约束:
       a) mask overlap ratio ≤ max_mask_overlap_ratio
       b) 新 pillar 导致已放置实例新增遮挡 kp 数 ≤ max_kp_occlusion_delta
  5. 遮挡传播: 后贴的 pillar (顶层) 的前景 mask 覆盖先贴实例的 kp → vis 降级为 1
  6. 随机 train/val split
  7. 输出 YOLO Pose 格式数据集 + dataset.yaml

用法:


  python compose_dataset.py \\
    --blender_dir ../data/装配站_v1.0 --output_dir ./pillar_dataset

  python /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/yolopose/data_process/compose_dataset.py \
    --blender_dir /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/data_side_off_v9.0/data_side_off_v9.0_compose \
    --bg_dir /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/data/background \
    --output_dir /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/dataset_side_off_v9.0 \
    --min_pillars 1 --max_pillars 2 \
    --vis_output /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/v9.0_vis \
    --val_split 0.1 --seed 27

"""

import argparse
import datetime
import json
import os
import random
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
#  常量
# ──────────────────────────────────────────────────────────────────────────────

SUPPORTED_IMAGE_EXTS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
    ".JPG",
    ".JPEG",
    ".PNG",
    ".BMP",
    ".WEBP",
    ".TIF",
    ".TIFF",
}
ALPHA_THRESH: float = 0.1  # alpha ≥ 此值视为前景像素
JPEG_QUALITY: int = 92  # 合成图输出质量

# 关键点可视化颜色 (BGR)
_PIL_KP_COLORS = [
    (0, 255, 0),
    (0, 200, 255),
    (255, 100, 0),
    (200, 0, 255),
    (0, 128, 255),
]  # pillar
_EXC_KP_COLORS = [
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (128, 255, 128),
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 128),
]  # exchange
_OCC_COLOR = (0, 165, 255)  # 橙色 (遮挡)
_OOV_COLOR = (100, 100, 100)  # 灰色 (视野外)


# ──────────────────────────────────────────────────────────────────────────────
#  可视化
# ──────────────────────────────────────────────────────────────────────────────


def _draw_kp_set(
    out,
    kps_2d,
    kps_vis,
    colors,
    radius,
    thick,
    edges=None,
    bbox=None,
    bbox_color=(80, 200, 80),
):
    """绘制一组关键点 + 可选连线 + bbox."""
    vis2 = {}
    for i, (kp, vis) in enumerate(zip(kps_2d, kps_vis)):
        x, y = int(kp[0]), int(kp[1])
        color = colors[i % len(colors)]
        if vis == 2:
            cv2.circle(out, (x, y), radius, color, -1, cv2.LINE_AA)
            vis2[i] = (x, y)
        elif vis == 1:
            cv2.circle(out, (x, y), radius, _OCC_COLOR, thick, cv2.LINE_AA)
            d = radius - 1
            cv2.line(
                out, (x - d, y - d), (x + d, y + d), _OCC_COLOR, thick, cv2.LINE_AA
            )
            cv2.line(
                out, (x + d, y - d), (x - d, y + d), _OCC_COLOR, thick, cv2.LINE_AA
            )
    if edges:
        for a, b in edges:
            if a in vis2 and b in vis2:
                cv2.line(out, vis2[a], vis2[b], (200, 200, 200), 1, cv2.LINE_AA)
    if bbox is not None:
        x0, y0, x1, y1 = (int(v) for v in bbox)
        cv2.rectangle(out, (x0, y0), (x1, y1), bbox_color, 1, cv2.LINE_AA)


def draw_kp_vis(
    img_bgr: np.ndarray, instances: list, pil_kp_names: list, exc_kp_names: list
) -> np.ndarray:
    """在 BGR 图像上绘制 pillar + exchange 关键点和 bbox.

    vis=2: 实心彩色圆  vis=1: 橙色空心圆+X  vis=0: 跳过
    """
    out = img_bgr.copy()
    h, w = out.shape[:2]
    radius = max(4, min(w, h) // 60)
    thick = max(1, radius // 3)

    for inst in instances:
        pil_kps = inst["kps_2d"]
        pil_vis = inst.get("kps_vis", [2] * len(pil_kps))
        pil_edges = [(0, 1), (2, 3), (0, 2), (1, 3)]
        if len(pil_kps) >= 5:
            pil_edges += [(4, 0), (4, 1), (4, 2), (4, 3)]
        _draw_kp_set(
            out,
            pil_kps,
            pil_vis,
            _PIL_KP_COLORS,
            radius,
            thick,
            edges=pil_edges,
            bbox=inst["bbox"],
            bbox_color=(80, 200, 80),
        )

        exc_kps = inst.get("exc_kps_2d", [])
        exc_vis = inst.get("exc_kps_vis", [])
        exc_bbox = inst.get("exc_bbox")
        if exc_kps:
            _draw_kp_set(
                out,
                exc_kps,
                exc_vis,
                _EXC_KP_COLORS,
                radius,
                thick,
                bbox=exc_bbox,
                bbox_color=(200, 200, 80),
            )

    return out


def _kps_to_yolo(
    kps_2d: list, kps_vis: list, img_w: int, img_h: int, n_slots: int
) -> list:
    """将关键点列表转换为 YOLO 格式的字符串片段, 不足 n_slots 的补 0 0 0."""
    parts = []
    for i in range(n_slots):
        if i < len(kps_2d) and i < len(kps_vis) and kps_vis[i] != 0:
            parts.append(
                f"{kps_2d[i][0]/img_w:.6f} {kps_2d[i][1]/img_h:.6f} {kps_vis[i]}"
            )
        else:
            parts.append("0.000000 0.000000 0")
    return parts


def write_yolo_label(
    path: Path,
    instances: list,
    img_w: int,
    img_h: int,
    n_pil_kps: int = 5,
    n_exc_kps: int = 7,
) -> None:
    """写出 YOLO Pose .txt 标签 (双类别: pillar + exchange).

    每个 instance 产生两行:
      class=0 (pillar):   pillar_bbox + [5 pillar kps | 7 padding]
      class=1 (exchange): exchange_bbox + [5 padding | 7 exchange kps]

    kpt_shape = [n_pil_kps + n_exc_kps, 3]
    vis: 0=视野外(占位 0 0 0)  1=遮挡  2=可见
    """
    n_total = n_pil_kps + n_exc_kps
    pad_exc = ["0.000000 0.000000 0"] * n_exc_kps
    pad_pil = ["0.000000 0.000000 0"] * n_pil_kps
    lines = []
    for inst in instances:
        # ── class 0: pillar ──
        x0, y0, x1, y1 = inst["bbox"]
        cx = ((x0 + x1) / 2) / img_w
        cy = ((y0 + y1) / 2) / img_h
        bw = (x1 - x0) / img_w
        bh = (y1 - y0) / img_h
        pil_vis = inst.get("kps_vis", [2] * len(inst["kps_2d"]))
        pil_parts = _kps_to_yolo(inst["kps_2d"], pil_vis, img_w, img_h, n_pil_kps)
        lines.append(
            " ".join([f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"] + pil_parts + pad_exc)
        )

        # ── class 1: exchange (如果有) ──
        exc_bbox = inst.get("exc_bbox")
        exc_kps = inst.get("exc_kps_2d", [])
        if exc_bbox is not None and exc_kps:
            ex0, ey0, ex1, ey1 = exc_bbox
            ecx = ((ex0 + ex1) / 2) / img_w
            ecy = ((ey0 + ey1) / 2) / img_h
            ebw = (ex1 - ex0) / img_w
            ebh = (ey1 - ey0) / img_h
            exc_vis = inst.get("exc_kps_vis", [2] * len(exc_kps))
            exc_parts = _kps_to_yolo(exc_kps, exc_vis, img_w, img_h, n_exc_kps)
            lines.append(
                " ".join(
                    [f"1 {ecx:.6f} {ecy:.6f} {ebw:.6f} {ebh:.6f}"] + pad_pil + exc_parts
                )
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def bbox_2d_from_kps_vis(
    kps_2d: list,
    kps_vis: list,
    img_w: int,
    img_h: int,
    margin: float = 2.0,
) -> list[float]:
    """当 Blender 未写出 seg bbox (None) 时, 用关键点外接框替代 [x0,y0,x1,y1] (裁剪图坐标)."""
    iw = max(1, int(img_w))
    ih = max(1, int(img_h))
    if not kps_2d:
        return [0.0, 0.0, float(iw - 1), float(ih - 1)]
    xs: list[float] = []
    ys: list[float] = []
    for i, kp in enumerate(kps_2d):
        vis = kps_vis[i] if i < len(kps_vis) else 2
        if vis == 0:
            continue
        xs.append(float(kp[0]))
        ys.append(float(kp[1]))
    if not xs:
        xs = [float(kp[0]) for kp in kps_2d]
        ys = [float(kp[1]) for kp in kps_2d]
    x0 = max(0.0, min(xs) - margin)
    y0 = max(0.0, min(ys) - margin)
    x1 = min(float(iw - 1), max(xs) + margin)
    y1 = min(float(ih - 1), max(ys) + margin)
    if x1 <= x0:
        x1 = min(float(iw - 1), x0 + 1.0)
    if y1 <= y0:
        y1 = min(float(ih - 1), y0 + 1.0)
    return [x0, y0, x1, y1]


def _is_valid_bbox2d(bb) -> bool:
    return (
        isinstance(bb, (list, tuple))
        and len(bb) == 4
        and all(isinstance(v, (int, float)) for v in bb)
    )


def _pillar_bbox_for_crop(ann: dict, cw: int, ch: int) -> list[float]:
    """装配柱 crop 坐标下的 bbox; 兼容 Blender 写出 null 或旧 annotations."""
    bb = ann.get("bbox_2d")
    if _is_valid_bbox2d(bb):
        return [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]
    kps = ann.get("kps_2d") or []
    vis = ann.get("kps_vis", [2] * len(kps))
    return bbox_2d_from_kps_vis(kps, vis, cw, ch)


def _exchange_bbox_for_crop(ann: dict, cw: int, ch: int) -> list[float]:
    """exchange crop 坐标下的 bbox; 缺省则用 exchange 关键点或退回 pillar bbox."""
    for key in ("exc_bbox_2d", "bbox_2d_exchange"):
        bb = ann.get(key)
        if _is_valid_bbox2d(bb):
            return [float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])]
    exc_kps = ann.get("exc_kps_2d") or []
    if exc_kps:
        return bbox_2d_from_kps_vis(
            exc_kps,
            ann.get("exc_kps_vis", [2] * len(exc_kps)),
            cw,
            ch,
        )
    return _pillar_bbox_for_crop(ann, cw, ch)


# ──────────────────────────────────────────────────────────────────────────────
#  Luma 工具
# ──────────────────────────────────────────────────────────────────────────────


def masked_luma_from_rgba(crop_rgba: np.ndarray) -> float:
    """前景区域归一化 luma 均值 [0,1]."""
    alpha = crop_rgba[..., 3].astype(np.float32) / 255.0
    fg_mask = alpha > ALPHA_THRESH
    if fg_mask.sum() == 0:
        return 0.5
    return (
        float(cv2.cvtColor(crop_rgba[..., :3], cv2.COLOR_RGB2GRAY)[fg_mask].mean())
        / 255.0
    )


def alpha_auto_crop(item: dict) -> dict:
    """根据 alpha > ALPHA_THRESH 区域自动裁剪 RGBA 图像，并同步偏移所有像素坐标。

    修改 item 中的：
      - "rgba"        → 裁剪后的 RGBA
      - "kps_2d"      → 减去左上角偏移
      - "bbox_2d"     → 减去左上角偏移
      - "exc_kps_2d"  → 减去左上角偏移（如存在）
      - "exc_bbox_2d" → 减去左上角偏移（如存在）

    若 alpha 全透明（无前景），则原样返回。
    """
    rgba = item["rgba"]  # (H, W, 4) uint8
    alpha = rgba[..., 3]
    fg = alpha > int(ALPHA_THRESH * 255)

    rows = np.any(fg, axis=1)
    cols = np.any(fg, axis=0)
    if not rows.any():
        return item  # 全透明，不裁

    y0, y1 = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1])) + 1
    x0, x1 = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1])) + 1

    # 裁剪 RGBA
    item["rgba"] = rgba[y0:y1, x0:x1]

    # 偏移关键点坐标（vis=0 的点保持 [0,0] 不动，偏移后仍无意义但不影响合成）
    dx, dy = float(x0), float(y0)
    item["kps_2d"] = [[kp[0] - dx, kp[1] - dy] for kp in item["kps_2d"]]

    # 偏移 bbox
    if item.get("bbox_2d") is not None:
        bx0, by0, bx1, by1 = item["bbox_2d"]
        item["bbox_2d"] = [bx0 - dx, by0 - dy, bx1 - dx, by1 - dy]

    # 偏移 exchange 关键点
    if "exc_kps_2d" in item:
        item["exc_kps_2d"] = [[kp[0] - dx, kp[1] - dy] for kp in item["exc_kps_2d"]]
    if "exc_bbox_2d" in item:
        ex0, ey0, ex1, ey1 = item["exc_bbox_2d"]
        item["exc_bbox_2d"] = [ex0 - dx, ey0 - dy, ex1 - dx, ey1 - dy]

    return item


def luma_align(
    crop_rgb: np.ndarray, alpha: np.ndarray, bg_region: np.ndarray
) -> np.ndarray:
    """残差式 L 通道对齐 (仅 --residual_luma_align 时调用)."""
    fg_mask = alpha > ALPHA_THRESH
    if fg_mask.sum() == 0:
        return crop_rgb
    crop_lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(bg_region, cv2.COLOR_RGB2LAB).astype(np.float32)
    mean_cL = crop_lab[..., 0][fg_mask].mean()
    mean_bL = bg_lab[..., 0].mean()
    if mean_cL < 1e-3:
        return crop_rgb
    crop_lab[..., 0] = np.clip(crop_lab[..., 0] * (mean_bL / mean_cL), 0, 100)
    return cv2.cvtColor(crop_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)


# ──────────────────────────────────────────────────────────────────────────────
#  背景池
# ──────────────────────────────────────────────────────────────────────────────


def build_bg_pool(bg_dir: Path, n_bins: int = 10) -> list:
    """扫描 bg_dir 下图片, 计算并缓存 luma, 返回按 luma 排序的列表."""
    paths = [
        p for p in bg_dir.rglob("*") if p.is_file() and p.suffix in SUPPORTED_IMAGE_EXTS
    ]
    if not paths:
        return []

    cache_path = bg_dir / ".luma_cache.json"
    luma_cache = {}
    if cache_path.exists():
        try:
            luma_cache = json.loads(cache_path.read_text())
        except Exception:
            pass

    lumas, dirty, n_cached = [], False, 0
    t0 = time.time()
    for p in tqdm(paths, desc="build_bg_pool luma", unit="img", dynamic_ncols=True):
        key = str(p)
        if key in luma_cache:
            lumas.append(luma_cache[key])
            n_cached += 1
            continue
        img = cv2.imread(str(p), cv2.IMREAD_REDUCED_COLOR_8)
        val = (
            float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()) / 255.0
            if img is not None
            else 0.5
        )
        luma_cache[key] = val
        lumas.append(val)
        dirty = True

    print(
        f"[build_bg_pool] {n_cached} 缓存 / {len(paths)-n_cached} 新计算 ({time.time()-t0:.1f}s)"
    )
    if dirty:
        try:
            cache_path.write_text(json.dumps(luma_cache))
        except Exception as e:
            print(f"[build_bg_pool] 缓存写入失败: {e}")

    bins = [0] * n_bins
    for l in lumas:
        bins[min(int(l * n_bins), n_bins - 1)] += 1
    print(f"[build_bg_pool] luma分箱[暗→亮]: {' '.join(f'{v:3d}' for v in bins)}")

    return [
        {"path": p, "luma": float(l)}
        for p, l in sorted(zip(paths, lumas), key=lambda x: x[1])
    ]


def select_bg(
    bg_pool: list,
    target: float,
    rng: random.Random,
    thresh: float = 0.08,
    max_expand: int = 6,
) -> Optional[dict]:
    """按亮度匹配背景图, 逐步扩大阈值, 最终 fallback 最近 luma."""
    if not bg_pool:
        return None
    t = thresh
    for _ in range(max_expand + 1):
        cands = [x for x in bg_pool if abs(x["luma"] - target) <= t]
        if cands:
            return rng.choice(cands)
        t *= 1.5
    return min(bg_pool, key=lambda x: abs(x["luma"] - target))


def make_fallback_bg(w: int, h: int, rng: random.Random) -> np.ndarray:
    """随机灰度纯色背景."""
    return np.full((h, w, 3), rng.randint(30, 200), dtype=np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
#  自适应背景尺寸
# ──────────────────────────────────────────────────────────────────────────────


def adaptive_bg_size(
    crops: list,
    rng: random.Random,
    frac_range: tuple = (0.20, 0.50),
    aspect: float = 4 / 3,
    min_wh: tuple = (320, 240),
    max_wh: tuple = (1280, 960),
) -> tuple:
    """让最大 crop 前景面积占背景总面积的 frac_range 随机比例.
    背景宽高比 aspect, 对齐 32px, 并保证能容纳最大 crop.
    """
    max_fg = (
        max(
            float((c["rgba"][:, :, 3].astype(np.float32) / 255.0 > ALPHA_THRESH).sum())
            for c in crops
        )
        if crops
        else 1.0
    )
    if max_fg < 100:
        max_fg = float(max(c["rgba"].shape[0] * c["rgba"].shape[1] for c in crops))

    bg_area = max_fg / rng.uniform(*frac_range)
    bg_w_r = (bg_area * aspect) ** 0.5
    bg_h_r = bg_w_r / aspect
    bg_w = max(min_wh[0], min(max_wh[0], int(round(bg_w_r / 32) * 32)))
    bg_h = max(min_wh[1], min(max_wh[1], int(round(bg_h_r / 32) * 32)))
    bg_w = max(bg_w, max(c["rgba"].shape[1] for c in crops))
    bg_h = max(bg_h, max(c["rgba"].shape[0] for c in crops))
    return bg_w, bg_h


# ──────────────────────────────────────────────────────────────────────────────
#  Mask 重叠率
# ──────────────────────────────────────────────────────────────────────────────


def mask_overlap_ratio(
    mask_a: np.ndarray, off_a: tuple, mask_b: np.ndarray, off_b: tuple
) -> float:
    """inter / min(area_a, area_b) — 比 IoU 更敏感的小目标遮挡指标."""
    ax, ay = off_a
    bx, by = off_b
    ah, aw = mask_a.shape
    bh, bw = mask_b.shape
    ix0, iy0 = max(ax, bx), max(ay, by)
    ix1, iy1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = np.logical_and(
        mask_a[iy0 - ay : iy1 - ay, ix0 - ax : ix1 - ax],
        mask_b[iy0 - by : iy1 - by, ix0 - bx : ix1 - bx],
    ).sum()
    denom = min(mask_a.sum(), mask_b.sum())
    return float(inter) / float(denom) if denom > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  单 pillar 放置 + 合成
# ──────────────────────────────────────────────────────────────────────────────


def place_pillar(
    ann: dict,
    composite: np.ndarray,
    instances: list,
    placed_masks: list,
    rng: random.Random,
    max_attempts: int,
    max_overlap_ratio: float,
    max_kp_delta: int,
    do_luma_align: bool,
) -> bool:
    """尝试将 crop 放置到 composite 上.

    约束:
      1. 与已放置 crop 的 mask 重叠率 ≤ max_overlap_ratio
      2. 新 crop 的 mask 覆盖已放置实例的可见/遮挡 kp 数 ≤ max_kp_delta

    放置成功后:
      - 更新已放置实例的 kps_vis (顶层遮挡底层)
      - 追加新实例到 instances, 新 mask 到 placed_masks

    Returns True 成功 / False 超出尝试次数
    """
    crop_rgba = ann["rgba"]
    ch, cw = crop_rgba.shape[:2]
    bg_h, bg_w = composite.shape[:2]
    alpha_full = (crop_rgba[:, :, 3].astype(np.float32) / 255.0) > ALPHA_THRESH

    for _ in range(max_attempts):
        ox = rng.randint(0, max(0, bg_w - cw))
        oy = rng.randint(0, max(0, bg_h - ch))
        rx = min(ox + cw, bg_w) - ox
        ry = min(oy + ch, bg_h) - oy
        cand = alpha_full[:ry, :rx]

        # 约束 1: mask 重叠率
        if any(
            mask_overlap_ratio(cand, (ox, oy), pm["mask"], pm["offset"])
            > max_overlap_ratio
            for pm in placed_masks
        ):
            continue

        # 约束 2: kp 遮挡增量
        kp_delta = sum(
            1
            for prev in instances
            for kp, vis in zip(prev["kps_2d"], prev["kps_vis"])
            if vis != 0
            and 0 <= int(kp[0]) - ox < rx
            and 0 <= int(kp[1]) - oy < ry
            and cand[int(kp[1]) - oy, int(kp[0]) - ox]
        )
        if kp_delta > max_kp_delta:
            continue

        # ── 合成 ──
        crop_rgb = crop_rgba[:ry, :rx, :3]
        alpha = crop_rgba[:ry, :rx, 3].astype(np.float32) / 255.0
        bg_region = composite[oy : oy + ry, ox : ox + rx].copy()
        if do_luma_align:
            crop_rgb = luma_align(crop_rgb, alpha, bg_region)
        a3 = alpha[:, :, np.newaxis]
        composite[oy : oy + ry, ox : ox + rx] = (
            crop_rgb * a3 + bg_region * (1 - a3)
        ).astype(np.uint8)

        # ── 遮挡传播: 新 pillar (顶层) 遮挡已放置实例 (底层) 的 kp ──
        new_mask = alpha > ALPHA_THRESH
        for prev in instances:
            # pillar kps
            for ki, (kp, vis) in enumerate(zip(prev["kps_2d"], prev["kps_vis"])):
                if vis == 0:
                    continue
                lx, ly = int(kp[0]) - ox, int(kp[1]) - oy
                if (
                    0 <= lx < new_mask.shape[1]
                    and 0 <= ly < new_mask.shape[0]
                    and new_mask[ly, lx]
                ):
                    prev["kps_vis"][ki] = min(vis, 1)
            # exchange kps
            for ki, (kp, vis) in enumerate(
                zip(prev.get("exc_kps_2d", []), prev.get("exc_kps_vis", []))
            ):
                if vis == 0:
                    continue
                lx, ly = int(kp[0]) - ox, int(kp[1]) - oy
                if (
                    0 <= lx < new_mask.shape[1]
                    and 0 <= ly < new_mask.shape[0]
                    and new_mask[ly, lx]
                ):
                    prev["exc_kps_vis"][ki] = min(vis, 1)

        # ── 记录新实例 ──
        bx0, by0, bx1, by1 = _pillar_bbox_for_crop(ann, cw, ch)
        inst = {
            "kps_2d": [[kp[0] + ox, kp[1] + oy] for kp in ann["kps_2d"]],
            "kps_vis": list(ann.get("kps_vis", [2] * len(ann["kps_2d"]))),
            "bbox": [bx0 + ox, by0 + oy, bx1 + ox, by1 + oy],
            "material_luma": ann["material_luma"],
            "file_name": ann["file_name"],
        }
        # exchange 数据 (如果有)
        if "exc_kps_2d" in ann:
            inst["exc_kps_2d"] = [[kp[0] + ox, kp[1] + oy] for kp in ann["exc_kps_2d"]]
            inst["exc_kps_vis"] = list(
                ann.get("exc_kps_vis", [2] * len(ann["exc_kps_2d"]))
            )
            ebx0, eby0, ebx1, eby1 = _exchange_bbox_for_crop(ann, cw, ch)
            inst["exc_bbox"] = [ebx0 + ox, eby0 + oy, ebx1 + ox, eby1 + oy]
        instances.append(inst)
        placed_masks.append({"mask": new_mask, "offset": (ox, oy)})
        return True

    return False


# ──────────────────────────────────────────────────────────────────────────────
#  标注 + Crop 预加载助手
# ──────────────────────────────────────────────────────────────────────────────


def _load_blender_dir(
    blender_dir: Path,
    kp_names_ref: Optional[list] = None,
    exc_kp_names_ref: Optional[list] = None,
    auto_crop_alpha: bool = False,
) -> tuple:
    """加载单个 blender_dir 的标注和 RGBA crop.

    返回: (crop_pool, kp_names, exc_kp_names, has_exchange)
    若 kp_names_ref 非 None, 则校验 kp schema 一致性 (仅 warn, 不 abort).
    """
    ann_path = blender_dir / "annotations.json"
    if not ann_path.exists():
        print(f"[WARN] 找不到 {ann_path}, 跳过")
        return [], kp_names_ref or [], exc_kp_names_ref or [], bool(exc_kp_names_ref)

    data = json.loads(ann_path.read_text(encoding="utf-8"))
    kp_names = data.get("keypoint_names", [])
    n_kps = len(kp_names)

    # 检测 exchange target
    targets_meta = data.get("targets", [])
    exc_kp_names: list = []
    has_exchange = False
    if isinstance(targets_meta, list):
        for tgt_meta in targets_meta:
            tgt_kp = tgt_meta.get("keypoint_names", [])
            if tgt_kp and tgt_kp != kp_names:
                exc_kp_names = tgt_kp
                has_exchange = True
                break
    elif isinstance(targets_meta, dict):
        for _, tgt_meta in targets_meta.items():
            tgt_kp = tgt_meta.get("keypoint_names", [])
            if tgt_kp and tgt_kp != kp_names:
                exc_kp_names = tgt_kp
                has_exchange = True
                break
    if not has_exchange and data["images"]:
        for _, tgt_data in data["images"][0].get("targets", {}).items():
            sec_names = tgt_data.get("keypoint_names", [])
            if sec_names and sec_names != kp_names:
                exc_kp_names = sec_names
                has_exchange = True
                break

    # schema 一致性检查
    if kp_names_ref is not None and kp_names != kp_names_ref:
        print(f"[WARN] {blender_dir.name}: pillar kp_names 与参考不一致, 强制使用参考")
        kp_names = kp_names_ref
    if (
        exc_kp_names_ref is not None
        and exc_kp_names
        and exc_kp_names != exc_kp_names_ref
    ):
        print(f"[WARN] {blender_dir.name}: exc_kp_names 与参考不一致, 强制使用参考")
        exc_kp_names = exc_kp_names_ref

    print(
        f"[load] {blender_dir.name}: {len(data['images'])} crop, "
        f"pillar {n_kps} kp"
        + (f", exchange {len(exc_kp_names)} kp" if has_exchange else "")
    )

    crop_pool, n_skip = [], 0
    t0 = time.time()
    for ann in tqdm(
        data["images"],
        desc=f"  {blender_dir.name}",
        unit="img",
        dynamic_ncols=True,
        leave=False,
    ):
        img_path = blender_dir / "images" / ann["file_name"]
        crop_rgba = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if crop_rgba is None or crop_rgba.ndim < 3:
            tqdm.write(f"  [WARN] 跳过: {img_path}")
            n_skip += 1
            continue
        if crop_rgba.shape[2] == 4:
            crop_rgba = cv2.cvtColor(crop_rgba, cv2.COLOR_BGRA2RGBA)
        else:
            alpha_ch = np.full(crop_rgba.shape[:2], 255, dtype=np.uint8)
            crop_rgba = np.dstack(
                [cv2.cvtColor(crop_rgba, cv2.COLOR_BGR2RGB), alpha_ch]
            )

        kps = [[float(kp[0]), float(kp[1])] for kp in ann["keypoints_2d"]]
        kps_vis = [int(kp[2]) if len(kp) >= 3 else 2 for kp in ann["keypoints_2d"]]
        bh, bw = crop_rgba.shape[:2]
        pillar_bb = _pillar_bbox_for_crop(
            {"bbox_2d": ann.get("bbox_2d"), "kps_2d": kps, "kps_vis": kps_vis}, bw, bh
        )

        item = {
            "rgba": crop_rgba,
            "kps_2d": kps,
            "kps_vis": kps_vis,
            "bbox_2d": pillar_bb,
            "file_name": ann["file_name"],
            "material_luma": masked_luma_from_rgba(crop_rgba),
        }
        if _is_valid_bbox2d(ann.get("bbox_2d_exchange")):
            item["bbox_2d_exchange"] = [float(v) for v in ann["bbox_2d_exchange"]]

        if has_exchange:
            exc_found = False
            for _, tgt_data in ann.get("targets", {}).items():
                sec_names = tgt_data.get("keypoint_names", [])
                if sec_names and sec_names != kp_names:
                    exc_raw = tgt_data.get("keypoints_2d", [])
                    item["exc_kps_2d"] = [
                        [float(kp[0]), float(kp[1])] for kp in exc_raw
                    ]
                    item["exc_kps_vis"] = [
                        int(kp[2]) if len(kp) >= 3 else 2 for kp in exc_raw
                    ]
                    item["exc_bbox_2d"] = tgt_data.get(
                        "bbox_2d", ann.get("bbox_2d_exchange", ann.get("bbox_2d"))
                    )
                    exc_found = True
                    break
            if not exc_found:
                item["exc_kps_2d"] = [[0.0, 0.0]] * len(exc_kp_names)
                item["exc_kps_vis"] = [0] * len(exc_kp_names)
                item["exc_bbox_2d"] = pillar_bb
            item["exc_bbox_2d"] = _exchange_bbox_for_crop(item, bw, bh)

        if auto_crop_alpha:
            item = alpha_auto_crop(item)

        crop_pool.append(item)

    print(f"  \u2192 {len(crop_pool)} \u52a0\u8f7d (\u8df3\u8fc7 {n_skip}), \u8017\u65f6 {time.time()-t0:.1f}s")
    return crop_pool, kp_names, exc_kp_names, has_exchange


# ──────────────────────────────────────────────────────────────────────────────
#  主流程
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Blender RGBA → YOLO Pose 数据集合成")
    parser.add_argument(
        "--blender_dir",
        default=None,
        help="单个 Blender 输出目录 (与 --blender_dirs 二选一或混用)",
    )
    parser.add_argument(
        "--blender_dirs",
        nargs="+",
        default=None,
        help="多个 Blender 输出目录, 合并到同一数据集",
    )
    parser.add_argument(
        "--bg_dir", default=None, help="背景图目录; 留空则用随机纯色背景"
    )
    parser.add_argument("--output_dir", default="./pillar_dataset")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument(
        "--bg_size",
        default="auto",
        help="'auto' (前景占fg_frac, 4:3) 或 WxH 固定尺寸",
    )
    parser.add_argument(
        "--fg_frac",
        type=float,
        nargs=2,
        default=[0.20, 0.50],
        metavar=("MIN", "MAX"),
        help="auto模式下最大前景面积占背景面积的随机范围, 默认 0.20 0.50; "
        "近距大视野场景可设为 0.50 0.90",
    )
    parser.add_argument("--min_pillars", type=int, default=1)
    parser.add_argument("--max_pillars", type=int, default=2)
    parser.add_argument(
        "--n_output", type=int, default=None, help="输出帧数 (默认 = Blender crop 数)"
    )
    parser.add_argument("--placement_attempts", type=int, default=50)
    parser.add_argument("--max_mask_overlap_ratio", type=float, default=0)
    parser.add_argument(
        "--max_kp_occlusion_delta",
        type=int,
        default=0,
        help="新 pillar 允许新增遮挡已放置实例 kp 数上限 (默认 2)",
    )
    parser.add_argument("--bg_luma_thresh", type=float, default=0.08)
    parser.add_argument("--residual_luma_align", action="store_true")
    parser.add_argument(
        "--auto_crop_alpha",
        action="store_true",
        help="自动将每张 Blender 渲染图 crop 到 alpha>0 的有效区域，去除透明边框",
    )
    parser.add_argument(
        "--vis_output", default=None, help="指定目录则额外输出带 kp 可视化图 (调试用)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 4),
        help="线程池并发数 (默认 min(8, cpu_count))",
    )
    args = parser.parse_args()

    assert args.min_pillars >= 1
    assert args.max_pillars >= args.min_pillars

    # ── 收集所有 blender 目录 ──
    all_blender_dirs: list[Path] = []
    if args.blender_dir:
        all_blender_dirs.append(Path(args.blender_dir))
    if args.blender_dirs:
        all_blender_dirs.extend(Path(p) for p in args.blender_dirs)
    if not all_blender_dirs:
        sys.exit("[ERROR] 至少指定 --blender_dir 或 --blender_dirs 之一")
    # 去重同时保持顺序
    seen: set = set()
    all_blender_dirs = [
        d for d in all_blender_dirs if not (str(d) in seen or seen.add(str(d)))
    ]  # type: ignore[func-returns-value]

    multi_dir = len(all_blender_dirs) > 1
    print(
        f"[compose] blender 目录 ({len(all_blender_dirs)} 个): "
        + ", ".join(d.name for d in all_blender_dirs)
    )

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # ── 输出目录: 单目录加 _compose 后缀, 多目录直接用 output_dir ──
    if multi_dir:
        output_dir = Path(args.output_dir)
        vis_dir = Path(args.vis_output) if args.vis_output else None
    else:
        output_dir = Path(args.output_dir) / (all_blender_dirs[0].name + "_compose")
        vis_dir = (
            Path(args.vis_output) / (all_blender_dirs[0].name + "_compose")
            if args.vis_output
            else None
        )

    use_auto_bg = args.bg_size.lower() == "auto"
    if not use_auto_bg:
        bg_w_fixed, bg_h_fixed = (int(v) for v in args.bg_size.split("x"))

    # ── 逐目录加载标注 + crop ──
    crop_pool: list = []
    kp_names: list = []
    exc_kp_names: list = []
    has_exchange: bool = False
    t0_total = time.time()
    for bdir in all_blender_dirs:
        pool_i, kp_i, exc_i, has_exc_i = _load_blender_dir(
            bdir,
            kp_names_ref=kp_names if kp_names else None,
            exc_kp_names_ref=exc_kp_names if exc_kp_names else None,
            auto_crop_alpha=args.auto_crop_alpha,
        )
        if not kp_names and kp_i:
            kp_names = kp_i
        if not exc_kp_names and exc_i:
            exc_kp_names = exc_i
        if has_exc_i:
            has_exchange = True
        crop_pool.extend(pool_i)

    n_kps = len(kp_names)
    lumas = [c["material_luma"] for c in crop_pool]
    print(
        f"[compose] 合并 crop 池: {len(crop_pool)} 张, 耗时 {time.time()-t0_total:.1f}s"
    )
    if lumas:
        print(
            f"[compose] luma: {min(lumas):.3f}~{max(lumas):.3f}, 均值 {np.mean(lumas):.3f}"
        )
    if not crop_pool:
        sys.exit("[ERROR] crop 池为空")
    print(f"[compose] pillar: {n_kps} kp {kp_names}")
    if has_exchange:
        print(f"[compose] exchange: {len(exc_kp_names)} kp {exc_kp_names}")
    print(f"[compose] output_dir: {output_dir}")

    # ── 背景池 ──
    bg_pool = build_bg_pool(Path(args.bg_dir)) if args.bg_dir else []
    if bg_pool:
        print(f"[compose] 背景池: {len(bg_pool)} 张")
    else:
        print("[compose] 无背景图 → 随机纯色背景")

    # ── 输出目录 ──
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)
        print(f"[compose] vis_output → {vis_dir}")

    n_output = args.n_output or len(crop_pool)
    n_val = max(1, int(n_output * args.val_split))
    val_set = set(range(n_val))
    bg_str = "auto(4:3,20%~50%)" if use_auto_bg else f"{bg_w_fixed}x{bg_h_fixed}"
    print(
        f"[compose] 输出 {n_output} 帧 (train={n_output-n_val}, val={n_val}), "
        f"pillar/image={args.min_pillars}~{args.max_pillars}, bg={bg_str}, "
        f"attempts={args.placement_attempts}, "
        f"max_overlap={args.max_mask_overlap_ratio}, "
        f"max_kp_delta={args.max_kp_occlusion_delta}, "
        f"workers={args.workers}"
    )

    # ── 预生成每帧参数 (用主 rng, 保证 crop 选取可复现) ──
    frame_params = []
    for frame_idx in range(n_output):
        selected = [
            rng.choice(crop_pool)
            for _ in range(rng.randint(args.min_pillars, args.max_pillars))
        ]
        frame_params.append(
            {
                "frame_idx": frame_idx,
                "selected": selected,
                "split": "val" if frame_idx in val_set else "train",
                "worker_seed": args.seed + frame_idx,  # 每帧独立 rng → 线程安全
            }
        )

    # ── 单帧处理函数 (在线程池中执行) ──
    _skipped_lock = threading.Lock()
    _skipped_total = [0]  # 用列表使闭包可写

    def _process_frame(fp: dict) -> None:
        frame_idx = fp["frame_idx"]
        selected = fp["selected"]
        split = fp["split"]
        frng = random.Random(fp["worker_seed"])

        target_luma = float(np.mean([c["material_luma"] for c in selected]))

        if use_auto_bg:
            bg_w, bg_h = adaptive_bg_size(
                selected, frng, frac_range=tuple(args.fg_frac)
            )
        else:
            bg_w, bg_h = bg_w_fixed, bg_h_fixed

        if bg_pool:
            item = select_bg(bg_pool, target_luma, frng, args.bg_luma_thresh)
            bg_img = cv2.imread(str(item["path"]))
            if bg_img is None:
                bg_img = make_fallback_bg(bg_w, bg_h, frng)
            else:
                bg_img = cv2.cvtColor(
                    cv2.resize(bg_img, (bg_w, bg_h), interpolation=cv2.INTER_AREA),
                    cv2.COLOR_BGR2RGB,
                )
        else:
            bg_img = make_fallback_bg(bg_w, bg_h, frng)

        composite = bg_img.copy()
        placed_masks = []
        instances = []
        local_skip = 0

        for ann in selected:
            if not place_pillar(
                ann,
                composite,
                instances,
                placed_masks,
                frng,
                max_attempts=args.placement_attempts,
                max_overlap_ratio=args.max_mask_overlap_ratio,
                max_kp_delta=args.max_kp_occlusion_delta,
                do_luma_align=args.residual_luma_align,
            ):
                local_skip += 1

        if local_skip:
            with _skipped_lock:
                _skipped_total[0] += local_skip

        stem = f"frame_{frame_idx:06d}"
        img_h, img_w = composite.shape[:2]

        cv2.imwrite(
            str(output_dir / "images" / split / f"{stem}.jpg"),
            cv2.cvtColor(composite, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        )
        write_yolo_label(
            output_dir / "labels" / split / f"{stem}.txt", instances, img_w, img_h
        )

        if vis_dir is not None:
            cv2.imwrite(
                str(vis_dir / f"{stem}.jpg"),
                draw_kp_vis(
                    cv2.cvtColor(composite, cv2.COLOR_RGB2BGR),
                    instances,
                    kp_names,
                    exc_kp_names,
                ),
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )

    # ── 线程池并发执行 ──
    with tqdm(total=n_output, desc="compose", unit="frame", dynamic_ncols=True) as pbar:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futs = {
                executor.submit(_process_frame, fp): fp["frame_idx"]
                for fp in frame_params
            }
            for fut in as_completed(futs):
                fut.result()  # 传播子线程异常
                pbar.update(1)

    total_skipped = _skipped_total[0]

    # ── 汇总 ──
    n_train = len(list((output_dir / "images" / "train").glob("*.jpg")))
    n_val_c = len(list((output_dir / "images" / "val").glob("*.jpg")))
    print(f"\n[compose] 完成! 输出: {output_dir}")
    print(f"  train={n_train}, val={n_val_c}, skipped pillars={total_skipped}")

    # ── dataset.yaml ──
    all_kp_names = kp_names + exc_kp_names
    n_total_kps = len(all_kp_names)
    yaml_path = output_dir / "dataset.yaml"

    # flip_idx: pillar 角点左右对称 (TL↔TR, BL↔BR, ring→ring)
    # exchange 光带左右对称 (light_BR↔light_BL, light_TR↔light_TL, shell_R↔shell_L, shell_M→shell_M)
    pil_flip = [1, 0, 3, 2, 4]
    exc_flip = [
        6,
        5,
        4,
        3,
        2,
        1,
        0,
    ]  # light_BR↔light_BL, light_TR↔light_TL, shell_R↔shell_L, shell_M↔shell_M
    flip_idx = pil_flip + [n_kps + i for i in exc_flip]

    n_classes = 2 if has_exchange else 1
    names_block = "names:\n  0: pillar\n"
    if has_exchange:
        names_block += "  1: exchange\n"

    yaml_path.write_text(
        f"# YOLO Pose Dataset — {n_classes} 类, {n_total_kps} 关键点\n"
        f"# pillar: {n_kps} kp ({kp_names})\n"
        f"# exchange: {len(exc_kp_names)} kp ({exc_kp_names})\n"
        f"# 生成自 compose_dataset.py  背景: {args.bg_dir or '随机纯色'}\n\n"
        f"path: {output_dir.resolve()}\ntrain: images/train\nval:   images/val\n\n"
        f"kpt_shape: [{n_total_kps}, 3]\nflip_idx: {flip_idx}\n\n"
        + names_block
        + f"\nkpt_names:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(all_kp_names)),
        encoding="utf-8",
    )

    # ── build_args.json ──
    build_args = {
        **vars(args),
        "all_blender_dirs": [str(d) for d in all_blender_dirs],
        "generated_at": datetime.datetime.now().isoformat(),
    }
    (output_dir / "build_args.json").write_text(
        json.dumps(build_args, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ── 复制 material config (从第一个 blender_dir 取) ──
    src = all_blender_dirs[0] / "config.yaml"
    if src.exists():
        shutil.copy2(src, output_dir / "material_config.yaml")

    local_yaml = Path(__file__).resolve().parent.parent / "config" / "pillar_pose.yaml"
    shutil.copy2(yaml_path, local_yaml)
    print(f"  dataset.yaml → {yaml_path}  (同步 {local_yaml})")


if __name__ == "__main__":
    main()
