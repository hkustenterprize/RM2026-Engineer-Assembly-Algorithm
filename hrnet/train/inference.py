#!/usr/bin/env python3
"""
Two-stage inference: Ultralytics YOLO Detect (bbox) → MMPose HRNet keypoints.

- YOLO: Detect model; filter by ``bbox_class_id`` (default 0 for 5-kp, 1 for 12-kp).
- HRNet: top-down; bbox is passed into MMPose, which applies the same crop+warp as ``hrnet.py``
  val/test pipeline (``TopdownAffine``), then predicts keypoints in **original image coordinates**.

Dependencies: ``torch``, ``numpy``, ``opencv-python``, ``ultralytics``, ``mmpose``, ``mmengine``.

PYTHONPATH=/data/datasets/zguobd/mmpose:$PYTHONPATH \
/data/datasets/zguobd/miniconda3/envs/mm/bin/python \
/data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/train/inference.py \
  --yolo-weights /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/yolopose/runs/detect/.../weights/best.pt \
  --hrnet-config /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/model_configs/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py \
  --hrnet-checkpoint /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/runs/hrnet_v7.0/best_coco_AP_epoch_20.pth \
  --images /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/data/eval/eval_samples/image*.png \
  --device cuda:0

Example (CLI)::

    python pillar_hrnet_infer.py \\
        --yolo-weights /path/to/best.pt \\
        --hrnet-config /path/to/hrnet.py \\
        --hrnet-checkpoint /path/to/best_coco_AP_epoch_xx.pth \\
        --images img0.jpg img1.jpg \\
        --device cuda:0

Example (Python API)::

    pipe = PillarHrnetPipeline(
        yolo_weights=\".../best.pt\",
        hrnet_config=\".../hrnet.py\",
        hrnet_checkpoint=\".../best.pth\",
        device=\"cuda:0\",
    )
    out = pipe.predict_image(\"scene.jpg\")
    batch = pipe.predict_images([\"a.jpg\", \"b.jpg\"])
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import numpy as np

# ---------------------------------------------------------------------------
# Output types (stable public API)
# ---------------------------------------------------------------------------


@dataclass
class PillarInstance:
    """One pillar after YOLO bbox + HRNet keypoints."""

    bbox_xyxy: np.ndarray  # (4,) float32, xyxy in original image pixels
    yolo_score: float  # pillar box confidence from YOLO
    keypoints_xy: np.ndarray  # (K, 2) float32, image space — heatmap DARK decode
    keypoint_scores: np.ndarray  # (K,) float32, HRNet per-joint confidence
    keypoint_visible_scores: Optional[np.ndarray] = None  # (K,) float32, vis/in-frame score in [0, 1]
    debug_info: Any = field(default=None, repr=False)  # optional diagnostics


@dataclass
class ImagePoseResult:
    """Result for a single image (file path or in-memory array)."""

    image: Union[str, Path, np.ndarray]
    image_shape_hw: tuple[int, int]  # (H, W)
    pillars: list[PillarInstance] = field(default_factory=list)
    keypoint_names: Optional[list[str]] = None
    skeleton: Optional[list[tuple[int, int]]] = None
    keypoint_colors: Optional[list[tuple[int, int, int]]] = None


@dataclass
class BatchPoseResult:
    """Batch wrapper: one ``ImagePoseResult`` per input image (same order)."""

    images: list[ImagePoseResult] = field(default_factory=list)

    def __iter__(self):
        return iter(self.images)

    def __len__(self) -> int:
        return len(self.images)


# ---------------------------------------------------------------------------
# Internal geometry helpers
# ---------------------------------------------------------------------------


@dataclass
class _PeakCandidate:
    """One candidate peak in original-image coordinates."""

    xy: np.ndarray  # (2,) float32, image space
    score: float  # heatmap value / decoded confidence


@dataclass
class _CornerComboScore:
    """One feasible corner assignment with cost breakdown."""

    total_cost: float
    channel_cost: float
    dlt_err: float
    chiral_cost: float
    anchor_cost: float
    candidate_indices: tuple[int, int, int, int]
    corners_xy: np.ndarray  # (4, 2)


@dataclass
class _CornerRefineDebug:
    """Debug bundle for one pillar corner-refine pass."""

    status: str
    norm_size: float
    combo_min_dist_px: float
    decoded_keypoints_xy: np.ndarray
    decoded_keypoint_scores: np.ndarray
    refined_keypoints_xy: np.ndarray
    refined_keypoint_scores: np.ndarray
    corner_candidates: list[list[_PeakCandidate]] = field(default_factory=list)
    base_combo: Optional[_CornerComboScore] = None
    best_combo: Optional[_CornerComboScore] = None
    second_combo: Optional[_CornerComboScore] = None
    chosen_candidate_indices: Optional[tuple[int, int, int, int]] = None
    max_corner_shift_px: float = 0.0


_PTS3D_H = np.array(
    [
        [-0.040, 0.040, 0.012, 1.0],
        [0.040, 0.040, 0.012, 1.0],
        [-0.040, -0.040, 0.012, 1.0],
        [0.040, -0.040, 0.012, 1.0],
        [0.000, 0.000, 0.112, 1.0],
    ],
    dtype=np.float64,
)


def _sample_heatmap_value(heatmap: np.ndarray, xy: np.ndarray) -> float:
    """Bilinear sample on one heatmap in image coordinates."""
    h, w = heatmap.shape[:2]
    if h <= 0 or w <= 0:
        return 0.0

    x = float(np.clip(xy[0], 0.0, max(w - 1, 0)))
    y = float(np.clip(xy[1], 0.0, max(h - 1, 0)))
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    dx = x - x0
    dy = y - y0

    v00 = float(heatmap[y0, x0])
    v01 = float(heatmap[y0, x1])
    v10 = float(heatmap[y1, x0])
    v11 = float(heatmap[y1, x1])
    top = (1.0 - dx) * v00 + dx * v01
    bot = (1.0 - dx) * v10 + dx * v11
    return (1.0 - dy) * top + dy * bot


def _append_unique_candidate(
    candidates: list[_PeakCandidate],
    xy: np.ndarray,
    score: float,
    min_dist_px: float,
) -> None:
    """Append a candidate unless it is near-duplicate of an existing one."""
    xy = np.asarray(xy, dtype=np.float32).reshape(2)
    for cand in candidates:
        if np.linalg.norm(cand.xy - xy) < min_dist_px:
            return
    candidates.append(_PeakCandidate(xy=xy, score=float(score)))


def _extract_peak_candidates(
    heatmap: np.ndarray,
    decoded_xy: Optional[np.ndarray],
    decoded_score: Optional[float],
    topk: int,
    nms_radius_px: int,
    rel_threshold: float,
    abs_threshold: float,
    dedup_dist_px: float,
) -> list[_PeakCandidate]:
    """
    Extract a small candidate set from one reverted heatmap.

    The decoded HRNet point is always inserted first as a safe fallback, then
    local maxima are added with simple heatmap-NMS.
    """
    import cv2

    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.ndim != 2 or heatmap.size == 0:
        return []

    candidates: list[_PeakCandidate] = []
    if decoded_xy is not None and np.isfinite(decoded_xy).all():
        score = (
            float(decoded_score)
            if decoded_score is not None
            else _sample_heatmap_value(heatmap, decoded_xy)
        )
        _append_unique_candidate(candidates, decoded_xy, score, dedup_dist_px)

    radius = max(int(nms_radius_px), 1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    hm_smooth = cv2.GaussianBlur(heatmap, (0, 0), 0.8)
    max_val = float(np.max(hm_smooth))
    if max_val <= 0.0 or topk <= 0:
        return candidates[:topk]
    hm_dilated = cv2.dilate(hm_smooth, kernel)
    threshold = max(float(abs_threshold), float(rel_threshold) * max_val)
    peak_mask = (hm_smooth >= hm_dilated - 1e-6) & (hm_smooth >= threshold)
    ys, xs = np.where(peak_mask)

    if len(xs) == 0:
        y, x = np.unravel_index(int(np.argmax(hm_smooth)), hm_smooth.shape)
        ys = np.array([y], dtype=np.int32)
        xs = np.array([x], dtype=np.int32)

    peak_scores = hm_smooth[ys, xs]
    order = np.argsort(-peak_scores)
    for idx in order:
        xy = np.array([float(xs[idx]), float(ys[idx])], dtype=np.float32)
        _append_unique_candidate(
            candidates,
            xy,
            float(peak_scores[idx]),
            dedup_dist_px,
        )
        if len(candidates) >= topk:
            break

    return candidates[:topk]


def _min_pairwise_distance(points_xy: np.ndarray) -> float:
    """Minimum Euclidean distance across a small set of 2-D points."""
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return np.inf

    best = np.inf
    for i in range(pts.shape[0]):
        for j in range(i + 1, pts.shape[0]):
            best = min(best, float(np.linalg.norm(pts[i] - pts[j])))
    return best


def _dlt_reprojection_error(coords_xy: np.ndarray, input_size: float) -> float:
    """
    5-point 3D→2D DLT reprojection error, mirroring `GeometricConsistencyLoss`.
    """
    coords_xy = np.asarray(coords_xy, dtype=np.float64)
    if coords_xy.shape != (5, 2) or not np.isfinite(coords_xy).all():
        return np.inf

    input_size = max(float(input_size), 1.0)
    uv = coords_xy / input_size
    u = uv[:, 0]
    v = uv[:, 1]
    X = _PTS3D_H
    z = np.zeros_like(u)

    row1 = np.stack(
        [
            -X[:, 0],
            -X[:, 1],
            -X[:, 2],
            -X[:, 3],
            z,
            z,
            z,
            z,
            u * X[:, 0],
            u * X[:, 1],
            u * X[:, 2],
            u * X[:, 3],
        ],
        axis=-1,
    )
    row2 = np.stack(
        [
            z,
            z,
            z,
            z,
            -X[:, 0],
            -X[:, 1],
            -X[:, 2],
            -X[:, 3],
            v * X[:, 0],
            v * X[:, 1],
            v * X[:, 2],
            v * X[:, 3],
        ],
        axis=-1,
    )
    A = np.stack([row1, row2], axis=1).reshape(10, 12)

    try:
        _, _, vh = np.linalg.svd(A, full_matrices=True)
    except np.linalg.LinAlgError:
        return np.inf

    p = vh[-1]
    p_norm = np.linalg.norm(p)
    if p_norm < 1e-8:
        return np.inf
    P = (p / p_norm).reshape(3, 4)

    proj = P @ X.T  # (3, 5)
    if proj[2].mean() < 0.0:
        proj = -proj

    w = np.clip(np.abs(proj[2]), 1e-6, None)
    uv_proj = (proj[:2] / w).T
    err = np.square(uv_proj - uv).sum(axis=-1)
    err = np.clip(err, 0.0, 100.0)
    return float(np.mean(err))


def _chirality_penalty(corners_xy: np.ndarray, margin: float = 0.0) -> float:
    """Penalty when TL→TR→BR→BL winding becomes invalid."""
    corners_xy = np.asarray(corners_xy, dtype=np.float64)
    if corners_xy.shape != (4, 2) or not np.isfinite(corners_xy).all():
        return np.inf

    tl, tr, bl, br = corners_xy
    d1 = br - tl
    d2 = bl - tr
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    denom = max(float(np.linalg.norm(d1) * np.linalg.norm(d2)), 1e-6)
    sin_theta = cross / denom
    return max(float(margin) - float(sin_theta), 0.0)


def _copy_peak_candidates(
    candidates: list[list[_PeakCandidate]],
) -> list[list[_PeakCandidate]]:
    """Deep-copy candidate lists for debug snapshots."""
    out: list[list[_PeakCandidate]] = []
    for per_joint in candidates:
        copied = []
        for cand in per_joint:
            copied.append(
                _PeakCandidate(xy=np.asarray(cand.xy, dtype=np.float32).copy(), score=float(cand.score))
            )
        out.append(copied)
    return out


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


_DEFAULT_KPT_NAMES = ["TL", "TR", "BL", "BR", "ring"]
_DEFAULT_SKELETON = [(0, 1), (1, 3), (3, 2), (2, 0)]
_DEFAULT_KPT_COLORS = [
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
]


def _rgb_to_bgr(color: Any) -> tuple[int, int, int]:
    """Convert RGB-style metainfo color to OpenCV BGR tuple."""
    if color is None:
        return (255, 255, 255)
    vals = list(np.asarray(color).reshape(-1)[:3].astype(int))
    if len(vals) < 3:
        vals = (vals + [255, 255, 255])[:3]
    return (int(vals[2]), int(vals[1]), int(vals[0]))


def _fallback_kpt_color(idx: int) -> tuple[int, int, int]:
    return _DEFAULT_KPT_COLORS[idx % len(_DEFAULT_KPT_COLORS)]


def _shorten_kpt_label(name: str) -> str:
    """Shorten verbose keypoint names for cleaner overlays."""
    label = str(name).strip()
    lower = label.lower()
    if lower == "ring":
        return "RG"
    if lower.startswith("light_"):
        return "L" + label.split("_", 1)[1].upper()
    if lower.startswith("shell_"):
        return "S" + label.split("_", 1)[1].upper()
    return label.upper()


def _draw_outlined_text(
    canvas_bgr: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    """Draw readable text on complex backgrounds."""
    import cv2

    cv2.putText(
        canvas_bgr,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        max(3, thickness + 2),
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas_bgr,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _resolve_vis_spec(
    n_kpts: int,
    keypoint_names: Optional[Sequence[str]] = None,
    skeleton: Optional[Sequence[Sequence[int]]] = None,
    keypoint_colors: Optional[Sequence[Sequence[int]]] = None,
) -> tuple[list[str], list[tuple[int, int]], list[tuple[int, int, int]]]:
    """Resolve dynamic visualization metadata with sane fallbacks."""
    if keypoint_names is None:
        names = list(_DEFAULT_KPT_NAMES[: min(n_kpts, len(_DEFAULT_KPT_NAMES))])
        if len(names) < n_kpts:
            names.extend([f"kp{i}" for i in range(len(names), n_kpts)])
    else:
        names = [str(v) for v in keypoint_names[:n_kpts]]
        if len(names) < n_kpts:
            names.extend([f"kp{i}" for i in range(len(names), n_kpts)])

    if skeleton is None:
        links = [link for link in _DEFAULT_SKELETON if max(link) < n_kpts]
    else:
        links = []
        for link in skeleton:
            if len(link) < 2:
                continue
            a, b = int(link[0]), int(link[1])
            if a < n_kpts and b < n_kpts:
                links.append((a, b))

    if keypoint_colors is None:
        colors = [_fallback_kpt_color(i) for i in range(n_kpts)]
    else:
        colors = [_rgb_to_bgr(c) for c in keypoint_colors[:n_kpts]]
        if len(colors) < n_kpts:
            colors.extend(_fallback_kpt_color(i) for i in range(len(colors), n_kpts))

    return names, links, colors


def _result_vis_spec(
    result: Optional[ImagePoseResult],
    n_kpts: int,
) -> tuple[list[str], list[tuple[int, int]], list[tuple[int, int, int]]]:
    """Resolve visualization metadata from ImagePoseResult."""
    if result is None:
        return _resolve_vis_spec(n_kpts=n_kpts)
    return _resolve_vis_spec(
        n_kpts=n_kpts,
        keypoint_names=result.keypoint_names,
        skeleton=result.skeleton,
        keypoint_colors=result.keypoint_colors,
    )


def _draw_single_pillar_overlay(
    canvas_bgr: np.ndarray,
    bbox_xyxy: np.ndarray,
    keypoints_xy: np.ndarray,
    keypoint_names: Sequence[str],
    keypoint_colors: Sequence[tuple[int, int, int]],
    box_color: tuple[int, int, int] = (80, 220, 80),
) -> None:
    """Draw one pillar's bbox + keypoints onto an existing canvas."""
    import cv2

    x0, y0, x1, y1 = np.round(bbox_xyxy).astype(int)
    box_thickness = 10
    point_radius = 16
    label_font_scale = 0.85
    label_thickness = 2
    cv2.rectangle(
        canvas_bgr,
        (x0, y0),
        (x1, y1),
        box_color,
        box_thickness,
        cv2.LINE_AA,
    )

    kpts = np.round(keypoints_xy).astype(int)
    for k, (x, y) in enumerate(kpts):
        color = keypoint_colors[k % len(keypoint_colors)]
        raw_label = keypoint_names[k] if k < len(keypoint_names) else f"kp{k}"
        label = _shorten_kpt_label(raw_label)
        cv2.circle(canvas_bgr, (x, y), point_radius, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas_bgr, (x, y), point_radius - 4, color, -1, cv2.LINE_AA)
        _draw_outlined_text(
            canvas_bgr,
            label,
            (x + point_radius + 6, y - point_radius - 6),
            color,
            label_font_scale,
            label_thickness,
        )


def draw_pose_result(
    image_bgr: np.ndarray,
    result: ImagePoseResult,
    draw_scores: bool = True,
) -> np.ndarray:
    """Draw pillar bboxes and HRNet keypoints on a copy of the input image."""
    out = image_bgr.copy()
    for pillar in result.pillars:
        names, _, colors = _result_vis_spec(result, len(pillar.keypoints_xy))
        _draw_single_pillar_overlay(
            out,
            bbox_xyxy=pillar.bbox_xyxy,
            keypoints_xy=pillar.keypoints_xy,
            keypoint_names=names,
            keypoint_colors=colors,
        )

    return out


def draw_corner_refine_comparison(
    image_bgr: np.ndarray,
    result: ImagePoseResult,
    draw_scores: bool = True,
) -> np.ndarray:
    """Side-by-side comparison: decoded HRNet points vs. refined points."""
    import cv2

    base = image_bgr.copy()
    refined = image_bgr.copy()

    for pillar in result.pillars:
        names, _, colors = _result_vis_spec(result, len(pillar.keypoints_xy))
        debug = getattr(pillar, "debug_info", None)
        base_kpts = pillar.keypoints_xy

        if isinstance(debug, _CornerRefineDebug):
            base_kpts = debug.decoded_keypoints_xy

        _draw_single_pillar_overlay(
            base,
            bbox_xyxy=pillar.bbox_xyxy,
            keypoints_xy=base_kpts,
            keypoint_names=names,
            keypoint_colors=colors,
            box_color=(180, 180, 180),
        )
        _draw_single_pillar_overlay(
            refined,
            bbox_xyxy=pillar.bbox_xyxy,
            keypoints_xy=pillar.keypoints_xy,
            keypoint_names=names,
            keypoint_colors=colors,
            box_color=(80, 220, 80),
        )

        if isinstance(debug, _CornerRefineDebug):
            x0, y0, _, _ = np.round(pillar.bbox_xyxy).astype(int)
            text_y = min(image_bgr.shape[0] - 8, max(40, y0 + 16))
            if debug.base_combo is not None and debug.best_combo is not None:
                cv2.putText(
                    refined,
                    f"E {debug.base_combo.total_cost:.3f}->{debug.best_combo.total_cost:.3f}",
                    (x0, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    refined,
                    f"DLT {debug.base_combo.dlt_err:.2e}->{debug.best_combo.dlt_err:.2e}",
                    (x0, min(image_bgr.shape[0] - 8, text_y + 16)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    refined,
                    f"shift {debug.max_corner_shift_px:.1f}px",
                    (x0, min(image_bgr.shape[0] - 8, text_y + 32)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

    sep = np.full((image_bgr.shape[0], 12, 3), 24, dtype=np.uint8)
    out = np.concatenate([base, sep, refined], axis=1)
    cv2.putText(
        out,
        "base decode",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        "corner refine",
        (image_bgr.shape[1] + sep.shape[1] + 12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (80, 220, 80),
        2,
        cv2.LINE_AA,
    )
    return out


def draw_heatmap_result(
    image_bgr: np.ndarray,
    pose_samples: list,
    xyxy: np.ndarray,
    kpt_idx: int = -1,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay HRNet predicted heatmaps on the original image (combined max across keypoints).
    Use :func:`draw_heatmap_grid` for per-keypoint grid layout.
    """
    import cv2

    out = image_bgr.copy()
    h_img, w_img = image_bgr.shape[:2]

    for i, sample in enumerate(pose_samples):
        if not hasattr(sample, "pred_fields") or sample.pred_fields is None:
            continue
        heatmaps = sample.pred_fields.heatmaps  # (K, H, W)
        if hasattr(heatmaps, "cpu"):
            heatmaps = heatmaps.cpu().numpy()
        heatmaps = np.array(heatmaps, dtype=np.float32)

        if kpt_idx == -1:
            combined = np.max(heatmaps, axis=0)
        else:
            combined = heatmaps[kpt_idx % heatmaps.shape[0]]

        lo, hi = combined.min(), combined.max()
        combined = (combined - lo) / (hi - lo + 1e-8)
        hm_uint8 = (combined * 255).astype(np.uint8)

        x0, y0, x1, y1 = xyxy[i].astype(int)
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(w_img, x1)
        y1 = min(h_img, y1)
        bw, bh = x1 - x0, y1 - y0
        if bw <= 0 or bh <= 0:
            continue

        hm_resized = cv2.resize(hm_uint8, (bw, bh), interpolation=cv2.INTER_LINEAR)
        hm_color = cv2.applyColorMap(hm_resized, cv2.COLORMAP_JET)
        roi = out[y0:y1, x0:x1]
        out[y0:y1, x0:x1] = cv2.addWeighted(roi, 1.0 - alpha, hm_color, alpha, 0)

    return out


def draw_heatmap_grid(
    image_bgr: np.ndarray,
    pose_samples: list,
    xyxy: np.ndarray,
    pillar_results: list,
    cell_size: int = 192,
    show_corner_debug: bool = False,
    draw_scores: bool = True,
    keypoint_names: Optional[Sequence[str]] = None,
    skeleton: Optional[Sequence[Sequence[int]]] = None,
    keypoint_colors: Optional[Sequence[Sequence[int]]] = None,
) -> np.ndarray:
    """
    Grid layout: one row per detected pillar.

    Columns::

        [ bbox crop + keypoints ] | [ one column per keypoint channel ]

    Returns:
        BGR grid image (H × W × 3).
    """
    import cv2

    h_img, w_img = image_bgr.shape[:2]
    if pose_samples and hasattr(pose_samples[0], "pred_fields") and pose_samples[0].pred_fields is not None:
        heatmaps0 = pose_samples[0].pred_fields.heatmaps
        if hasattr(heatmaps0, "shape"):
            n_kpts = int(heatmaps0.shape[0])
        else:
            n_kpts = len(keypoint_names) if keypoint_names is not None else len(_DEFAULT_KPT_NAMES)
    elif pillar_results:
        n_kpts = len(pillar_results[0].keypoints_xy)
    else:
        n_kpts = len(keypoint_names) if keypoint_names is not None else len(_DEFAULT_KPT_NAMES)
    keypoint_names, skeleton, keypoint_colors = _resolve_vis_spec(
        n_kpts=n_kpts,
        keypoint_names=keypoint_names,
        skeleton=skeleton,
        keypoint_colors=keypoint_colors,
    )
    n_cols = 1 + n_kpts  # original crop  +  K heatmaps
    n_rows = max(1, len(pose_samples))
    grid_h = n_rows * cell_size
    grid_w = n_cols * cell_size
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for row_i, sample in enumerate(pose_samples):
        y_off = row_i * cell_size

        # ── Column 0: original crop with keypoints ──────────────────────────
        x0, y0, x1, y1 = xyxy[row_i].astype(int)
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(w_img, x1)
        y1 = min(h_img, y1)
        bw = max(x1 - x0, 1)
        bh = max(y1 - y0, 1)
        crop = (
            image_bgr[y0:y1, x0:x1].copy()
            if (x1 > x0 and y1 > y0)
            else np.zeros((cell_size, cell_size, 3), dtype=np.uint8)
        )

        # ── Compute affine warp that matches the TopdownAffine network input ──
        # input_scale is in pixels (equalized + padded by GetBBoxCenterScale).
        # Mapping: cell_x = (orig_x - cx) / sw * cell_size + cell_size/2
        try:
            meta = sample.metainfo
            cx, cy = float(meta["input_center"][0]), float(meta["input_center"][1])
            sw, sh = float(meta["input_scale"][0]), float(meta["input_scale"][1])
            # Build 2×3 affine matrix for cv2.warpAffine
            scale_x = cell_size / sw
            scale_y = cell_size / sh
            M_affine = np.array(
                [
                    [scale_x, 0.0, -cx * scale_x + cell_size / 2.0],
                    [0.0, scale_y, -cy * scale_y + cell_size / 2.0],
                ],
                dtype=np.float32,
            )
            affine_bg = cv2.warpAffine(
                image_bgr, M_affine, (cell_size, cell_size), flags=cv2.INTER_LINEAR
            )
            has_affine = True
        except Exception:
            affine_bg = cv2.resize(
                crop, (cell_size, cell_size), interpolation=cv2.INTER_LINEAR
            )
            has_affine = False

        def to_cell(pts_xy: np.ndarray) -> np.ndarray:
            """Map (N,2) original-image coords → cell_size coords."""
            if has_affine:
                p = pts_xy.astype(np.float32)
                cell_x = (p[:, 0] - cx) / sw * cell_size + cell_size / 2.0
                cell_y = (p[:, 1] - cy) / sh * cell_size + cell_size / 2.0
            else:
                cell_x = (pts_xy[:, 0] - x0) * (cell_size / bw)
                cell_y = (pts_xy[:, 1] - y0) * (cell_size / bh)
            return np.stack([cell_x, cell_y], axis=1).astype(int)

        # ── Column 0: affine background with keypoints ───────────────────────
        cell0 = affine_bg.copy()
        if row_i < len(pillar_results):
            pil = pillar_results[row_i]
            debug = getattr(pil, "debug_info", None)
            if show_corner_debug and isinstance(debug, _CornerRefineDebug):
                base_kpts_cell = to_cell(debug.decoded_keypoints_xy)
                for px, py in base_kpts_cell:
                    cv2.drawMarker(
                        cell0,
                        (int(px), int(py)),
                        (255, 255, 255),
                        markerType=cv2.MARKER_TILTED_CROSS,
                        markerSize=8,
                        thickness=1,
                        line_type=cv2.LINE_AA,
                    )
            kpts_cell = to_cell(pil.keypoints_xy)  # (K, 2)
            for k_i, (px, py) in enumerate(kpts_cell):
                color = keypoint_colors[k_i % len(keypoint_colors)]
                cv2.circle(cell0, (int(px), int(py)), 16, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(cell0, (int(px), int(py)), 12, color, -1, cv2.LINE_AA)
        if row_i < len(pillar_results):
            pil = pillar_results[row_i]
            debug = getattr(pil, "debug_info", None)
            if show_corner_debug and isinstance(debug, _CornerRefineDebug):
                cv2.putText(
                    cell0,
                    f"shift {debug.max_corner_shift_px:.1f}px",
                    (4, cell_size - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                if debug.base_combo is not None and debug.best_combo is not None:
                    cv2.putText(
                        cell0,
                        f"E {debug.base_combo.total_cost:.2f}->{debug.best_combo.total_cost:.2f}",
                        (4, cell_size - 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.38,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        cell0,
                        f"DLT {debug.base_combo.dlt_err:.1e}->{debug.best_combo.dlt_err:.1e}",
                        (4, cell_size - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.34,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
        grid[y_off : y_off + cell_size, 0:cell_size] = cell0

        # Background for heatmap cells (same affine crop, no keypoints)
        crop_bg = affine_bg

        # ── Columns 1..K: per-keypoint heatmaps ─────────────────────────────
        if not hasattr(sample, "pred_fields") or sample.pred_fields is None:
            continue
        heatmaps = sample.pred_fields.heatmaps  # (K, H, W)
        if hasattr(heatmaps, "cpu"):
            heatmaps = heatmaps.cpu().numpy()
        heatmaps = np.array(heatmaps, dtype=np.float32)

        for k in range(min(n_kpts, heatmaps.shape[0])):
            hm = heatmaps[k]
            lo, hi = hm.min(), hm.max()
            hm_norm = (hm - lo) / (hi - lo + 1e-8)
            hm_uint8 = (hm_norm * 255).astype(np.uint8)
            # Smooth before upsampling to avoid stride-4 quantization grid artifacts
            hm_blurred = cv2.GaussianBlur(hm_uint8, (3, 3), 0)
            hm_color = cv2.applyColorMap(
                cv2.resize(
                    hm_blurred, (cell_size, cell_size), interpolation=cv2.INTER_CUBIC
                ),
                cv2.COLORMAP_JET,
            )
            # Blend heatmap over the affine crop background (exact same coordinate space)
            cell = cv2.addWeighted(crop_bg, 0.45, hm_color, 0.55, 0)
            debug = None
            if row_i < len(pillar_results):
                debug = getattr(pillar_results[row_i], "debug_info", None)
            # Draw decoded keypoint marker on this heatmap cell.
            # In debug mode the white cross is the pre-refine HRNet decode, while
            # the coloured dot is the final selected point after corner refine.
            if row_i < len(pillar_results) and k < len(kpts_cell):
                if (
                    show_corner_debug
                    and isinstance(debug, _CornerRefineDebug)
                    and k < len(debug.decoded_keypoints_xy)
                ):
                    base_pt = to_cell(debug.decoded_keypoints_xy[[k]])[0]
                    mx, my = int(base_pt[0]), int(base_pt[1])
                else:
                    mx, my = int(kpts_cell[k][0]), int(kpts_cell[k][1])
                arm = 5
                cv2.line(
                    cell,
                    (mx - arm, my),
                    (mx + arm, my),
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.line(
                    cell,
                    (mx, my - arm),
                    (mx, my + arm),
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.circle(
                    cell,
                    (mx, my),
                    5,
                    keypoint_colors[k % len(keypoint_colors)],
                    -1,
                    cv2.LINE_AA,
                )
                if (
                    show_corner_debug
                    and isinstance(debug, _CornerRefineDebug)
                    and k < 4
                    and debug.corner_candidates
                ):
                    chosen_idx = None
                    if debug.chosen_candidate_indices is not None:
                        chosen_idx = debug.chosen_candidate_indices[k]
                    for cand_idx, cand in enumerate(debug.corner_candidates[k]):
                        cand_cell = to_cell(cand.xy.reshape(1, 2))[0]
                        cx_c, cy_c = int(cand_cell[0]), int(cand_cell[1])
                        cand_color = (255, 255, 255)
                        thickness = 1
                        radius = 4
                        if chosen_idx is not None and cand_idx == chosen_idx:
                            cand_color = (0, 220, 255)
                            thickness = 2
                            radius = 6
                        cv2.circle(
                            cell,
                            (cx_c, cy_c),
                            radius,
                            cand_color,
                            thickness,
                            cv2.LINE_AA,
                        )
                        cv2.putText(
                            cell,
                            str(cand_idx),
                            (cx_c + 4, cy_c + 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            cand_color,
                            1,
                            cv2.LINE_AA,
                        )
                    if chosen_idx is not None:
                        chosen_pt = kpts_cell[k]
                        cv2.line(
                            cell,
                            (mx, my),
                            (int(chosen_pt[0]), int(chosen_pt[1])),
                            (0, 220, 255),
                            1,
                            cv2.LINE_AA,
                        )
            # Label and score on the blended cell
            kpt_name = keypoint_names[k] if k < len(keypoint_names) else f"kp{k}"
            kpt_label = _shorten_kpt_label(kpt_name)
            color = keypoint_colors[k % len(keypoint_colors)]
            _draw_outlined_text(
                cell,
                kpt_label,
                (4, 18),
                color,
                0.75,
                2,
            )
            # Score
            if draw_scores and row_i < len(pillar_results):
                sc = pillar_results[row_i].keypoint_scores
                if k < len(sc):
                    cv2.putText(
                        cell,
                        f"{sc[k]:.3f}",
                        (4, cell_size - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                if (
                    show_corner_debug
                    and isinstance(debug, _CornerRefineDebug)
                    and k < 4
                    and debug.chosen_candidate_indices is not None
                ):
                    cand_idx = debug.chosen_candidate_indices[k]
                    cv2.putText(
                        cell,
                        f"pick c{cand_idx}",
                        (4, cell_size - 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
            x_off = (k + 1) * cell_size
            grid[y_off : y_off + cell_size, x_off : x_off + cell_size] = cell

    # Column header labels on top row are embedded in row 0 via putText.
    return grid


def save_pose_result_image(
    image_bgr: np.ndarray,
    result: ImagePoseResult,
    output_path: Union[str, Path],
    draw_scores: bool = True,
) -> Path:
    """Render one result and save it to disk."""
    import cv2

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vis = draw_pose_result(
        image_bgr,
        result,
        draw_scores=draw_scores,
    )
    ok = cv2.imwrite(str(output_path), vis)
    if not ok:
        raise IOError(f"Failed to write visualization image: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# YOLO: pillar bboxes only
# ---------------------------------------------------------------------------


def yolo_extract_class_bboxes(
    result: Any,
    class_id: int = 0,
    conf_threshold: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """
    From one Ultralytics ``Results``, keep only boxes of the requested class.

    Returns:
        xyxy: (N, 4) float32
        scores: (N,) float32
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()

    mask = (cls == class_id) & (conf >= conf_threshold)
    xyxy = xyxy[mask]
    conf = conf[mask]
    if xyxy.size == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    order = np.argsort(-conf)
    return xyxy[order].astype(np.float32), conf[order].astype(np.float32)


# ---------------------------------------------------------------------------
# MMPose: HRNet samples → numpy
# ---------------------------------------------------------------------------


def _pose_sample_to_instance(
    sample: Any,
    bbox_xyxy: np.ndarray,
    yolo_score: float,
) -> PillarInstance:
    """Convert one ``PoseDataSample`` to ``PillarInstance``."""
    inst = sample.pred_instances
    kpts = inst.keypoints
    scores = inst.keypoint_scores
    vis_scores = getattr(inst, "keypoints_visible", None)
    if hasattr(kpts, "cpu"):
        kpts = kpts.cpu().numpy()
    else:
        kpts = np.asarray(kpts)
    if hasattr(scores, "cpu"):
        scores = scores.cpu().numpy()
    else:
        scores = np.asarray(scores)
    if vis_scores is not None:
        if hasattr(vis_scores, "cpu"):
            vis_scores = vis_scores.cpu().numpy()
        else:
            vis_scores = np.asarray(vis_scores)

    if kpts.ndim == 3 and kpts.shape[0] == 1:
        kpts = kpts[0]
    if scores.ndim == 2 and scores.shape[0] == 1:
        scores = scores[0]
    if vis_scores is not None and vis_scores.ndim == 2 and vis_scores.shape[0] == 1:
        vis_scores = vis_scores[0]

    return PillarInstance(
        bbox_xyxy=bbox_xyxy.astype(np.float32),
        yolo_score=float(yolo_score),
        keypoints_xy=kpts.astype(np.float32),
        keypoint_scores=scores.astype(np.float32),
        keypoint_visible_scores=(
            vis_scores.astype(np.float32) if vis_scores is not None else None
        ),
    )


def _extract_vis_spec_from_mmpose_model(
    model: Any,
) -> tuple[list[str], list[tuple[int, int]], list[tuple[int, int, int]]]:
    """Read dynamic keypoint metadata from a loaded MMPose model."""
    dataset_meta = getattr(model, "dataset_meta", None) or {}
    num_keypoints = int(dataset_meta.get("num_keypoints", 0))

    keypoint_id2name = dataset_meta.get("keypoint_id2name", {}) or {}
    keypoint_names = [
        str(keypoint_id2name[i]) for i in range(num_keypoints) if i in keypoint_id2name
    ]
    if len(keypoint_names) < num_keypoints:
        keypoint_names = None

    skeleton = dataset_meta.get("skeleton_links", None)
    keypoint_colors = dataset_meta.get("keypoint_colors", None)
    n_kpts = max(
        num_keypoints,
        len(keypoint_names) if keypoint_names is not None else 0,
        len(keypoint_colors) if keypoint_colors is not None else 0,
    )
    if n_kpts <= 0:
        n_kpts = len(_DEFAULT_KPT_NAMES)

    return _resolve_vis_spec(
        n_kpts=n_kpts,
        keypoint_names=keypoint_names,
        skeleton=skeleton,
        keypoint_colors=keypoint_colors,
    )


def _as_numpy(value: Any) -> np.ndarray:
    """Convert torch/np/list-like values to numpy arrays."""
    if hasattr(value, "cpu"):
        value = value.cpu().numpy()
    return np.asarray(value)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class PillarHrnetPipeline:
    """
    YOLO (pillar bbox) + HRNet keypoints.

    Parameters
    ----------
    yolo_weights:
        Path to ``.pt`` for Ultralytics YOLO Detect (bbox only).
    hrnet_config:
        MMPose config path (e.g. ``hrnet.py``).
    hrnet_checkpoint:
        Trained HRNet weights (``best_*.pth``).
    device:
        e.g. ``\"cuda:0\"``, ``\"cpu\"``.
    pillar_class_id:
        YOLO class index for pillar (default ``0`` per ``compose_dataset.py``).
    yolo_conf:
        Min YOLO box confidence before HRNet.
    """

    def __init__(
        self,
        yolo_weights: Union[str, Path],
        hrnet_config: Union[str, Path],
        hrnet_checkpoint: Union[str, Path],
        device: str = "cuda:0",
        pillar_class_id: int = 0,
        bbox_class_id: Optional[int] = None,
        yolo_conf: float = 0.25,
        crop_margin: float = 0.5,
        corner_refine: bool = True,
        corner_topk: int = 4,
        corner_peak_rel_thr: float = 0.15,
        corner_peak_abs_thr: float = 0.05,
        corner_dlt_weight: float = 20.0,
        corner_chiral_weight: float = 2.0,
        corner_anchor_weight: float = 0.5,
    ) -> None:
        self.pillar_class_id = pillar_class_id
        self.bbox_class_id = bbox_class_id
        self.yolo_conf = yolo_conf
        self.crop_margin = crop_margin
        self.device = device
        self.corner_refine = bool(corner_refine)
        self.corner_topk = max(int(corner_topk), 1)
        self.corner_peak_rel_thr = float(corner_peak_rel_thr)
        self.corner_peak_abs_thr = float(corner_peak_abs_thr)
        self.corner_dlt_weight = float(corner_dlt_weight)
        self.corner_chiral_weight = float(corner_chiral_weight)
        self.corner_anchor_weight = float(corner_anchor_weight)

        from ultralytics import YOLO

        self._yolo = YOLO(str(yolo_weights))

        from mmpose.apis import init_model

        self._pose = init_model(
            str(hrnet_config),
            str(hrnet_checkpoint),
            device=device,
        )
        (
            self._vis_keypoint_names,
            self._vis_skeleton,
            self._vis_keypoint_colors,
        ) = _extract_vis_spec_from_mmpose_model(self._pose)
        self.bbox_class_id = self._resolve_bbox_class_id(self.bbox_class_id)

    @property
    def yolo_model(self) -> Any:
        return self._yolo

    @property
    def hrnet_model(self) -> Any:
        return self._pose

    def _resolve_bbox_class_id(self, bbox_class_id: Optional[int]) -> int:
        """
        Choose which YOLO class provides the top-down crop.

        Default rule:
          - 5-keypoint model  -> pillar bbox (class 0 by default)
          - >5-keypoint model -> exchange bbox (class 1 by default)
        """
        if bbox_class_id is not None:
            return int(bbox_class_id)
        if len(self._vis_keypoint_names) > 5:
            return 1
        return int(self.pillar_class_id)

    def _run_pose_inference(
        self,
        image_bgr: np.ndarray,
        xyxy: np.ndarray,
        output_heatmaps: bool,
    ) -> list[Any]:
        """Run MMPose top-down and optionally force heatmap export."""
        from mmpose.apis import inference_topdown

        need_heatmaps = output_heatmaps or self.corner_refine
        if not need_heatmaps:
            return inference_topdown(self._pose, image_bgr, xyxy, bbox_format="xyxy")

        orig_output_heatmaps = self._pose.test_cfg.get("output_heatmaps", False)
        self._pose.test_cfg["output_heatmaps"] = True
        try:
            return inference_topdown(self._pose, image_bgr, xyxy, bbox_format="xyxy")
        finally:
            self._pose.test_cfg["output_heatmaps"] = orig_output_heatmaps

    def _score_corner_combo(
        self,
        combo: tuple[_PeakCandidate, ...],
        candidate_indices: tuple[int, int, int, int],
        decoded_kpts: np.ndarray,
        ring_xy: np.ndarray,
        norm_size: float,
    ) -> Optional[_CornerComboScore]:
        """Score one 4-corner assignment with the current objective."""
        corners_xy = np.stack([cand.xy for cand in combo], axis=0).astype(np.float32)
        dlt_err = _dlt_reprojection_error(
            np.vstack([corners_xy, ring_xy]),
            input_size=norm_size,
        )
        if not np.isfinite(dlt_err):
            return None

        peak_scores = np.array([cand.score for cand in combo], dtype=np.float32)
        channel_cost = float(np.mean(1.0 - np.clip(peak_scores, 0.0, 1.0)))
        anchor_cost = float(
            np.mean(np.linalg.norm(corners_xy - decoded_kpts[:4], axis=1) / norm_size)
        )
        chiral_cost = _chirality_penalty(corners_xy, margin=0.0)
        total_cost = (
            channel_cost
            + self.corner_dlt_weight * dlt_err
            + self.corner_chiral_weight * chiral_cost
            + self.corner_anchor_weight * anchor_cost
        )
        return _CornerComboScore(
            total_cost=float(total_cost),
            channel_cost=channel_cost,
            dlt_err=float(dlt_err),
            chiral_cost=float(chiral_cost),
            anchor_cost=anchor_cost,
            candidate_indices=candidate_indices,
            corners_xy=corners_xy.copy(),
        )

    def _refine_single_pillar(
        self,
        pillar: PillarInstance,
        heatmaps_img: np.ndarray,
        input_scale: np.ndarray,
    ) -> None:
        """
        Reassign the 4 corners by global search over heatmap peaks.

        Score = channel confidence + λ_dlt * 5-point DLT reprojection error
                + λ_chiral * chirality penalty
                + λ_anchor * distance-to-decoded-point regularisation.
        """
        if heatmaps_img.ndim != 3 or heatmaps_img.shape[0] < 5:
            return
        if pillar.keypoints_xy.shape[0] < 5 or pillar.keypoint_scores.shape[0] < 5:
            return

        decoded_kpts = pillar.keypoints_xy.astype(np.float32, copy=True)
        decoded_scores = pillar.keypoint_scores.astype(np.float32, copy=True)
        ring_xy = decoded_kpts[4]

        input_scale = np.asarray(input_scale, dtype=np.float32).reshape(-1)
        norm_size = float(max(np.max(input_scale), 1.0))
        peak_radius_px = max(2, int(round(0.015 * norm_size)))
        candidate_dedup_px = max(1.5, 0.008 * norm_size)
        combo_min_dist_px = max(3.0, 0.015 * norm_size)
        debug = _CornerRefineDebug(
            status="init",
            norm_size=norm_size,
            combo_min_dist_px=float(combo_min_dist_px),
            decoded_keypoints_xy=decoded_kpts.copy(),
            decoded_keypoint_scores=decoded_scores.copy(),
            refined_keypoints_xy=decoded_kpts.copy(),
            refined_keypoint_scores=decoded_scores.copy(),
        )

        corner_candidates: list[list[_PeakCandidate]] = []
        for k in range(4):
            candidates = _extract_peak_candidates(
                heatmaps_img[k],
                decoded_xy=decoded_kpts[k],
                decoded_score=float(decoded_scores[k]),
                topk=self.corner_topk,
                nms_radius_px=peak_radius_px,
                rel_threshold=self.corner_peak_rel_thr,
                abs_threshold=self.corner_peak_abs_thr,
                dedup_dist_px=candidate_dedup_px,
            )
            if not candidates:
                debug.status = f"no_candidates_k{k}"
                pillar.debug_info = debug
                return
            corner_candidates.append(candidates)
        debug.corner_candidates = _copy_peak_candidates(corner_candidates)

        best_score = np.inf
        best_combo: Optional[tuple[_PeakCandidate, ...]] = None
        best_score_info: Optional[_CornerComboScore] = None
        second_score_info: Optional[_CornerComboScore] = None
        base_indices = (0, 0, 0, 0)
        base_combo = tuple(cands[0] for cands in corner_candidates)
        debug.base_combo = self._score_corner_combo(
            base_combo,
            candidate_indices=base_indices,
            decoded_kpts=decoded_kpts,
            ring_xy=ring_xy,
            norm_size=norm_size,
        )

        index_ranges = [range(len(cands)) for cands in corner_candidates]
        for idx_tuple in product(*index_ranges):
            combo = tuple(corner_candidates[k][idx_tuple[k]] for k in range(4))
            corners_xy = np.stack([cand.xy for cand in combo], axis=0)
            if _min_pairwise_distance(corners_xy) < combo_min_dist_px:
                continue

            score_info = self._score_corner_combo(
                combo,
                candidate_indices=tuple(int(v) for v in idx_tuple),
                decoded_kpts=decoded_kpts,
                ring_xy=ring_xy,
                norm_size=norm_size,
            )
            if score_info is None:
                continue

            if score_info.total_cost < best_score:
                if best_score_info is not None:
                    second_score_info = best_score_info
                best_score = score_info.total_cost
                best_score_info = score_info
                best_combo = combo
            elif (
                second_score_info is None
                or score_info.total_cost < second_score_info.total_cost
            ):
                second_score_info = score_info

        if best_combo is None:
            debug.status = "no_feasible_combo"
            pillar.debug_info = debug
            return

        for idx, cand in enumerate(best_combo):
            pillar.keypoints_xy[idx] = cand.xy
            pillar.keypoint_scores[idx] = float(cand.score)
        debug.best_combo = best_score_info
        debug.second_combo = second_score_info
        debug.chosen_candidate_indices = (
            best_score_info.candidate_indices if best_score_info is not None else None
        )
        debug.refined_keypoints_xy = pillar.keypoints_xy.astype(np.float32, copy=True)
        debug.refined_keypoint_scores = pillar.keypoint_scores.astype(
            np.float32, copy=True
        )
        debug.max_corner_shift_px = float(
            np.max(np.linalg.norm(pillar.keypoints_xy[:4] - decoded_kpts[:4], axis=1))
        )
        debug.status = "refined" if debug.max_corner_shift_px > 1e-3 else "unchanged"
        pillar.debug_info = debug

    def _refine_pillars_with_geometry(
        self,
        pillars: list[PillarInstance],
        pose_samples: list[Any],
        image_shape_hw: tuple[int, int],
    ) -> None:
        """Apply corner reassignment on each pillar if heatmaps are available."""
        if not self.corner_refine or not pillars or not pose_samples:
            return

        from mmpose.structures.utils import revert_heatmap

        for pillar, sample in zip(pillars, pose_samples):
            if not hasattr(sample, "pred_fields") or sample.pred_fields is None:
                continue

            heatmaps = getattr(sample.pred_fields, "heatmaps", None)
            if heatmaps is None:
                continue
            if hasattr(heatmaps, "cpu"):
                heatmaps = heatmaps.cpu().numpy()
            heatmaps = np.asarray(heatmaps, dtype=np.float32)
            if heatmaps.ndim != 3 or heatmaps.shape[0] < 5:
                continue

            meta = getattr(sample, "metainfo", {}) or {}
            input_center = meta.get("input_center", None)
            input_scale = meta.get("input_scale", None)
            if input_center is None or input_scale is None:
                continue

            try:
                heatmaps_img = revert_heatmap(
                    heatmaps,
                    np.asarray(input_center, dtype=np.float32),
                    np.asarray(input_scale, dtype=np.float32),
                    image_shape_hw,
                )
            except Exception:
                continue

            self._refine_single_pillar(
                pillar,
                heatmaps_img=heatmaps_img,
                input_scale=np.asarray(input_scale, dtype=np.float32),
            )

    def _predict_impl(
        self,
        image_bgr: np.ndarray,
        image_id: Optional[Union[str, Path]],
        return_pose_samples: bool,
    ) -> tuple[ImagePoseResult, list[Any], np.ndarray]:
        """Shared inference path for plain prediction and heatmap visualization."""
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("image_bgr must be HxWx3 BGR uint8/float")

        h, w = image_bgr.shape[:2]
        ref: Union[str, Path, np.ndarray] = (
            image_id if image_id is not None else image_bgr
        )
        empty = ImagePoseResult(
            image=ref,
            image_shape_hw=(h, w),
            pillars=[],
            keypoint_names=list(self._vis_keypoint_names),
            skeleton=list(self._vis_skeleton),
            keypoint_colors=list(self._vis_keypoint_colors),
        )

        yolo_results = self._yolo(image_bgr, verbose=False, conf=self.yolo_conf)
        if not yolo_results:
            return empty, [], np.zeros((0, 4), dtype=np.float32)

        y0 = yolo_results[0]
        xyxy, box_scores = yolo_extract_class_bboxes(
            y0,
            class_id=self.bbox_class_id,
            conf_threshold=self.yolo_conf,
        )
        if xyxy.shape[0] == 0:
            return empty, [], xyxy

        if self.crop_margin > 0:
            ws = (xyxy[:, 2] - xyxy[:, 0]) * self.crop_margin
            hs = (xyxy[:, 3] - xyxy[:, 1]) * self.crop_margin
            xyxy_exp = xyxy.copy()
            xyxy_exp[:, 0] = np.clip(xyxy[:, 0] - ws, 0, w)
            xyxy_exp[:, 1] = np.clip(xyxy[:, 1] - hs, 0, h)
            xyxy_exp[:, 2] = np.clip(xyxy[:, 2] + ws, 0, w)
            xyxy_exp[:, 3] = np.clip(xyxy[:, 3] + hs, 0, h)
            xyxy = xyxy_exp

        pose_samples = self._run_pose_inference(
            image_bgr,
            xyxy,
            output_heatmaps=return_pose_samples,
        )
        if len(pose_samples) != len(xyxy):
            raise RuntimeError(
                f"HRNet returned {len(pose_samples)} samples for {len(xyxy)} bboxes"
            )

        pillars: list[PillarInstance] = []
        for i, sample in enumerate(pose_samples):
            pillars.append(
                _pose_sample_to_instance(sample, xyxy[i], float(box_scores[i]))
            )

        self._refine_pillars_with_geometry(
            pillars,
            pose_samples,
            image_shape_hw=(h, w),
        )

        result = ImagePoseResult(
            image=ref,
            image_shape_hw=(h, w),
            pillars=pillars,
            keypoint_names=list(self._vis_keypoint_names),
            skeleton=list(self._vis_skeleton),
            keypoint_colors=list(self._vis_keypoint_colors),
        )
        return result, (pose_samples if return_pose_samples else []), xyxy

    def predict_numpy(
        self,
        image_bgr: np.ndarray,
        image_id: Optional[Union[str, Path]] = None,
    ) -> ImagePoseResult:
        """
        Run on one BGR image (``cv2.imread`` format).

        ``image_id`` is only stored in the result for tracing; inference does not read files.
        """
        result, _, _ = self._predict_impl(
            image_bgr,
            image_id=image_id,
            return_pose_samples=False,
        )
        return result

    def predict_with_heatmaps(
        self,
        image_bgr: np.ndarray,
        image_id: Optional[Union[str, Path]] = None,
    ) -> tuple:
        """
        Same as :meth:`predict_numpy` but also returns raw ``pose_samples``
        and the expanded ``xyxy`` bboxes, enabling heatmap visualization.

        Returns:
            (ImagePoseResult, pose_samples, xyxy_expanded)
            ``pose_samples`` is an empty list when no pillar is detected.
        """
        return self._predict_impl(
            image_bgr,
            image_id=image_id,
            return_pose_samples=True,
        )

    def predict_image(self, path: Union[str, Path]) -> ImagePoseResult:
        """Load image from path (BGR) and run :meth:`predict_numpy`."""
        import cv2

        p = Path(path)
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {p}")
        return self.predict_numpy(img, image_id=p)

    def predict_images(
        self,
        paths: Sequence[Union[str, Path]],
    ) -> BatchPoseResult:
        """
        Batch over **file paths**. Each file is processed independently (YOLO + HRNet).
        Order matches ``paths``.
        """
        out: list[ImagePoseResult] = []
        for p in paths:
            out.append(self.predict_image(p))
        return BatchPoseResult(images=out)

    def predict_batch_numpy(
        self,
        images_bgr: Sequence[np.ndarray],
        image_ids: Optional[Sequence[Optional[Union[str, Path]]]] = None,
    ) -> BatchPoseResult:
        """
        Batch over in-memory BGR images. Same order as input.

        ``image_ids`` optional parallel list of ids for :attr:`ImagePoseResult.image`.
        """
        if image_ids is not None and len(image_ids) != len(images_bgr):
            raise ValueError("image_ids length must match images_bgr")
        out: list[ImagePoseResult] = []
        for i, im in enumerate(images_bgr):
            iid = image_ids[i] if image_ids is not None else None
            out.append(self.predict_numpy(im, image_id=iid))
        return BatchPoseResult(images=out)

    def save_visualization(
        self,
        image: Union[str, Path, np.ndarray],
        result: ImagePoseResult,
        output_path: Union[str, Path],
        draw_scores: bool = True,
    ) -> Path:
        """Save one visualization from a file path or in-memory BGR image."""
        import cv2

        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {image}")
        else:
            img = image
        return save_pose_result_image(
            img,
            result,
            output_path,
            draw_scores=draw_scores,
        )


class RtmoPipeline:
    """
    Single-stage RTMO inference.

    This adapter maps MMPose bottom-up ``pred_instances`` to the same
    ``ImagePoseResult`` / ``PillarInstance`` output type used by
    ``PillarHrnetPipeline`` so downstream visualization can be shared.
    """

    def __init__(
        self,
        rtmo_config: Union[str, Path],
        rtmo_checkpoint: Union[str, Path],
        device: str = "cuda:0",
        score_thr: Optional[float] = None,
    ) -> None:
        self.device = device
        self.score_thr = score_thr

        from mmpose.apis import init_model

        self._pose = init_model(
            str(rtmo_config),
            str(rtmo_checkpoint),
            device=device,
        )
        (
            self._vis_keypoint_names,
            self._vis_skeleton,
            self._vis_keypoint_colors,
        ) = _extract_vis_spec_from_mmpose_model(self._pose)

    @property
    def rtmo_model(self) -> Any:
        return self._pose

    def _run_pose_inference(self, image_bgr: np.ndarray) -> list[Any]:
        from mmpose.apis import inference_bottomup

        return inference_bottomup(self._pose, image_bgr)

    def _instances_to_pillars(self, pred_instances: Any) -> list[PillarInstance]:
        if pred_instances is None or "keypoints" not in pred_instances:
            return []

        keypoints = _as_numpy(pred_instances.keypoints).astype(np.float32)
        if keypoints.ndim == 2:
            keypoints = keypoints[None, ...]

        keypoint_scores = getattr(pred_instances, "keypoint_scores", None)
        if keypoint_scores is None:
            keypoint_scores = np.ones(keypoints.shape[:2], dtype=np.float32)
        else:
            keypoint_scores = _as_numpy(keypoint_scores).astype(np.float32)
            if keypoint_scores.ndim == 1:
                keypoint_scores = keypoint_scores[None, ...]

        bboxes = getattr(pred_instances, "bboxes", None)
        if bboxes is None:
            bboxes = np.stack(
                [
                    np.nanmin(keypoints[..., 0], axis=1),
                    np.nanmin(keypoints[..., 1], axis=1),
                    np.nanmax(keypoints[..., 0], axis=1),
                    np.nanmax(keypoints[..., 1], axis=1),
                ],
                axis=1,
            )
        else:
            bboxes = _as_numpy(bboxes).astype(np.float32)
            if bboxes.ndim == 1:
                bboxes = bboxes[None, ...]

        bbox_scores = getattr(pred_instances, "bbox_scores", None)
        if bbox_scores is None:
            bbox_scores = getattr(pred_instances, "scores", None)
        if bbox_scores is None:
            bbox_scores = np.ones((keypoints.shape[0],), dtype=np.float32)
        else:
            bbox_scores = _as_numpy(bbox_scores).astype(np.float32).reshape(-1)

        vis_scores = getattr(pred_instances, "keypoints_visible", None)
        if vis_scores is None:
            vis_scores = None
        else:
            vis_scores = _as_numpy(vis_scores).astype(np.float32)
            if vis_scores.ndim == 1:
                vis_scores = vis_scores[None, ...]

        order = np.argsort(-bbox_scores)
        pillars: list[PillarInstance] = []
        for idx in order:
            score = float(bbox_scores[idx])
            if self.score_thr is not None and score < float(self.score_thr):
                continue
            pillars.append(
                PillarInstance(
                    bbox_xyxy=bboxes[idx].astype(np.float32),
                    yolo_score=score,
                    keypoints_xy=keypoints[idx].astype(np.float32),
                    keypoint_scores=keypoint_scores[idx].astype(np.float32),
                    keypoint_visible_scores=(
                        vis_scores[idx].astype(np.float32)
                        if vis_scores is not None and idx < len(vis_scores)
                        else None
                    ),
                )
            )
        return pillars

    def predict_numpy(
        self,
        image_bgr: np.ndarray,
        image_id: Optional[Union[str, Path]] = None,
    ) -> ImagePoseResult:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("image_bgr must be HxWx3 BGR uint8/float")

        h, w = image_bgr.shape[:2]
        ref: Union[str, Path, np.ndarray] = (
            image_id if image_id is not None else image_bgr
        )
        samples = self._run_pose_inference(image_bgr)
        pillars: list[PillarInstance] = []
        if samples:
            pillars = self._instances_to_pillars(samples[0].pred_instances)

        return ImagePoseResult(
            image=ref,
            image_shape_hw=(h, w),
            pillars=pillars,
            keypoint_names=list(self._vis_keypoint_names),
            skeleton=list(self._vis_skeleton),
            keypoint_colors=list(self._vis_keypoint_colors),
        )

    def predict_image(self, path: Union[str, Path]) -> ImagePoseResult:
        import cv2

        p = Path(path)
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {p}")
        return self.predict_numpy(img, image_id=p)

    def predict_images(
        self,
        paths: Sequence[Union[str, Path]],
    ) -> BatchPoseResult:
        return BatchPoseResult(images=[self.predict_image(p) for p in paths])

    def save_visualization(
        self,
        image: Union[str, Path, np.ndarray],
        result: ImagePoseResult,
        output_path: Union[str, Path],
        draw_scores: bool = True,
    ) -> Path:
        import cv2

        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Cannot read image: {image}")
        else:
            img = image
        return save_pose_result_image(
            img,
            result,
            output_path,
            draw_scores=draw_scores,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pillar keypoint inference: YOLO+HRNet topdown or RTMO bottomup"
    )
    p.add_argument(
        "--mode",
        choices=("hrnet", "rtmo"),
        default="hrnet",
        help="Inference backend. hrnet keeps the old YOLO+HRNet pipeline.",
    )
    p.add_argument("--yolo-weights", help="Ultralytics YOLO Detect .pt")
    p.add_argument(
        "--hrnet-config", help="MMPose config for top-down HRNet/CSPNeXt model"
    )
    p.add_argument("--hrnet-checkpoint", help="Top-down checkpoint .pth")
    p.add_argument("--rtmo-config", help="MMPose RTMO config")
    p.add_argument("--rtmo-checkpoint", help="RTMO checkpoint .pth")
    p.add_argument(
        "--rtmo-score-thr",
        type=float,
        default=None,
        help="Optional extra filter on RTMO instance score; config test_cfg already has its own threshold.",
    )
    p.add_argument("--images", nargs="+", required=True, help="One or more image paths")
    p.add_argument(
        "--output-dir",
        default="vis",
        help="Directory to save visualization images (default: vis/)",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--pillar-class-id", type=int, default=0)
    p.add_argument(
        "--bbox-class-id",
        type=int,
        default=None,
        help="YOLO class id used for top-down crop; default auto: 0 for 5-kpt models, 1 for >5-kpt models",
    )
    p.add_argument("--yolo-conf", type=float, default=0.1)
    p.add_argument(
        "--crop-margin",
        type=float,
        default=0.5,
        help="Expand YOLO bbox by this fraction on each side before HRNet (0.5 → 2x box size)",
    )
    p.add_argument(
        "--no-corner-refine",
        action="store_true",
        help="Disable heatmap peak + geometry based corner reassignment",
    )
    p.add_argument(
        "--corner-topk",
        type=int,
        default=4,
        help="Top-K heatmap peaks kept per corner channel before global search",
    )
    p.add_argument(
        "--corner-peak-rel-thr",
        type=float,
        default=0.15,
        help="Relative threshold for heatmap candidate extraction (fraction of channel max)",
    )
    p.add_argument(
        "--corner-peak-abs-thr",
        type=float,
        default=0.05,
        help="Absolute threshold for heatmap candidate extraction",
    )
    p.add_argument(
        "--corner-dlt-weight",
        type=float,
        default=20.0,
        help="Weight of the 5-point DLT reprojection term in global corner scoring",
    )
    p.add_argument(
        "--corner-chiral-weight",
        type=float,
        default=2.0,
        help="Weight of the chirality term in global corner scoring",
    )
    p.add_argument(
        "--corner-anchor-weight",
        type=float,
        default=0.5,
        help="Weight of the distance-to-decoded-point regularizer in global corner scoring",
    )
    p.add_argument(
        "--no-draw-scores",
        action="store_true",
        help="Hide per-keypoint confidence text in saved visualizations",
    )
    p.add_argument(
        "--vis-corner-debug",
        action="store_true",
        help="Save corner-refine debug comparison image; with --vis-heatmap also overlays candidates and picked peaks",
    )
    p.add_argument(
        "--vis-heatmap",
        action="store_true",
        help="Overlay predicted heatmaps on output images (requires --output-dir)",
    )
    p.add_argument(
        "--heatmap-kpt",
        type=int,
        default=-1,
        help="Keypoint index to display in heatmap overlay (-1 = max across all keypoints)",
    )
    p.add_argument(
        "--heatmap-alpha",
        type=float,
        default=0.5,
        help="Heatmap blend alpha (0=no overlay, 1=full heatmap). Default 0.5",
    )
    return p.parse_args()


def main() -> None:
    import cv2

    args = _parse_args()
    if args.mode == "rtmo":
        missing = [
            name
            for name in ("rtmo_config", "rtmo_checkpoint")
            if getattr(args, name) is None
        ]
        if missing:
            raise ValueError(
                "--mode rtmo requires: "
                + ", ".join("--" + name.replace("_", "-") for name in missing)
            )
        if args.vis_heatmap:
            print("[WARN] --vis-heatmap is ignored for RTMO mode.")
        if args.vis_corner_debug:
            print("[WARN] --vis-corner-debug is ignored for RTMO mode.")
        pipe = RtmoPipeline(
            rtmo_config=args.rtmo_config,
            rtmo_checkpoint=args.rtmo_checkpoint,
            device=args.device,
            score_thr=args.rtmo_score_thr,
        )
    else:
        missing = [
            name
            for name in ("yolo_weights", "hrnet_config", "hrnet_checkpoint")
            if getattr(args, name) is None
        ]
        if missing:
            raise ValueError(
                "--mode hrnet requires: "
                + ", ".join("--" + name.replace("_", "-") for name in missing)
            )
        pipe = PillarHrnetPipeline(
            yolo_weights=args.yolo_weights,
            hrnet_config=args.hrnet_config,
            hrnet_checkpoint=args.hrnet_checkpoint,
            device=args.device,
            pillar_class_id=args.pillar_class_id,
            bbox_class_id=args.bbox_class_id,
            yolo_conf=args.yolo_conf,
            crop_margin=args.crop_margin,
            corner_refine=not args.no_corner_refine,
            corner_topk=args.corner_topk,
            corner_peak_rel_thr=args.corner_peak_rel_thr,
            corner_peak_abs_thr=args.corner_peak_abs_thr,
            corner_dlt_weight=args.corner_dlt_weight,
            corner_chiral_weight=args.corner_chiral_weight,
            corner_anchor_weight=args.corner_anchor_weight,
        )
        print("bbox_class_id:", pipe.bbox_class_id)
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    for image_path_str in args.images:
        image_path = Path(image_path_str)
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Cannot read image: {image_path}, skipping")
            continue

        if args.mode == "hrnet" and args.vis_heatmap:
            r, pose_samples, xyxy_exp = pipe.predict_with_heatmaps(
                img, image_id=image_path
            )
        else:
            r = pipe.predict_numpy(img, image_id=image_path)
            pose_samples, xyxy_exp = [], np.zeros((0, 4), dtype=np.float32)

        print("---")
        print("image:", r.image)
        print("shape_hw:", r.image_shape_hw)
        print("num_pillars:", len(r.pillars))
        result_names = (
            r.keypoint_names
            if r.keypoint_names is not None
            else [f"kp{i}" for i in range(len(r.pillars[0].keypoints_xy))]
            if r.pillars
            else []
        )
        for j, pil in enumerate(r.pillars):
            score_name = "bbox_score" if args.mode == "rtmo" else "yolo_score"
            print(f"  pillar[{j}] {score_name}={pil.yolo_score:.4f}")
            print("    bbox_xyxy:", pil.bbox_xyxy.tolist())
            print("    keypoints_xy:")
            for ki, (kname, kxy, ks) in enumerate(
                zip(result_names, pil.keypoints_xy.tolist(), pil.keypoint_scores.tolist())
            ):
                print(
                    f"      kp{ki} {kname}: ({kxy[0]:.1f}, {kxy[1]:.1f})  score={ks:.3f}"
                )
            if args.mode == "hrnet" and args.vis_corner_debug:
                debug = getattr(pil, "debug_info", None)
                if isinstance(debug, _CornerRefineDebug):
                    print(
                        "    corner_refine:",
                        f"status={debug.status}",
                        f"shift_max={debug.max_corner_shift_px:.1f}px",
                    )
                    if debug.base_combo is not None and debug.best_combo is not None:
                        print(
                            "      cost:",
                            f"E {debug.base_combo.total_cost:.3f}->{debug.best_combo.total_cost:.3f}",
                            f"DLT {debug.base_combo.dlt_err:.2e}->{debug.best_combo.dlt_err:.2e}",
                            f"anchor {debug.base_combo.anchor_cost:.3f}->{debug.best_combo.anchor_cost:.3f}",
                        )
                    if debug.second_combo is not None and debug.best_combo is not None:
                        print(
                            "      ranking:",
                            f"best={debug.best_combo.total_cost:.3f}",
                            f"second={debug.second_combo.total_cost:.3f}",
                        )

        if output_dir is not None:
            stem = image_path.stem
            # Keypoint overlay
            kpt_out = output_dir / f"{stem}_kpt{image_path.suffix}"
            save_pose_result_image(
                img,
                r,
                kpt_out,
                draw_scores=not args.no_draw_scores,
            )
            print("saved_kpt_vis:", kpt_out)
            if args.mode == "hrnet" and args.vis_corner_debug:
                corner_debug_img = draw_corner_refine_comparison(
                    img,
                    r,
                    draw_scores=not args.no_draw_scores,
                )
                corner_out = output_dir / f"{stem}_corner_debug{image_path.suffix}"
                cv2.imwrite(str(corner_out), corner_debug_img)
                print("saved_corner_debug_vis:", corner_out)
            # Heatmap overlay
            if args.mode == "hrnet" and args.vis_heatmap and len(pose_samples) > 0:
                grid_img = draw_heatmap_grid(
                    img,
                    pose_samples,
                    xyxy_exp,
                    pillar_results=r.pillars,
                    cell_size=192,
                    show_corner_debug=args.vis_corner_debug,
                    draw_scores=not args.no_draw_scores,
                    keypoint_names=r.keypoint_names,
                    skeleton=r.skeleton,
                    keypoint_colors=r.keypoint_colors,
                )
                hm_out = output_dir / f"{stem}_heatmap{image_path.suffix}"
                cv2.imwrite(str(hm_out), grid_img)
                print("saved_heatmap_vis:", hm_out)


if __name__ == "__main__":
    main()
