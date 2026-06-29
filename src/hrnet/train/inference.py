#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np


@dataclass
class PillarInstance:
    bbox_xyxy: np.ndarray
    yolo_score: float
    keypoints_xy: np.ndarray
    keypoint_scores: np.ndarray
    keypoint_visible_scores: Optional[np.ndarray] = None


@dataclass
class ImagePoseResult:
    image: Union[str, Path, np.ndarray]
    image_shape_hw: tuple[int, int]
    pillars: list[PillarInstance] = field(default_factory=list)
    keypoint_names: Optional[list[str]] = None
    skeleton: Optional[list[tuple[int, int]]] = None
    keypoint_colors: Optional[list[tuple[int, int, int]]] = None


_DEFAULT_KPT_NAMES = [
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
_DEFAULT_SKELETON = [
    (0, 1),
    (1, 3),
    (3, 2),
    (2, 0),
    (5, 6),
    (6, 10),
    (10, 11),
    (7, 8),
    (8, 9),
]
_DEFAULT_KPT_COLORS = [
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


def _rgb_to_bgr(color) -> tuple[int, int, int]:
    vals = list(np.asarray(color).reshape(-1)[:3].astype(int))
    if len(vals) < 3:
        vals = (vals + [255, 255, 255])[:3]
    return (int(vals[2]), int(vals[1]), int(vals[0]))


def _resolve_vis_spec(model) -> tuple[list[str], list[tuple[int, int]], list[tuple[int, int, int]]]:
    metainfo = getattr(model, "dataset_meta", None) or {}
    keypoint_info = metainfo.get("keypoint_info", {})
    skeleton_info = metainfo.get("skeleton_info", {})

    names = []
    for idx in sorted(keypoint_info):
        names.append(str(keypoint_info[idx].get("name", f"kp{idx}")))
    if not names:
        names = list(_DEFAULT_KPT_NAMES)

    name_to_idx = {name: idx for idx, name in enumerate(names)}
    skeleton = []
    for idx in sorted(skeleton_info):
        link = skeleton_info[idx].get("link")
        if not link or len(link) < 2:
            continue
        if link[0] in name_to_idx and link[1] in name_to_idx:
            skeleton.append((name_to_idx[link[0]], name_to_idx[link[1]]))
    if not skeleton:
        skeleton = list(_DEFAULT_SKELETON)

    colors = []
    for idx in sorted(keypoint_info):
        colors.append(_rgb_to_bgr(keypoint_info[idx].get("color", [255, 255, 255])))
    if not colors:
        colors = list(_DEFAULT_KPT_COLORS)
    return names, skeleton, colors


def _extract_class_bboxes(result, class_id: int, conf_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    if result.boxes is None or len(result.boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
    cls = boxes.cls.cpu().numpy().astype(np.int32)
    conf = boxes.conf.cpu().numpy().astype(np.float32)
    keep = (cls == int(class_id)) & (conf >= float(conf_threshold))
    return xyxy[keep], conf[keep]


def _pose_sample_to_instance(sample, bbox_xyxy: np.ndarray, yolo_score: float) -> PillarInstance:
    inst = sample.pred_instances
    keypoints_xy = np.asarray(inst.keypoints[0], dtype=np.float32)

    keypoint_scores = getattr(inst, "keypoint_scores", None)
    if keypoint_scores is None:
        keypoint_scores = np.ones((1, keypoints_xy.shape[0]), dtype=np.float32)
    keypoint_scores = np.asarray(keypoint_scores[0], dtype=np.float32)

    vis_scores = getattr(inst, "keypoints_visible", None)
    if vis_scores is not None:
        vis_scores = np.asarray(vis_scores[0], dtype=np.float32)

    return PillarInstance(
        bbox_xyxy=np.asarray(bbox_xyxy, dtype=np.float32),
        yolo_score=float(yolo_score),
        keypoints_xy=keypoints_xy,
        keypoint_scores=keypoint_scores,
        keypoint_visible_scores=vis_scores,
    )


def draw_pose_result(image_bgr: np.ndarray, result: ImagePoseResult, draw_scores: bool = True) -> np.ndarray:
    import cv2

    out = image_bgr.copy()
    names = result.keypoint_names or _DEFAULT_KPT_NAMES
    skeleton = result.skeleton or _DEFAULT_SKELETON
    colors = result.keypoint_colors or _DEFAULT_KPT_COLORS

    for pillar in result.pillars:
        x0, y0, x1, y1 = np.round(pillar.bbox_xyxy).astype(int)
        cv2.rectangle(out, (x0, y0), (x1, y1), (80, 220, 80), 4, cv2.LINE_AA)

        for a, b in skeleton:
            if a >= len(pillar.keypoints_xy) or b >= len(pillar.keypoints_xy):
                continue
            ax, ay = np.round(pillar.keypoints_xy[a]).astype(int)
            bx, by = np.round(pillar.keypoints_xy[b]).astype(int)
            cv2.line(out, (ax, ay), (bx, by), (220, 220, 220), 2, cv2.LINE_AA)

        for idx, (x, y) in enumerate(np.round(pillar.keypoints_xy).astype(int)):
            color = colors[idx % len(colors)]
            cv2.circle(out, (x, y), 7, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(out, (x, y), 5, color, -1, cv2.LINE_AA)
            label = names[idx] if idx < len(names) else f"kp{idx}"
            cv2.putText(
                out,
                label,
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

        if draw_scores:
            cv2.putText(
                out,
                f"{pillar.yolo_score:.2f}",
                (x0, max(20, y0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return out


def save_pose_result_image(
    image_bgr: np.ndarray,
    result: ImagePoseResult,
    output_path: Union[str, Path],
    draw_scores: bool = True,
) -> Path:
    import cv2

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), draw_pose_result(image_bgr, result, draw_scores=draw_scores))
    return output_path


class PillarHrnetPipeline:
    def __init__(
        self,
        yolo_weights: Union[str, Path],
        hrnet_config: Union[str, Path],
        hrnet_checkpoint: Union[str, Path],
        device: str = "cuda:0",
        bbox_class_id: int = 1,
        yolo_conf: float = 0.25,
        crop_margin: float = 0.05,
    ) -> None:
        from ultralytics import YOLO
        from mmpose.apis import init_model

        self.device = device
        self.bbox_class_id = int(bbox_class_id)
        self.yolo_conf = float(yolo_conf)
        self.crop_margin = float(crop_margin)

        self._yolo = YOLO(str(yolo_weights))
        self._pose = init_model(str(hrnet_config), str(hrnet_checkpoint), device=device)
        self._vis_keypoint_names, self._vis_skeleton, self._vis_keypoint_colors = _resolve_vis_spec(self._pose)

    def _run_pose_inference(self, image_bgr: np.ndarray, xyxy: np.ndarray):
        from mmpose.apis import inference_topdown

        return inference_topdown(self._pose, image_bgr, xyxy, bbox_format="xyxy")

    def predict_numpy(
        self,
        image_bgr: np.ndarray,
        image_id: Optional[Union[str, Path]] = None,
    ) -> ImagePoseResult:
        h, w = image_bgr.shape[:2]
        ref: Union[str, Path, np.ndarray] = image_id if image_id is not None else image_bgr

        yolo_results = self._yolo(image_bgr, verbose=False, conf=self.yolo_conf)
        if not yolo_results:
            return ImagePoseResult(
                image=ref,
                image_shape_hw=(h, w),
                keypoint_names=list(self._vis_keypoint_names),
                skeleton=list(self._vis_skeleton),
                keypoint_colors=list(self._vis_keypoint_colors),
            )

        xyxy, box_scores = _extract_class_bboxes(
            yolo_results[0],
            class_id=self.bbox_class_id,
            conf_threshold=self.yolo_conf,
        )
        if xyxy.shape[0] == 0:
            return ImagePoseResult(
                image=ref,
                image_shape_hw=(h, w),
                keypoint_names=list(self._vis_keypoint_names),
                skeleton=list(self._vis_skeleton),
                keypoint_colors=list(self._vis_keypoint_colors),
            )

        if self.crop_margin > 0:
            ws = (xyxy[:, 2] - xyxy[:, 0]) * self.crop_margin
            hs = (xyxy[:, 3] - xyxy[:, 1]) * self.crop_margin
            xyxy = xyxy.copy()
            xyxy[:, 0] = np.clip(xyxy[:, 0] - ws, 0, w)
            xyxy[:, 1] = np.clip(xyxy[:, 1] - hs, 0, h)
            xyxy[:, 2] = np.clip(xyxy[:, 2] + ws, 0, w)
            xyxy[:, 3] = np.clip(xyxy[:, 3] + hs, 0, h)

        pose_samples = self._run_pose_inference(image_bgr, xyxy)
        pillars = [
            _pose_sample_to_instance(sample, xyxy[idx], float(box_scores[idx]))
            for idx, sample in enumerate(pose_samples)
        ]

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

        path = Path(path)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return self.predict_numpy(image, image_id=path)

    def predict_images(self, paths: Sequence[Union[str, Path]]) -> list[ImagePoseResult]:
        return [self.predict_image(path) for path in paths]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO detect + LiteHRNet image inference")
    parser.add_argument("--yolo-weights", required=True)
    parser.add_argument("--hrnet-config", required=True)
    parser.add_argument("--hrnet-checkpoint", required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bbox-class-id", type=int, default=1)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--crop-margin", type=float, default=0.05)
    parser.add_argument("--no-draw-scores", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = PillarHrnetPipeline(
        yolo_weights=args.yolo_weights,
        hrnet_config=args.hrnet_config,
        hrnet_checkpoint=args.hrnet_checkpoint,
        device=args.device,
        bbox_class_id=args.bbox_class_id,
        yolo_conf=args.yolo_conf,
        crop_margin=args.crop_margin,
    )

    import cv2

    for image_path in [Path(path) for path in args.images]:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        result = pipeline.predict_numpy(image, image_id=image_path)
        output_path = output_dir / f"{image_path.stem}_kpt{image_path.suffix}"
        save_pose_result_image(
            image,
            result,
            output_path,
            draw_scores=not args.no_draw_scores,
        )
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
