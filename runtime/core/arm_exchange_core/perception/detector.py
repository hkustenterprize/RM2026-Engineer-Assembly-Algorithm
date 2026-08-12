from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml


@dataclass(frozen=True, slots=True)
class KeypointObservation:
    keypoints_2d: np.ndarray
    keypoint_names: tuple[str, ...]
    confidences: np.ndarray | None = None
    bbox_xyxy: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        keypoints = np.asarray(self.keypoints_2d, dtype=float)
        names = tuple(self.keypoint_names)
        if keypoints.ndim != 2 or keypoints.shape[1] != 2 or len(names) != len(keypoints):
            raise ValueError("keypoints and names must have shapes (N, 2) and (N,)")
        confidences = None if self.confidences is None else np.asarray(self.confidences, dtype=float)
        if confidences is not None and confidences.shape != (len(keypoints),):
            raise ValueError(f"confidences must have shape ({len(keypoints)},), got {confidences.shape}")
        bbox = None if self.bbox_xyxy is None else np.asarray(self.bbox_xyxy, dtype=float)
        if bbox is not None and bbox.shape != (4,):
            raise ValueError(f"bbox_xyxy must have shape (4,), got {bbox.shape}")
        object.__setattr__(self, "keypoints_2d", keypoints)
        object.__setattr__(self, "keypoint_names", names)
        object.__setattr__(self, "confidences", confidences)
        object.__setattr__(self, "bbox_xyxy", bbox)
        object.__setattr__(self, "metadata", dict(self.metadata))


_DEFAULT_KEYPOINT_NAMES = ("TL", "TR", "BL", "BR", "ring")
_DEFAULT_SKELETON = ((0, 1), (1, 3), (3, 2), (2, 0))


@dataclass(slots=True)
class DetectedInstance:
    bbox_xyxy: np.ndarray
    yolo_score: float
    keypoints_xy: np.ndarray
    keypoint_scores: np.ndarray
    bbox_class_id: int
    bbox_source: str
    debug_info: object | None = field(default=None, repr=False)


@dataclass(slots=True)
class KeypointDetectionResult:
    image: str | Path | np.ndarray
    image_shape_hw: tuple[int, int]
    instances: list[DetectedInstance] = field(default_factory=list)
    keypoint_names: list[str] | None = None
    skeleton: list[tuple[int, int]] | None = None
    yolo_detections: list[dict] = field(default_factory=list)


@dataclass(slots=True)
class YoloDetections:
    xyxy: np.ndarray
    class_ids: np.ndarray
    scores: np.ndarray
    names: dict[int, str]


def _resolve_openvino_model_path(model_path: str | Path, *, preferred_stem: str | None = None) -> Path:
    path = Path(model_path)
    if path.is_file():
        if path.suffix.lower() != ".xml":
            raise ValueError(f"OpenVINO model path must point to an .xml file, got {path}")
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"OpenVINO model path does not exist: {path}")

    xml_files = sorted(path.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"OpenVINO model directory has no .xml files: {path}")
    if preferred_stem is not None:
        preferred = [item for item in xml_files if item.stem == preferred_stem]
        if len(preferred) == 1:
            return preferred[0]
    if len(xml_files) != 1:
        raise ValueError(
            f"OpenVINO model directory must contain exactly one .xml file, got {len(xml_files)}: {path}"
        )
    return xml_files[0]


def _normalize_openvino_device(device: str) -> str:
    raw = str(device or "CPU")
    lowered = raw.lower()
    if lowered in {"cpu", "auto"}:
        return raw.upper()
    if lowered.startswith("cuda") or lowered == "0":
        return "GPU"
    return raw.upper()


def _as_bgr_image(image_bgr: np.ndarray) -> np.ndarray:
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"two-stage detector expects HxWx3 image, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def _extract_class_bboxes(
    detections: YoloDetections,
    *,
    class_id: int,
    conf_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    if detections.xyxy.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    mask = (detections.class_ids == int(class_id)) & (detections.scores >= float(conf_threshold))
    xyxy = detections.xyxy[mask]
    conf = detections.scores[mask]
    if xyxy.size == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    order = np.argsort(-conf)
    return xyxy[order].astype(np.float32), conf[order].astype(np.float32)


def _extract_all_detections(
    detections_batch: YoloDetections,
    *,
    conf_threshold: float,
) -> list[dict]:
    if detections_batch.xyxy.shape[0] == 0:
        return []

    detections = []
    for bbox, class_id, score in zip(
        detections_batch.xyxy,
        detections_batch.class_ids,
        detections_batch.scores,
        strict=True,
    ):
        if float(score) < float(conf_threshold):
            continue
        detections.append(
            {
                "bbox_xyxy": np.asarray(bbox, dtype=np.float32).tolist(),
                "class_id": int(class_id),
                "class_name": str(detections_batch.names.get(int(class_id), int(class_id))),
                "score": float(score),
            }
        )
    detections.sort(key=lambda item: float(item["score"]), reverse=True)
    return detections


class OpenVinoYoloBackend:
    def __init__(self, model_path: str | Path, *, device: str) -> None:
        from openvino import Core

        self._model_path = _resolve_openvino_model_path(model_path, preferred_stem="best")
        core = Core()
        model = core.read_model(str(self._model_path))
        self._compiled = core.compile_model(model, _normalize_openvino_device(device))
        self._input = self._compiled.inputs[0]
        self._output = self._compiled.outputs[0]
        shape = tuple(int(v) for v in self._input.shape)
        if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
            raise ValueError(f"OpenVINO YOLO expects NCHW batch-1 input, got {shape}")
        self._input_h = shape[2]
        self._input_w = shape[3]
        self._names = self._load_names(self._model_path)

    def predict(self, image_bgr: np.ndarray, *, conf: float) -> YoloDetections:
        blob, scale, pad_x, pad_y = self._preprocess(image_bgr)
        output = np.asarray(self._compiled([blob])[self._output], dtype=np.float32).reshape(-1, 6)
        output = output[output[:, 4] >= float(conf)]
        if output.shape[0] == 0:
            return _empty_yolo_detections(names=self._names)
        xyxy = output[:, :4].copy()
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - pad_x) / scale
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - pad_y) / scale
        height, width = image_bgr.shape[:2]
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0.0, float(width))
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0.0, float(height))
        class_ids = output[:, 5].astype(np.int32)
        scores = output[:, 4].astype(np.float32)
        order = np.argsort(-scores)
        return YoloDetections(
            xyxy=xyxy[order].astype(np.float32),
            class_ids=class_ids[order],
            scores=scores[order],
            names=self._names,
        )

    def _preprocess(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        import cv2

        height, width = image_bgr.shape[:2]
        scale = min(float(self._input_w) / float(width), float(self._input_h) / float(height))
        resized_w = int(round(float(width) * scale))
        resized_h = int(round(float(height) * scale))
        resized = cv2.resize(image_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self._input_h, self._input_w, 3), 114, dtype=np.uint8)
        pad_x = (self._input_w - resized_w) // 2
        pad_y = (self._input_h - resized_h) // 2
        canvas[pad_y:pad_y + resized_h, pad_x:pad_x + resized_w] = resized
        rgb = canvas[..., ::-1]
        blob = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return blob, scale, float(pad_x), float(pad_y)

    @staticmethod
    def _load_names(model_path: Path) -> dict[int, str]:
        metadata_path = model_path.with_name("metadata.yaml")
        if not metadata_path.exists():
            return {0: "local_target", 1: "exchange"}
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = yaml.safe_load(f) or {}
        names = metadata.get("names", {})
        if not isinstance(names, dict):
            return {0: "local_target", 1: "exchange"}
        return {int(k): str(v) for k, v in names.items()}


def _empty_yolo_detections(names: dict[int, str] | None = None) -> YoloDetections:
    return YoloDetections(
        xyxy=np.zeros((0, 4), dtype=np.float32),
        class_ids=np.zeros((0,), dtype=np.int32),
        scores=np.zeros((0,), dtype=np.float32),
        names={} if names is None else dict(names),
    )


def _third_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direction = a - b
    return b + np.array([-direction[1], direction[0]], dtype=np.float32)


def _bbox_to_square_affine(
    bbox_xyxy: Sequence[float],
    input_size: tuple[int, int],
    padding: float,
) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = [float(value) for value in bbox_xyxy]
    scale = np.array([max(x2 - x1, 1.0), max(y2 - y1, 1.0)], dtype=np.float32)
    scale *= float(padding)
    input_w, input_h = input_size
    aspect = float(input_w) / float(input_h)
    if scale[0] > scale[1] * aspect:
        scale[1] = scale[0] / aspect
    else:
        scale[0] = scale[1] * aspect

    center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
    src = np.array(
        [
            center,
            center + np.array([-0.5 * scale[0], 0.0], dtype=np.float32),
            np.zeros(2, dtype=np.float32),
        ]
    )
    src[2] = _third_point(src[0], src[1])
    dst_center = np.array([input_w * 0.5, input_h * 0.5], dtype=np.float32)
    dst = np.array(
        [
            dst_center,
            dst_center + np.array([-0.5 * input_w, 0.0], dtype=np.float32),
            np.zeros(2, dtype=np.float32),
        ]
    )
    dst[2] = _third_point(dst[0], dst[1])
    return cv2.getAffineTransform(src, dst), cv2.getAffineTransform(dst, src)


def _decode_heatmaps(
    heatmaps: np.ndarray,
    input_size: tuple[int, int],
    inverse_affine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    num_keypoints, heatmap_h, heatmap_w = heatmaps.shape
    coordinates = np.zeros((num_keypoints, 2), dtype=np.float32)
    scores = np.zeros(num_keypoints, dtype=np.float32)
    input_w, input_h = input_size
    for index, heatmap in enumerate(heatmaps):
        y, x = divmod(int(np.argmax(heatmap)), heatmap_w)
        px, py = float(x), float(y)
        if 1 <= x < heatmap_w - 1 and 1 <= y < heatmap_h - 1:
            px += np.sign(float(heatmap[y, x + 1] - heatmap[y, x - 1])) * 0.25
            py += np.sign(float(heatmap[y + 1, x] - heatmap[y - 1, x])) * 0.25
        input_point = np.array(
            [(px + 0.5) * input_w / heatmap_w, (py + 0.5) * input_h / heatmap_h, 1.0],
            dtype=np.float32,
        )
        coordinates[index] = inverse_affine @ input_point
        scores[index] = float(heatmap[y, x])
    return coordinates, scores


def _select_keypoints_by_name(
    keypoints_xy: np.ndarray,
    scores: np.ndarray,
    source_names: Sequence[str] | None,
    target_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    keypoints = np.asarray(keypoints_xy, dtype=float)
    confidences = np.asarray(scores, dtype=float).reshape(-1)
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError(f"detector keypoints must be Nx2, got {keypoints.shape}")
    if confidences.shape[0] != keypoints.shape[0]:
        raise ValueError("detector keypoint scores length must match keypoints")

    names = tuple(str(v) for v in (source_names or ()))
    wanted = tuple(str(v) for v in target_names)
    if names:
        index_by_name = {name: idx for idx, name in enumerate(names)}
        missing = [name for name in wanted if name not in index_by_name]
        if missing:
            raise ValueError(f"detector output missing keypoints: {missing}")
        indices = [index_by_name[name] for name in wanted]
    else:
        if keypoints.shape[0] < len(wanted):
            raise ValueError(
                f"detector produced {keypoints.shape[0]} keypoints, need {len(wanted)}"
            )
        indices = list(range(len(wanted)))

    return keypoints[indices], confidences[indices], wanted


def _keypoint_bbox(
    keypoints_xy: np.ndarray,
    scores: np.ndarray,
    *,
    score_threshold: float,
    min_keypoints: int,
    width: int,
    height: int,
) -> np.ndarray | None:
    keypoints = np.asarray(keypoints_xy, dtype=np.float32)
    confidences = np.asarray(scores, dtype=np.float32).reshape(-1)
    visible = confidences >= float(score_threshold)
    if int(np.count_nonzero(visible)) < int(min_keypoints):
        return None

    visible_points = keypoints[visible]
    x1, y1 = np.min(visible_points, axis=0)
    x2, y2 = np.max(visible_points, axis=0)
    if not np.isfinite([x1, y1, x2, y2]).all():
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return np.asarray(
        [
            np.clip(x1, 0.0, float(width)),
            np.clip(y1, 0.0, float(height)),
            np.clip(x2, 0.0, float(width)),
            np.clip(y2, 0.0, float(height)),
        ],
        dtype=np.float32,
    )


class OpenVinoLiteHrnetBackend:
    def __init__(
        self,
        *,
        model_path: str | Path,
        input_size: Sequence[int],
        mean: Sequence[float],
        std: Sequence[float],
        device: str,
        bbox_padding: float,
    ) -> None:
        from openvino import Core

        self.input_size = tuple(int(value) for value in input_size)
        if len(self.input_size) != 2:
            raise ValueError("LiteHRNet input_size must contain width and height")
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        if self.mean.shape != (3,) or self.std.shape != (3,):
            raise ValueError("LiteHRNet mean and std must contain three channels")
        self.bbox_padding = float(bbox_padding)

        self._model_path = _resolve_openvino_model_path(model_path, preferred_stem="hrnet_heatmap")
        core = Core()
        model = core.read_model(str(self._model_path))
        self._compiled = core.compile_model(model, _normalize_openvino_device(device))
        self._input = self._compiled.inputs[0]
        self._output = self._compiled.outputs[0]
        self._validate_model_shape()

    def _validate_model_shape(self) -> None:
        input_shape = self._input.partial_shape
        if len(input_shape) != 4:
            raise ValueError(f"OpenVINO LiteHRNet expects NCHW input, got {input_shape}")
        channels = input_shape[1]
        if channels.is_static and int(channels.get_length()) != 3:
            raise ValueError(f"OpenVINO LiteHRNet expects 3 input channels, got {input_shape}")
        input_w, input_h = self.input_size
        height = input_shape[2]
        width = input_shape[3]
        if height.is_static and int(height.get_length()) != input_h:
            raise ValueError(f"OpenVINO LiteHRNet input height mismatch: model={height}, config={input_h}")
        if width.is_static and int(width.get_length()) != input_w:
            raise ValueError(f"OpenVINO LiteHRNet input width mismatch: model={width}, config={input_w}")

    def predict_bbox(self, image_bgr: np.ndarray, bbox_xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"image_bgr must be HxWx3, got {image_bgr.shape}")

        affine, inverse = _bbox_to_square_affine(
            bbox_xyxy, self.input_size, padding=self.bbox_padding
        )
        input_w, input_h = self.input_size
        crop = cv2.warpAffine(
            image_bgr,
            affine,
            (input_w, input_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        crop = crop[..., ::-1]
        normalized = (crop.astype(np.float32) - self.mean) / self.std
        blob = normalized.transpose(2, 0, 1)[None].astype(np.float32)
        output = self._compiled([blob])[self._output]
        heatmaps = np.asarray(output, dtype=np.float32)[0]
        return _decode_heatmaps(heatmaps, self.input_size, inverse)


class _YoloHRNetPipeline:
    """Two-stage YOLO bbox + local LiteHRNet keypoint inference."""

    def __init__(
        self,
        *,
        yolo_model: str | Path,
        hrnet_model: str | Path,
        yolo_device: str = "GPU",
        hrnet_device: str = "CPU",
        hrnet_input_size: Sequence[int] = (256, 256),
        hrnet_mean: Sequence[float] = (123.675, 116.28, 103.53),
        hrnet_std: Sequence[float] = (58.395, 57.12, 57.375),
        bbox_class_id: int = 1,
        yolo_conf: float = 0.25,
        crop_margin: float = 0.0,
        bbox_padding: float = 1.25,
        keypoint_names: Sequence[str] = _DEFAULT_KEYPOINT_NAMES,
        fallback_score_threshold: float = 0.20,
        fallback_min_keypoints: int = 4,
        fallback_crop_margin: float = 0.5,
    ) -> None:
        self.yolo_conf = float(yolo_conf)
        self.crop_margin = float(crop_margin)
        self.bbox_class_id = int(bbox_class_id)
        self._yolo = OpenVinoYoloBackend(yolo_model, device=yolo_device)
        self.keypoint_names = [str(v) for v in keypoint_names]
        self.skeleton = [link for link in _DEFAULT_SKELETON if max(link) < len(self.keypoint_names)]
        self.fallback_score_threshold = float(fallback_score_threshold)
        self.fallback_min_keypoints = int(fallback_min_keypoints)
        self.fallback_crop_margin = float(fallback_crop_margin)
        self._hrnet = OpenVinoLiteHrnetBackend(
            model_path=hrnet_model,
            input_size=hrnet_input_size,
            mean=hrnet_mean,
            std=hrnet_std,
            device=hrnet_device,
            bbox_padding=float(bbox_padding),
        )

    def _crop_bboxes(
        self,
        xyxy: np.ndarray,
        *,
        width: int,
        height: int,
        margin: float,
    ) -> np.ndarray:
        if margin <= 0.0:
            return xyxy

        ws = (xyxy[:, 2] - xyxy[:, 0]) * margin
        hs = (xyxy[:, 3] - xyxy[:, 1]) * margin
        expanded = xyxy.copy()
        expanded[:, 0] = np.clip(xyxy[:, 0] - ws, 0, width)
        expanded[:, 1] = np.clip(xyxy[:, 1] - hs, 0, height)
        expanded[:, 2] = np.clip(xyxy[:, 2] + ws, 0, width)
        expanded[:, 3] = np.clip(xyxy[:, 3] + hs, 0, height)
        return expanded

    def predict_numpy(
        self,
        image_bgr: np.ndarray,
        image_id: str | Path | None = None,
    ) -> KeypointDetectionResult:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(f"image_bgr must be HxWx3, got {image_bgr.shape}")

        height, width = image_bgr.shape[:2]
        empty = KeypointDetectionResult(
            image=image_id if image_id is not None else image_bgr,
            image_shape_hw=(height, width),
            instances=[],
            keypoint_names=list(self.keypoint_names),
            skeleton=list(self.skeleton),
        )

        yolo_results = self._yolo.predict(image_bgr, conf=self.yolo_conf)
        if yolo_results.xyxy.shape[0] == 0:
            return empty
        yolo_detections = _extract_all_detections(
            yolo_results,
            conf_threshold=self.yolo_conf,
        )

        xyxy, box_scores = _extract_class_bboxes(
            yolo_results,
            class_id=self.bbox_class_id,
            conf_threshold=self.yolo_conf,
        )
        bbox_class_id = self.bbox_class_id
        bbox_source = "yolo"
        crop_margin = self.crop_margin
        if xyxy.shape[0] == 0:
            full_image_bbox = np.asarray([0.0, 0.0, float(width), float(height)], dtype=np.float32)
            coarse_keypoints_xy, coarse_keypoint_scores = self._hrnet.predict_bbox(
                image_bgr,
                full_image_bbox,
            )
            coarse_bbox = _keypoint_bbox(
                coarse_keypoints_xy,
                coarse_keypoint_scores,
                score_threshold=self.fallback_score_threshold,
                min_keypoints=self.fallback_min_keypoints,
                width=width,
                height=height,
            )
            if coarse_bbox is None:
                empty.yolo_detections = yolo_detections
                return empty
            xyxy = coarse_bbox.reshape(1, 4)
            box_scores = np.asarray([float(np.mean(coarse_keypoint_scores))], dtype=np.float32)
            bbox_class_id = -1
            bbox_source = "hrnet_fallback"
            crop_margin = self.fallback_crop_margin

        xyxy = self._crop_bboxes(xyxy, width=width, height=height, margin=crop_margin)

        instances = []
        for idx, bbox in enumerate(xyxy):
            keypoints_xy, keypoint_scores = self._hrnet.predict_bbox(image_bgr, bbox)
            instances.append(
                DetectedInstance(
                    bbox_xyxy=np.asarray(bbox, dtype=np.float32),
                    yolo_score=float(box_scores[idx]),
                    keypoints_xy=np.asarray(keypoints_xy, dtype=np.float32),
                    keypoint_scores=np.asarray(keypoint_scores, dtype=np.float32),
                    bbox_class_id=int(bbox_class_id),
                    bbox_source=str(bbox_source),
                )
            )
        return KeypointDetectionResult(
            image=image_id if image_id is not None else image_bgr,
            image_shape_hw=(height, width),
            instances=instances,
            keypoint_names=list(self.keypoint_names),
            skeleton=list(self.skeleton),
            yolo_detections=yolo_detections,
        )


class YoloHRNetBackend:
    """YOLO bbox + HRNet top-down keypoint detector adapter."""

    def __init__(
        self,
        *,
        yolo_model: str | Path,
        hrnet_model: str | Path,
        yolo_device: str = "GPU",
        hrnet_device: str = "CPU",
        hrnet_input_size: Sequence[int] = (256, 256),
        hrnet_mean: Sequence[float] = (123.675, 116.28, 103.53),
        hrnet_std: Sequence[float] = (58.395, 57.12, 57.375),
        bbox_class_id: int = 1,
        yolo_conf: float = 0.25,
        crop_margin: float = 0.5,
        bbox_padding: float = 1.25,
        keypoint_names: Sequence[str] = _DEFAULT_KEYPOINT_NAMES,
        min_keypoint_score: float = 0.0,
        drop_low_confidence_keypoints: bool = False,
        visibility_score_threshold: float | None = None,
        min_visible_keypoints: int = 4,
        fallback_score_threshold: float = 0.20,
        fallback_min_keypoints: int = 4,
        fallback_crop_margin: float = 0.5,
    ) -> None:
        self.keypoint_names = tuple(str(v) for v in keypoint_names)
        self.min_keypoint_score = float(min_keypoint_score)
        self.drop_low_confidence_keypoints = bool(drop_low_confidence_keypoints)
        self.visibility_score_threshold = (
            float(visibility_score_threshold)
            if visibility_score_threshold is not None
            else self.min_keypoint_score
        )
        self.min_visible_keypoints = int(min_visible_keypoints)
        self._pipeline = _YoloHRNetPipeline(
            yolo_model=yolo_model,
            hrnet_model=hrnet_model,
            yolo_device=yolo_device,
            hrnet_device=hrnet_device,
            hrnet_input_size=hrnet_input_size,
            hrnet_mean=hrnet_mean,
            hrnet_std=hrnet_std,
            bbox_class_id=int(bbox_class_id),
            yolo_conf=float(yolo_conf),
            crop_margin=float(crop_margin),
            bbox_padding=float(bbox_padding),
            keypoint_names=self.keypoint_names,
            fallback_score_threshold=float(fallback_score_threshold),
            fallback_min_keypoints=int(fallback_min_keypoints),
            fallback_crop_margin=float(fallback_crop_margin),
        )

    def detect(
        self,
        image_bgr: np.ndarray,
        *,
        image_id: str | None = None,
    ) -> KeypointObservation:
        image_bgr = _as_bgr_image(image_bgr)
        result = self._pipeline.predict_numpy(image_bgr, image_id=image_id)
        if not result.instances:
            raise RuntimeError("YOLO-HRNet backend found no target instance")

        instance = max(result.instances, key=lambda item: float(item.yolo_score))
        keypoints, scores, names = _select_keypoints_by_name(
            instance.keypoints_xy,
            instance.keypoint_scores,
            result.keypoint_names,
            self.keypoint_names,
        )
        all_keypoints = keypoints.copy()
        all_scores = scores.copy()
        all_names = tuple(names)
        if self.drop_low_confidence_keypoints:
            visible_mask = scores >= self.visibility_score_threshold
            keypoints = keypoints[visible_mask]
            scores = scores[visible_mask]
            names = tuple(name for name, visible in zip(names, visible_mask, strict=True) if visible)
            if len(names) < self.min_visible_keypoints:
                raise RuntimeError(
                    "YOLO-HRNet backend visible keypoints below threshold: "
                    f"visible={len(names)}, required={self.min_visible_keypoints}"
                )
        elif np.any(scores < self.min_keypoint_score):
            raise RuntimeError(
                "YOLO-HRNet backend keypoint score below threshold: "
                f"min={float(np.min(scores)):.3f}, threshold={self.min_keypoint_score:.3f}"
            )

        observation = KeypointObservation(
            keypoints_2d=keypoints,
            keypoint_names=names,
            confidences=scores,
            bbox_xyxy=instance.bbox_xyxy,
            metadata={
                "source": "yolo_hrnet",
                "yolo_score": float(instance.yolo_score),
                "bbox_class_id": int(instance.bbox_class_id),
                "bbox_source": str(instance.bbox_source),
                "yolo_detections": tuple(result.yolo_detections),
                "all_keypoint_names": tuple(result.keypoint_names or ()),
                "status_keypoint_names": all_names,
                "status_keypoints_2d": tuple(
                    (float(point[0]), float(point[1])) for point in all_keypoints
                ),
                "status_keypoint_scores": tuple(float(score) for score in all_scores),
                "visibility_score_threshold": float(self.visibility_score_threshold),
                "selected_keypoint_names": tuple(names),
            },
        )
        return observation
