from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from arm_exchange_interfaces.msg import ArmCtrlGimbalControlMsg
from arm_exchange_core import load_config
from arm_exchange_core.transform import quaternions_from_rotations
from arm_exchange_core.perception.detector import YoloHRNetBackend
from arm_exchange_core.perception.pose import PnPEstimator
from .ros_utils import (
    image_to_bgr,
    pose_stamped_from_transform,
    transform_from_pose_stamped,
    transform_pose_stamped,
)


_KEYPOINT_COLORS = (
    (0, 255, 0),
    (0, 200, 255),
    (255, 160, 0),
    (255, 0, 255),
    (0, 120, 255),
    (180, 255, 0),
    (255, 80, 80),
    (120, 120, 255),
    (80, 255, 180),
    (255, 220, 80),
    (160, 80, 255),
    (80, 180, 255),
)


def _camera_matrix_from_info(msg: CameraInfo) -> np.ndarray:
    return np.asarray(msg.k, dtype=float).reshape(3, 3)


def _dist_coeffs_from_info(msg: CameraInfo) -> np.ndarray:
    return np.asarray(msg.d, dtype=float).reshape(-1)


def _quat_wxyz_to_rpy_deg(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=float).reshape(4)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_arg = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(pitch_arg)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.degrees([roll, pitch, yaw])


def _pose_line(prefix: str, transform: np.ndarray) -> str:
    transform = np.asarray(transform, dtype=float)
    t = transform[:3, 3]
    quat = quaternions_from_rotations(transform[None, :3, :3])[0]
    rpy = _quat_wxyz_to_rpy_deg(quat)
    return (
        f"{prefix} t=[{t[0]:+.3f},{t[1]:+.3f},{t[2]:+.3f}] "
        f"Rrpy=[{rpy[0]:+.1f},{rpy[1]:+.1f},{rpy[2]:+.1f}]deg"
    )


def _draw_text_panel(
    overlay: np.ndarray,
    lines: tuple[str, ...],
    *,
    x: int,
    y: int,
    width: int,
) -> None:
    import cv2

    if not lines:
        return
    pad = 8
    line_h = 18
    panel_h = pad * 2 + line_h * len(lines)
    panel = overlay.copy()
    cv2.rectangle(panel, (x, y), (x + width, y + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.55, overlay, 0.45, 0, overlay)
    cv2.rectangle(overlay, (x, y), (x + width, y + panel_h), (220, 220, 220), 1)
    for row, line in enumerate(lines):
        cv2.putText(
            overlay,
            line,
            (x + pad, y + pad + 14 + row * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _draw_yolo_detections(overlay: np.ndarray, detections) -> None:
    import cv2

    for det in detections or ():
        bbox = det.get("bbox_xyxy")
        if bbox is None:
            continue
        x1, y1, x2, y2 = np.round(np.asarray(bbox, dtype=float)).astype(int).tolist()
        cls_id = det.get("class_id", "?")
        score = float(det.get("score", 0.0))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (180, 180, 180), 1)
        cv2.putText(
            overlay,
            f"yolo:{cls_id} {score:.2f}",
            (x1, max(14, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )


def _draw_keypoint_status_panel(overlay: np.ndarray, observation) -> None:
    import cv2

    metadata = dict(observation.metadata or {})
    names = tuple(metadata.get("status_keypoint_names", observation.keypoint_names))
    scores = metadata.get("status_keypoint_scores", observation.confidences)
    scores = None if scores is None else np.asarray(scores, dtype=float).reshape(-1)
    selected = set(metadata.get("selected_keypoint_names", observation.keypoint_names))
    threshold = float(metadata.get("visibility_score_threshold", -np.inf))

    lines = ["keypoints"]
    if not names:
        lines.append("MISSING no output")
    for index, name in enumerate(names):
        if scores is None or index >= len(scores):
            lines.append(f"{name}: MISSING")
            continue
        score = float(scores[index])
        state = "OK" if str(name) in selected and score >= threshold else "LOW"
        lines.append(f"{name}: {score:.2f} {state}")

    line_h = 18
    pad = 8
    panel_w = 230
    panel_h = pad * 2 + line_h * len(lines)
    h, w = overlay.shape[:2]
    x0 = max(0, w - panel_w - 10)
    y0 = max(0, h - panel_h - 10)
    x1 = min(w, x0 + panel_w)
    y1 = min(h, y0 + panel_h)

    panel = overlay.copy()
    cv2.rectangle(panel, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(panel, 0.55, overlay, 0.45, 0, overlay)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (220, 220, 220), 1)

    for row, line in enumerate(lines):
        y = y0 + pad + 14 + row * line_h
        color = (255, 255, 255)
        if " LOW" in line or "MISSING" in line:
            color = (0, 80, 255)
        elif " OK" in line:
            color = (80, 255, 80)
        cv2.putText(overlay, line, (x0 + pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _draw_detection_overlay(
    image_bgr: np.ndarray,
    observation,
    *,
    title: str,
    pose_lines: tuple[str, ...] = (),
) -> np.ndarray:
    import cv2

    overlay = np.asarray(image_bgr).copy()
    if observation is None:
        cv2.putText(
            overlay,
            title,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return overlay

    metadata = dict(observation.metadata or {})
    _draw_yolo_detections(overlay, metadata.get("yolo_detections", ()))

    if observation.bbox_xyxy is not None:
        x1, y1, x2, y2 = np.round(observation.bbox_xyxy).astype(int).tolist()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 255), 2)

    status_names = tuple(metadata.get("status_keypoint_names", observation.keypoint_names))
    status_points = np.asarray(
        metadata.get("status_keypoints_2d", observation.keypoints_2d),
        dtype=float,
    )
    status_scores = metadata.get("status_keypoint_scores", observation.confidences)
    status_scores = None if status_scores is None else np.asarray(status_scores, dtype=float).reshape(-1)
    selected_names = set(metadata.get("selected_keypoint_names", observation.keypoint_names))
    threshold = float(metadata.get("visibility_score_threshold", -np.inf))

    for index, (name, point) in enumerate(zip(status_names, status_points, strict=True)):
        x, y = np.round(point).astype(int).tolist()
        score = None if status_scores is None else float(status_scores[index])
        is_low = score is not None and score < threshold
        is_selected = str(name) in selected_names
        color = (80, 80, 80) if is_low or not is_selected else _KEYPOINT_COLORS[index % len(_KEYPOINT_COLORS)]
        outline = (0, 0, 255) if is_low else (255, 255, 255)
        cv2.circle(overlay, (x, y), 5, color, -1)
        cv2.circle(overlay, (x, y), 8, outline, 1)
        cv2.putText(overlay, str(name), (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    cv2.putText(
        overlay,
        title,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    _draw_text_panel(overlay, pose_lines, x=10, y=42, width=560)
    _draw_keypoint_status_panel(overlay, observation)
    return overlay


class PerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("arm_exchange_perception")

        self.camera_side = str(self.declare_parameter("camera_side", "left").value)
        if self.camera_side not in {"left", "right"}:
            raise ValueError("camera_side must be 'left' or 'right'")
        perception_cfg = load_config()["perception"]
        self.active_camera_side = self.camera_side
        camera_sources = {
            side: dict(cfg)
            for side, cfg in dict(perception_cfg["camera_sources"]).items()
        }
        self.camera_sources = camera_sources
        self.frame_to_side = {str(cfg["frame_id"]): side for side, cfg in camera_sources.items()}
        self.camera_info_by_side: dict[str, CameraInfo] = {}
        self.show_window = bool(self.declare_parameter("show_window", True).value)
        self.publish_annotated_image = bool(
            self.declare_parameter("publish_annotated_image", False).value
        )
        self.annotated_image_topic = str(
            self.declare_parameter(
                "annotated_image_topic",
                "/host/perception/annotated_image",
            ).value
        )
        self.tf_lookup_timeout_s = float(self.declare_parameter("tf_lookup_timeout_s", 0.05).value)
        self.window_name = str(
            self.declare_parameter("window_name", "second_camera_perception").value
        )
        if self.show_window:
            import cv2

            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        cfg = perception_cfg.get("pipeline", {})
        detector_cfg = dict(cfg["keypoint_detector"])
        detector_cfg.pop("schema", None)
        detector_cfg["yolo_model"] = Path(detector_cfg["yolo_model"])
        detector_cfg["hrnet_model"] = Path(detector_cfg["hrnet_model"])
        self.detector = YoloHRNetBackend(**detector_cfg)
        self._warmup_detector(dict(cfg.get("warmup", {})))

        solver_cfg = dict(cfg["pose_solver"])
        self.output_frame = str(solver_cfg.pop("output_frame"))
        schema_name = str(solver_cfg.pop("schema"))
        solver_cfg["object_points"] = perception_cfg["keypoint_schemas"][schema_name]["points"]
        self.pose_estimator = PnPEstimator(**solver_cfg)

        self.tf_buffer = Buffer()
        self.tf_node = Node(
            f"{self.get_name()}_tf_listener",
            context=self.context,
            use_global_arguments=False,
        )
        self.tf_listener = TransformListener(self.tf_buffer, self.tf_node)
        self._tf_stop = threading.Event()
        self.tf_executor = SingleThreadedExecutor(context=self.context)
        self.tf_executor.add_node(self.tf_node)
        self._tf_thread = threading.Thread(target=self._spin_tf_listener, name="perception_tf", daemon=False)
        self._tf_thread.start()

        self.pose_pub = self.create_publisher(PoseStamped, "/host/perception/exchange_station_pose", 10)
        self.status_pub = self.create_publisher(String, "/host/perception/status", 10)
        self.annotated_image_pub = (
            self.create_publisher(Image, self.annotated_image_topic, qos_profile_sensor_data)
            if self.publish_annotated_image
            else None
        )
        self.create_subscription(ArmCtrlGimbalControlMsg, "/mcu/gimbal/control", self._on_gimbal_control, 10)
        for side, camera_cfg in self.camera_sources.items():
            self.create_subscription(
                CameraInfo,
                str(camera_cfg["camera_info_topic"]),
                lambda msg, camera_side=side: self._on_camera_info(camera_side, msg),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                Image,
                str(camera_cfg["image_topic"]),
                self._on_image,
                qos_profile_sensor_data,
            )

    def _warmup_detector(self, cfg: dict) -> None:
        enabled = bool(cfg.get("enabled", True))
        if not enabled:
            return

        image_path = cfg.get("image_path", "weights/eval_samples/image0.png")
        repeat = max(1, int(cfg.get("repeat", 1)))
        fallback_size = tuple(int(v) for v in cfg.get("fallback_size", (640, 640)))

        image_bgr = None
        if image_path:
            path = Path(image_path)
            image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                self.get_logger().warning(
                    f"perception warmup image unavailable: {path}; using blank fallback"
                )
        if image_bgr is None:
            width, height = fallback_size
            image_bgr = np.zeros((height, width, 3), dtype=np.uint8)

        pipeline = getattr(self.detector, "_pipeline", None)
        if pipeline is None:
            self.get_logger().warning("perception warmup skipped: detector has no pipeline")
            return

        try:
            start = time.perf_counter()
            result = None
            for idx in range(repeat):
                result = pipeline.predict_numpy(image_bgr, image_id=f"warmup_{idx}")
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            detections = 0 if result is None else len(result.yolo_detections)
            crops = 0 if result is None else len(result.instances)
            self.get_logger().info(
                "perception warmup complete: "
                f"repeat={repeat} elapsed={elapsed_ms:.1f}ms "
                f"detections={detections} crops={crops}"
            )
        except Exception as exc:
            self.get_logger().warning(
                f"perception warmup failed: {type(exc).__name__}: {exc}"
            )

    def _spin_tf_listener(self) -> None:
        while rclpy.ok() and not self._tf_stop.is_set():
            try:
                self.tf_executor.spin_once(timeout_sec=0.001)
            except ExternalShutdownException:
                break
            except Exception:
                if not rclpy.ok() or self._tf_stop.is_set():
                    break
                raise

    def _on_gimbal_control(self, msg: ArmCtrlGimbalControlMsg) -> None:
        self.active_camera_side = "right" if int(msg.camera) == ArmCtrlGimbalControlMsg.CAMERA_RIGHT else "left"

    def _on_camera_info(self, camera_side: str, msg: CameraInfo) -> None:
        self.camera_info_by_side[camera_side] = msg

    def _on_image(self, msg: Image) -> None:
        frame_id = str(msg.header.frame_id)
        camera_side = self.frame_to_side.get(frame_id)
        if camera_side is None:
            raise ValueError(
                "Image frame_id mismatch: got unknown frame "
                f"{frame_id!r}, expected one of {sorted(self.frame_to_side)}"
            )
        if camera_side != self.active_camera_side:
            return
        camera_info = self.camera_info_by_side.get(camera_side)
        if camera_info is None:
            return

        image_bgr = image_to_bgr(msg)
        input_frame = str(self.camera_sources[camera_side]["frame_id"])
        camera_info_frame = str(camera_info.header.frame_id)
        if camera_info_frame != input_frame:
            raise ValueError(
                "CameraInfo frame_id mismatch: "
                f"got {camera_info_frame!r}, expected {input_frame!r}"
            )
        observation = None
        try:
            observation = self.detector.detect(
                image_bgr,
                image_id=f"{msg.header.stamp.sec}.{msg.header.stamp.nanosec}",
            )
            camera_matrix = _camera_matrix_from_info(camera_info)
            distortion = _dist_coeffs_from_info(camera_info)
            transform, reprojection_error = self.pose_estimator.estimate(
                observation,
                camera_matrix,
                distortion,
            )
        except Exception as exc:
            status = f"perception_failed: {exc}"
            self.status_pub.publish(String(data=status))
            self._show_detection(
                image_bgr,
                observation,
                status,
                stamp=msg.header.stamp,
                frame_id=frame_id,
            )
            return

        pose_msg = pose_stamped_from_transform(
            transform,
            stamp=msg.header.stamp,
            frame_id=input_frame,
        )
        if pose_msg.header.frame_id != self.output_frame:
            if not self.tf_buffer.can_transform(
                self.output_frame,
                pose_msg.header.frame_id,
                pose_msg.header.stamp,
                timeout=Duration(seconds=self.tf_lookup_timeout_s),
            ):
                status = (
                    "tf_unavailable: "
                    f"{self.output_frame} <- {pose_msg.header.frame_id} "
                    f"at {pose_msg.header.stamp.sec}.{pose_msg.header.stamp.nanosec:09d}"
                )
                self.status_pub.publish(String(data=status))
                self._show_detection(
                    image_bgr,
                    observation,
                    status,
                    stamp=msg.header.stamp,
                    frame_id=frame_id,
                )
                return
            tf_msg = self.tf_buffer.lookup_transform(
                self.output_frame,
                pose_msg.header.frame_id,
                pose_msg.header.stamp,
            )
            pose_msg = transform_pose_stamped(pose_msg, tf_msg, self.output_frame)

        self.pose_pub.publish(pose_msg)
        self.status_pub.publish(String(data="ok"))
        bbox_source = str(observation.metadata.get("bbox_source", ""))
        input_pose_line = _pose_line(f"T_{input_frame}_exchange", transform)
        output_pose_line = _pose_line(
            f"T_{self.output_frame}_exchange",
            transform_from_pose_stamped(pose_msg),
        )
        title = f"ok | bbox={bbox_source} | reproj={reprojection_error:.3f}px"
        pose_lines = (
            input_pose_line,
            output_pose_line,
        )
        self._show_detection(
            image_bgr,
            observation,
            title,
            pose_lines=pose_lines,
            stamp=msg.header.stamp,
            frame_id=frame_id,
        )

    def _show_detection(
        self,
        image_bgr: np.ndarray,
        observation,
        title: str,
        *,
        pose_lines: tuple[str, ...] = (),
        stamp=None,
        frame_id: str = "",
    ) -> None:
        if not self.show_window and self.annotated_image_pub is None:
            return

        overlay = _draw_detection_overlay(
            image_bgr,
            observation,
            title=title,
            pose_lines=pose_lines,
        )
        if self.annotated_image_pub is not None and stamp is not None:
            self._publish_annotated_image(overlay, stamp, frame_id)
        if self.show_window:
            import cv2

            cv2.imshow(self.window_name, overlay)
            cv2.waitKey(1)

    def _publish_annotated_image(self, image_bgr: np.ndarray, stamp, frame_id: str) -> None:
        image = np.ascontiguousarray(image_bgr, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"annotated image must be HxWx3 uint8 BGR, got shape={image.shape}")
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height = int(image.shape[0])
        msg.width = int(image.shape[1])
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = int(image.shape[1] * 3)
        msg.data = image.tobytes()
        self.annotated_image_pub.publish(msg)

    def destroy_node(self) -> bool:
        self._tf_stop.set()
        self.tf_executor.wake()
        if self._tf_thread.is_alive():
            self._tf_thread.join(timeout=1.0)
        self.tf_listener.unregister()
        self.tf_executor.remove_node(self.tf_node)
        self.tf_executor.shutdown()
        self.tf_node.destroy_node()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node.show_window:
            import cv2

            cv2.destroyWindow(node.window_name)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
