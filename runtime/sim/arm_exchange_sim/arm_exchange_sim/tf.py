from __future__ import annotations

import mujoco
import numpy as np
from arm_exchange_core.transform import quaternions_from_rotations
from geometry_msgs.msg import TransformStamped
from mujoco_engine.plugin_base import BasePlugin, PluginContext, PluginSetupContext
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from .camera_mount import ParallelMountKinematics, invert_transform


MJ_CAMERA_TO_OPENCV = np.diag([1.0, -1.0, -1.0])


def _matrix_from_pos_rot(position, rotation) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    matrix[:3, 3] = np.asarray(position, dtype=float).reshape(3)
    return matrix


def _body_matrix(data, body_id: int) -> np.ndarray:
    return _matrix_from_pos_rot(data.xpos[body_id], data.xmat[body_id].reshape(3, 3))


def _camera_matrix(data, camera_id: int) -> np.ndarray:
    return _matrix_from_pos_rot(data.cam_xpos[camera_id], data.cam_xmat[camera_id].reshape(3, 3))


def _transform_msg(parent: str, child: str, matrix: np.ndarray, stamp) -> TransformStamped:
    msg = TransformStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = parent
    msg.child_frame_id = child
    msg.transform.translation.x = float(matrix[0, 3])
    msg.transform.translation.y = float(matrix[1, 3])
    msg.transform.translation.z = float(matrix[2, 3])
    quat = quaternions_from_rotations(matrix[None, :3, :3])[0]
    msg.transform.rotation.w = float(quat[0])
    msg.transform.rotation.x = float(quat[1])
    msg.transform.rotation.y = float(quat[2])
    msg.transform.rotation.z = float(quat[3])
    return msg


def _identity_matrix() -> np.ndarray:
    return np.eye(4, dtype=float)


def _rot_z_pi_matrix() -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = np.diag([-1.0, -1.0, 1.0])
    return matrix


def _camera_optical_matrix() -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = MJ_CAMERA_TO_OPENCV.T
    return matrix


class ArmExchangeTfPlugin(BasePlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.world_frame = str(kwargs.get("world_frame", "mujoco_world"))
        self.base_frame = str(kwargs.get("base_frame", "base_link"))
        self.chassis_frame = str(kwargs.get("chassis_frame", "chassis"))
        self.arm_base_frame = str(kwargs.get("arm_base_frame", "arm_base"))
        self.station_frame = str(kwargs.get("station_frame", "exchange_station"))
        self.publish_station_frame = bool(kwargs.get("publish_station_frame", True))
        self.cameras = tuple(dict(item) for item in kwargs.get("cameras", ()))
        self.camera_mount_cfg = dict(kwargs.get("camera_mount", {}))
        self.camera_reduced_frame = str(self.camera_mount_cfg.get("reduced_frame", "camera_reduced_frame"))
        self.camera_mount_joint1 = str(self.camera_mount_cfg.get("joint1", "joint1"))
        self.camera_mount_joint2 = str(self.camera_mount_cfg.get("joint2", "joint2"))
        self.camera_mount_kinematics = ParallelMountKinematics.from_config()

    def setup(self, context: PluginSetupContext):
        self.tf_broadcaster = TransformBroadcaster(context.node)
        self.static_broadcaster = StaticTransformBroadcaster(context.node)

    def on_compile_callback(self, context: PluginContext):
        model = context.model
        self.chassis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        self.station_id = -1
        if self.publish_station_frame:
            self.station_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "E")
        self.camera_mount_joint1_qpos = self._joint_qpos_addr(model, self.camera_mount_joint1)
        self.camera_mount_joint2_qpos = self._joint_qpos_addr(model, self.camera_mount_joint2)
        self.camera_mount_body_ids = {}
        self.camera_link_body_ids = {}
        for cfg in self.cameras:
            self.camera_mount_body_ids[str(cfg["mount_frame"])] = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                str(cfg["mount_body"]),
            )
            self.camera_link_body_ids[str(cfg["link_frame"])] = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                str(cfg["link_body"]),
            )
        self.camera_ids = {
            str(cfg["mujoco_camera"]): mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                str(cfg["mujoco_camera"]),
            )
            for cfg in self.cameras
        }

        missing = []
        if self.chassis_id < 0:
            missing.append("body:chassis")
        if self.publish_station_frame and self.station_id < 0:
            missing.append("body:E")
        missing.extend(
            f"body:{frame}"
            for frame, body_id in self.camera_mount_body_ids.items()
            if body_id < 0
        )
        missing.extend(
            f"body:{frame}"
            for frame, body_id in self.camera_link_body_ids.items()
            if body_id < 0
        )
        missing.extend(f"camera:{name}" for name, cam_id in self.camera_ids.items() if cam_id < 0)
        if missing:
            raise ValueError("ArmExchangeTfPlugin missing MuJoCo objects: " + ", ".join(missing))

        self._publish_static(context, context.node.get_clock().now().to_msg())

    @staticmethod
    def _joint_qpos_addr(model, name: str) -> int:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"ArmExchangeTfPlugin missing joint: {name}")
        return int(model.jnt_qposadr[joint_id])

    def _publish_static(self, context: PluginContext, stamp) -> None:
        static_msgs = []
        static_msgs.extend(
            [
                _transform_msg(self.base_frame, self.chassis_frame, _identity_matrix(), stamp),
                _transform_msg(self.chassis_frame, self.arm_base_frame, _rot_z_pi_matrix(), stamp),
            ]
        )
        q1 = float(context.data.qpos[self.camera_mount_joint1_qpos])
        q2 = float(context.data.qpos[self.camera_mount_joint2_qpos])
        joints = np.asarray([[q1, q2, 0.0, 0.0, 0.0, 0.0]], dtype=float)
        t_world_arm = _body_matrix(context.data, self.chassis_id) @ _rot_z_pi_matrix()
        t_world_linkage = (
            t_world_arm
            @ self.camera_mount_kinematics.t_arm_linkage(joints)[0]
        )
        for cfg in self.cameras:
            t_world_camera_mount = _body_matrix(
                context.data,
                self.camera_mount_body_ids[str(cfg["mount_frame"])],
            )
            static_msgs.append(
                _transform_msg(
                    self.camera_reduced_frame,
                    str(cfg["mount_frame"]),
                    invert_transform(t_world_linkage) @ t_world_camera_mount,
                    stamp,
                )
            )
            camera_name = str(cfg["mujoco_camera"])
            t_world_link = _body_matrix(
                context.data,
                self.camera_link_body_ids[str(cfg["link_frame"])],
            )
            t_world_camera = _camera_matrix(context.data, self.camera_ids[camera_name])
            t_link_optical = invert_transform(t_world_link) @ t_world_camera @ _camera_optical_matrix()
            static_msgs.append(
                _transform_msg(
                    str(cfg["link_frame"]),
                    str(cfg["optical_frame"]),
                    t_link_optical,
                    stamp,
                )
            )
        self.static_broadcaster.sendTransform(static_msgs)

    def on_timer_callback(self, context: PluginContext, alias: str):
        self._publish_dynamic(context, context.node.get_clock().now().to_msg())

    def _publish_dynamic(self, context: PluginContext, stamp) -> None:
        data = context.data

        t_world_base = _body_matrix(data, self.chassis_id)

        dynamic_msgs = [
            _transform_msg(self.world_frame, self.base_frame, t_world_base, stamp),
        ]
        if self.publish_station_frame:
            station = _body_matrix(data, self.station_id)
            dynamic_msgs.append(
                _transform_msg(self.world_frame, self.station_frame, station, stamp)
            )

        q1 = float(data.qpos[self.camera_mount_joint1_qpos])
        q2 = float(data.qpos[self.camera_mount_joint2_qpos])
        joints = np.asarray([[q1, q2, 0.0, 0.0, 0.0, 0.0]], dtype=float)
        t_arm_linkage = self.camera_mount_kinematics.t_arm_linkage(joints)[0]
        dynamic_msgs.append(
            _transform_msg(
                self.arm_base_frame,
                self.camera_reduced_frame,
                t_arm_linkage,
                stamp,
            )
        )

        for cfg in self.cameras:
            t_world_camera_mount = _body_matrix(data, self.camera_mount_body_ids[str(cfg["mount_frame"])])
            t_world_link = _body_matrix(data, self.camera_link_body_ids[str(cfg["link_frame"])])
            dynamic_msgs.append(
                _transform_msg(
                    str(cfg["mount_frame"]),
                    str(cfg["link_frame"]),
                    invert_transform(t_world_camera_mount) @ t_world_link,
                    stamp,
                )
            )
        self.tf_broadcaster.sendTransform(dynamic_msgs)
