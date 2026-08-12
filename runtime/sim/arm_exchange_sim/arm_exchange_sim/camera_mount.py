from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from arm_exchange_core import load_config
from arm_exchange_core.arm_model import ArmModel
from arm_exchange_core.transform import quaternions_from_rotations
from mujoco_engine.plugin_base import BasePlugin, PluginContext
from scipy.spatial.transform import Rotation


def transform_from_pos_rotation(position, rotation=None) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    if rotation is not None:
        matrix[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    matrix[:3, 3] = np.asarray(position, dtype=float).reshape(3)
    return matrix


def invert_transform(matrix: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=float)
    out[:3, :3] = matrix[:3, :3].T
    out[:3, 3] = -(out[:3, :3] @ matrix[:3, 3])
    return out


@dataclass(frozen=True, slots=True)
class ParallelMountKinematics:
    arm: ArmModel
    t_linkage_mount: np.ndarray

    @classmethod
    def from_config(
        cls,
        *,
        reduced_to_mount_xyz=(0.0, 0.0, 0.0),
        reduced_to_mount_rpy_deg=(0.0, 0.0, 0.0),
    ) -> "ParallelMountKinematics":
        rotation = Rotation.from_euler(
            "xyz",
            np.asarray(reduced_to_mount_rpy_deg, dtype=float),
            degrees=True,
        ).as_matrix()
        return cls(
            arm=ArmModel.from_config(load_config()["arm"]),
            t_linkage_mount=transform_from_pos_rotation(
                reduced_to_mount_xyz,
                rotation,
            ),
        )

    def t_arm_linkage(self, joints: np.ndarray) -> np.ndarray:
        """Return compensated linkage transforms for a ``(B, 6)`` joint batch.

        The origin follows the O2E point on link2, while the orientation is
        compensated back to F1.
        """

        links = self.arm.forward_kinematics(joints).link_transforms
        t_arm_f1 = links[:, 1]
        t_arm_f2 = links[:, 2]
        offset_f2 = np.asarray([self.arm.a[2], 0.0, 0.0], dtype=float)
        result = t_arm_f1.copy()
        result[:, :3, 3] = t_arm_f2[:, :3, 3] + np.einsum(
            "bij,j->bi",
            t_arm_f2[:, :3, :3],
            offset_f2,
        )
        return result

    def t_arm_mount(self, joints: np.ndarray) -> np.ndarray:
        return self.t_arm_linkage(joints) @ self.t_linkage_mount


def _body_matrix(data, body_id: int) -> np.ndarray:
    return transform_from_pos_rotation(
        data.xpos[body_id],
        data.xmat[body_id].reshape(3, 3),
    )


class CameraMountPlugin(BasePlugin):
    """Drive the simulated camera mount from the compensated parallel linkage."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mount_body = str(kwargs.get("mount_body", "parallel_camera_mount"))
        self.joint1 = str(kwargs.get("joint1", "joint1"))
        self.joint2 = str(kwargs.get("joint2", "joint2"))
        self.kinematics = ParallelMountKinematics.from_config(
            reduced_to_mount_xyz=kwargs.get("reduced_to_mount_xyz", [0.0, 0.0, 0.0]),
            reduced_to_mount_rpy_deg=kwargs.get(
                "reduced_to_mount_rpy_deg", [0.0, 0.0, 0.0]
            ),
        )

    def on_compile_callback(self, context: PluginContext):
        model = context.model
        mount_body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            self.mount_body,
        )
        if mount_body_id < 0:
            raise ValueError(f"CameraMountPlugin missing mount body: {self.mount_body}")
        self.mount_mocap_id = int(model.body_mocapid[mount_body_id])
        if self.mount_mocap_id < 0:
            raise ValueError(f"CameraMountPlugin mount body must be mocap: {self.mount_body}")
        self.chassis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        if self.chassis_id < 0:
            raise ValueError("CameraMountPlugin missing body: chassis")
        self.joint1_qpos = self._joint_qpos_addr(model, self.joint1)
        self.joint2_qpos = self._joint_qpos_addr(model, self.joint2)
        self._apply(context)
        mujoco.mj_forward(model, context.data)

    @staticmethod
    def _joint_qpos_addr(model, name: str) -> int:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"CameraMountPlugin missing joint: {name}")
        return int(model.jnt_qposadr[joint_id])

    def on_step_callback(self, context: PluginContext):
        self._apply(context)

    def _apply(self, context: PluginContext):
        data = context.data
        q1 = float(data.qpos[self.joint1_qpos])
        q2 = float(data.qpos[self.joint2_qpos])
        joints = np.asarray([[q1, q2, 0.0, 0.0, 0.0, 0.0]], dtype=float)
        t_world_chassis = _body_matrix(data, self.chassis_id)
        t_chassis_arm = transform_from_pos_rotation(
            [0.0, 0.0, 0.0],
            np.diag([-1.0, -1.0, 1.0]),
        )
        t_world_mount = (
            t_world_chassis
            @ t_chassis_arm
            @ self.kinematics.t_arm_mount(joints)[0]
        )
        data.mocap_pos[self.mount_mocap_id] = t_world_mount[:3, 3]
        data.mocap_quat[self.mount_mocap_id] = quaternions_from_rotations(
            t_world_mount[None, :3, :3]
        )[0]
