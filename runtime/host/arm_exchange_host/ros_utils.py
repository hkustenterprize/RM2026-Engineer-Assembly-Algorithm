from __future__ import annotations

import numpy as np
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from arm_exchange_interfaces.msg import ArmHost2MCUMsg
from arm_exchange_core.trajectory import JointTrajectory as CoreJointTrajectory
from arm_exchange_core.transform import (
    quaternions_from_rotations,
    rotations_from_quaternions,
)


JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")


def image_to_bgr(msg: Image) -> np.ndarray:
    import cv2

    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)
    encoding = str(msg.encoding)
    raw = np.asarray(msg.data, dtype=np.uint8)[: height * step]

    if encoding == "bayer_rggb8":
        image = raw.reshape(height, step)[:, :width]
        return cv2.cvtColor(image, cv2.COLOR_BayerRG2RGB)
    if encoding == "mono8":
        image = raw.reshape(height, step)[:, :width]
        return np.repeat(image[:, :, None], 3, axis=2)
    if encoding == "rgb8":
        image = raw.reshape(height, step)[:, : width * 3].reshape(height, width, 3)
        return image[..., ::-1].copy()
    if encoding == "bgr8":
        return raw.reshape(height, step)[:, : width * 3].reshape(height, width, 3)

    raise ValueError(f"unsupported sensor_msgs/Image encoding: {encoding!r}")


def transform_from_pose_stamped(msg: PoseStamped) -> np.ndarray:
    q = msg.pose.orientation
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotations_from_quaternions(np.asarray([[q.w, q.x, q.y, q.z]]))[0]
    transform[:3, 3] = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
    return transform


def pose_stamped_from_transform(transform: np.ndarray, *, stamp, frame_id: str) -> PoseStamped:
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"transform must have shape (4, 4), got {transform.shape}")
    msg = PoseStamped()
    msg.header.stamp = stamp
    msg.header.frame_id = str(frame_id)
    msg.pose.position.x = float(transform[0, 3])
    msg.pose.position.y = float(transform[1, 3])
    msg.pose.position.z = float(transform[2, 3])
    quat = quaternions_from_rotations(transform[None, :3, :3])[0]
    msg.pose.orientation.w = float(quat[0])
    msg.pose.orientation.x = float(quat[1])
    msg.pose.orientation.y = float(quat[2])
    msg.pose.orientation.z = float(quat[3])
    return msg


def transform_pose_stamped(msg: PoseStamped, tf_msg: TransformStamped, target_frame: str) -> PoseStamped:
    """Apply T_target_source to a PoseStamped storing T_source_object."""

    t = tf_msg.transform.translation
    r = tf_msg.transform.rotation
    T_target_source = np.eye(4, dtype=float)
    T_target_source[:3, :3] = rotations_from_quaternions(np.asarray([[r.w, r.x, r.y, r.z]]))[0]
    T_target_source[:3, 3] = [t.x, t.y, t.z]

    transform = T_target_source @ transform_from_pose_stamped(msg)
    return pose_stamped_from_transform(transform, stamp=msg.header.stamp, frame_id=target_frame)


def joint_positions_from_joint_state(msg: JointState, joint_names=JOINT_NAMES) -> np.ndarray:
    by_name = {name: idx for idx, name in enumerate(msg.name)}
    missing = [name for name in joint_names if name not in by_name]
    if missing:
        raise ValueError(f"JointState missing joints: {missing}")
    return np.asarray([msg.position[by_name[name]] for name in joint_names], dtype=float)


def host_output_from_trajectory_point(point: JointTrajectoryPoint, stamp, host_state: int) -> ArmHost2MCUMsg:
    position = np.asarray(point.positions, dtype=float)
    velocity = np.asarray(point.velocities, dtype=float)
    if position.shape != (len(JOINT_NAMES),):
        raise ValueError(f"ArmHost2MCUMsg position must have shape ({len(JOINT_NAMES)},), got {position.shape}")
    if velocity.shape != position.shape:
        raise ValueError(f"ArmHost2MCUMsg velocity must have shape {position.shape}, got {velocity.shape}")

    msg = ArmHost2MCUMsg()
    msg.header.stamp = stamp
    msg.host_state = int(host_state)
    msg.position = [float(value) for value in position]
    msg.velocity = [float(value) for value in velocity]
    return msg


def dense_trajectory_to_msg(
    trajectory: CoreJointTrajectory,
    *,
    stamp,
    joint_names=JOINT_NAMES,
) -> JointTrajectory:
    msg = JointTrajectory()
    msg.header.stamp = stamp
    msg.joint_names = list(joint_names)

    expected_shape = (trajectory.timestamps.shape[0], len(joint_names))
    if trajectory.positions.shape != expected_shape:
        raise ValueError(
            "dense trajectory positions must have shape "
            f"{expected_shape}, got {trajectory.positions.shape}"
        )
    for q, qd, qdd, t in zip(
        trajectory.positions,
        trajectory.velocities,
        trajectory.accelerations,
        trajectory.timestamps,
        strict=True,
    ):
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in q]
        point.velocities = [float(v) for v in qd]
        point.accelerations = [float(v) for v in qdd]
        point.time_from_start = seconds_to_duration(float(t))
        msg.points.append(point)
    return msg


def seconds_to_duration(value: float) -> Duration:
    whole = int(value)
    frac = float(value) - whole
    msg = Duration()
    msg.sec = whole
    msg.nanosec = int(round(frac * 1e9))
    if msg.nanosec >= 1_000_000_000:
        msg.sec += 1
        msg.nanosec -= 1_000_000_000
    return msg


def duration_to_seconds(msg: Duration) -> float:
    return float(msg.sec) + float(msg.nanosec) * 1e-9
