from __future__ import annotations

import math

import numpy as np
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

from arm_exchange_host.ros_utils import (
    dense_trajectory_to_msg,
    duration_to_seconds,
    joint_positions_from_joint_state,
    pose_stamped_from_transform,
    transform_from_pose_stamped,
)
from arm_exchange_core.transform import rotations_from_quaternions
from arm_exchange_core.trajectory import sample_quintic_trajectory


def test_pose_uses_parent_from_child_matrix_convention() -> None:
    msg = PoseStamped()
    msg.header.frame_id = "arm_base"
    msg.pose.position.x = 1.0
    msg.pose.position.y = 2.0
    msg.pose.position.z = 3.0
    msg.pose.orientation.z = math.sin(math.pi / 4.0)
    msg.pose.orientation.w = math.cos(math.pi / 4.0)

    transform = transform_from_pose_stamped(msg)

    expected = np.array(
        [
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(transform, expected, atol=1e-12)


def test_ros_xyzw_and_core_wxyz_quaternion_round_trip() -> None:
    quaternion_wxyz = np.array([0.5, -0.5, 0.5, -0.5])
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotations_from_quaternions(quaternion_wxyz[None, :])[0]
    transform[:3, 3] = [0.1, 0.2, 0.3]
    msg = pose_stamped_from_transform(
        transform,
        stamp=PoseStamped().header.stamp,
        frame_id="arm_base",
    )
    ros_wxyz = np.asarray(
        [
            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
        ]
    )
    assert np.isclose(abs(float(np.dot(ros_wxyz, quaternion_wxyz))), 1.0)
    np.testing.assert_allclose(transform_from_pose_stamped(msg), transform)


def test_joint_state_is_reordered_by_joint_name() -> None:
    msg = JointState()
    msg.name = ["joint4", "joint2", "joint6", "joint1", "joint5", "joint3"]
    msg.position = [4.0, 2.0, 6.0, 1.0, 5.0, 3.0]

    np.testing.assert_array_equal(
        joint_positions_from_joint_state(msg),
        np.arange(1.0, 7.0),
    )


def test_dense_trajectory_preserves_endpoints_and_timestamps() -> None:
    waypoints = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.1, -0.1, 0.3, -0.2, 0.4],
            [0.4, 0.2, -0.2, 0.6, -0.4, 0.8],
        ]
    )
    dense = sample_quintic_trajectory(
        waypoints,
        np.array([0.0, 1.0, 2.0]),
        0.2,
    )
    msg = dense_trajectory_to_msg(dense, stamp=PoseStamped().header.stamp)

    np.testing.assert_allclose(msg.points[0].positions, waypoints[0])
    np.testing.assert_allclose(msg.points[-1].positions, waypoints[-1])
    timestamps = np.array([duration_to_seconds(point.time_from_start) for point in msg.points])
    assert timestamps[0] == 0.0
    assert timestamps[-1] == 2.0
    assert np.all(np.diff(timestamps) > 0.0)
