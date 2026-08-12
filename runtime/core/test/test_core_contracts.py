from __future__ import annotations

import cv2
import inspect
import numpy as np

from arm_exchange_core.arm_model import ArmModel
from arm_exchange_core import load_config
from arm_exchange_core.perception import (
    KeypointObservation,
    PnPEstimator,
    YoloHRNetBackend,
)
from arm_exchange_core.planning.collision import CollisionChecker, CollisionModel
from arm_exchange_core.planning.type3 import (
    AssemblyPath,
    AssemblyState,
    Type3Planner,
)
from arm_exchange_core.planning.viterbi import solve_viterbi


CONFIG = load_config()


def _type3_planner(arm: ArmModel | None = None) -> Type3Planner:
    arm = ArmModel.from_config(CONFIG["arm"]) if arm is None else arm
    planning = CONFIG["planning"]
    stage = planning["exchange"]["stage_path"]
    graph = stage["joint_path"]
    return Type3Planner(
        arm,
        geometry=planning["type3"]["exchange_trajectory"],
        ik_branch=int(planning["exchange"]["ik_branches"][0]),
        roll_sample_step_deg=float(graph["roll_sample_step_deg"]),
        bandwidth=int(graph["bandwidth"]),
        collision_soft_margin=float(graph["collision_soft_margin"]),
        collision_soft_weight=float(graph["collision_soft_weight"]),
        motion_l1_weight=float(graph["motion_l1_weight"]),
        motion_l2_weight=float(graph["motion_l2_weight"]),
        joint_limit_margin_rad=float(graph["joint_limit_margin_rad"]),
        joint_limit_weight=float(graph["joint_limit_weight"]),
        best_effort=stage["best_ik"],
        continuation=graph["q_sweep"],
    )


def test_pnp_returns_camera_from_target_transform() -> None:
    object_point_map = CONFIG["perception"]["keypoint_schemas"]["exchange_12pt"]["points"]
    names = tuple(object_point_map)
    object_points = np.asarray([object_point_map[name] for name in names])
    camera_matrix = np.asarray(
        [[700.0, 0.0, 320.0], [0.0, 700.0, 320.0], [0.0, 0.0, 1.0]]
    )
    expected = np.eye(4, dtype=float)
    expected[:3, :3], _ = cv2.Rodrigues(np.asarray([0.08, -0.12, 0.05]))
    expected[:3, 3] = [0.04, -0.03, 1.2]
    projected, _ = cv2.projectPoints(
        object_points,
        cv2.Rodrigues(expected[:3, :3])[0],
        expected[:3, 3],
        camera_matrix,
        np.zeros(5),
    )
    observation = KeypointObservation(
        keypoints_2d=projected.reshape(-1, 2),
        keypoint_names=names,
    )

    transform, error = PnPEstimator(object_points=object_point_map).estimate(
        observation,
        camera_matrix,
    )

    assert error < 1e-6
    np.testing.assert_allclose(transform, expected, atol=1e-6)


def test_detector_config_matches_constructor() -> None:
    config_keys = set(
        CONFIG["perception"]["pipeline"]["keypoint_detector"]
    ) - {"schema"}
    parameters = set(
        inspect.signature(YoloHRNetBackend).parameters
    )

    assert config_keys <= parameters


def test_type3_pose_respects_arm_base_transform() -> None:
    planner = _type3_planner()
    path = AssemblyPath(AssemblyState(slide_m=0.06).as_array()[None, :])
    identity = np.eye(4, dtype=float)
    translated = np.eye(4, dtype=float)
    translated[:3, 3] = [0.4, -0.2, 0.1]
    local = planner.task_poses(path, identity, np.asarray([0.2]))
    shifted = planner.task_poses(path, translated, np.asarray([0.2]))

    np.testing.assert_allclose(shifted, translated[None, :, :] @ local)


def test_collision_assets_are_owned_by_core_package() -> None:
    model = CollisionModel.from_config(CONFIG["collision"])
    for path in model.station_meshes + model.local_exchange_meshes:
        assert path.is_file()
        assert "arm_exchange_core/assets/collision" in path.as_posix()


def test_collision_checker_batch_contracts() -> None:
    arm = ArmModel.from_config(CONFIG["arm"])
    model = CollisionModel.from_config(CONFIG["collision"])
    station = np.eye(4)
    station[:3, 3] = [0.2, -0.2, 0.3]
    checker = CollisionChecker(
        arm,
        [(model.station_meshes, station)],
        model.arm_capsules,
    )
    joints = np.asarray(
        [
            [0.0, 1.2, 1.0, 0.0, 1.0, 0.0],
            [0.3, 1.0, 1.2, 0.2, 0.8, -0.2],
        ]
    )

    collides = checker.check_configs(joints)
    inflated = checker.check_configs(joints, extra_radius=0.01)
    collides_with_penalty, penalties = checker.check_configs_with_penalty(
        joints,
        soft_margin=0.05,
        soft_weight=3.0,
    )

    assert collides.shape == inflated.shape == collides_with_penalty.shape == (2,)
    assert penalties.shape == (2,)
    assert np.all(inflated >= collides)
    assert np.array_equal(collides_with_penalty, collides)
    assert np.all(penalties >= 0.0)
    assert checker.check_configs(np.empty((0, 6))).shape == (0,)


def test_arm_model_batch_contracts() -> None:
    arm = ArmModel.from_config(CONFIG["arm"])
    joints = np.asarray(
        [
            [0.0, 1.2, 1.0, 1.5, 1.2, 0.0],
            [0.2, 1.1, 1.1, 1.4, 1.3, -0.2],
        ]
    )
    kinematics = arm.forward_kinematics(joints)
    assert kinematics.link_transforms.shape == (2, 7, 4, 4)
    assert kinematics.tcp_transforms.shape == (2, 4, 4)

    solutions, valid = arm.solve_ik(kinematics.tcp_transforms)
    assert solutions.shape == (2, 8, 6)
    assert valid.shape == (2, 8)
    assert np.all(np.any(valid, axis=1))
    for batch_index in range(2):
        recovered = arm.forward_kinematics(solutions[batch_index, valid[batch_index]])
        expected = np.broadcast_to(
            kinematics.tcp_transforms[batch_index], recovered.tcp_transforms.shape
        )
        np.testing.assert_allclose(recovered.tcp_transforms, expected, atol=1e-6)

    zeros = np.zeros_like(joints)
    assert arm.inverse_dynamics(joints, zeros, zeros).shape == joints.shape
    assert arm.external_wrench_torque(joints, np.zeros_like(joints)).shape == joints.shape
    batch_torque = arm.inverse_dynamics(joints, zeros, zeros)
    row_torque = np.vstack(
        [arm.inverse_dynamics(row[None, :], np.zeros((1, 6)), np.zeros((1, 6)))[0] for row in joints]
    )
    np.testing.assert_allclose(batch_torque, row_torque)


def test_type3_p_stage_includes_q_sweep() -> None:
    planning = CONFIG["planning"]
    stage = planning["exchange"]["stage_path"]
    joint_path = stage["joint_path"]
    arm = ArmModel.from_config(CONFIG["arm"])
    planner = _type3_planner(arm)
    collision_model = CollisionModel.from_config(CONFIG["collision"])
    station = np.asarray(
        [
            [-1.0, 0.0, 0.0, 0.2],
            [0.0, -1.0, 0.0, -0.2],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    checker = CollisionChecker(
        arm,
        [(collision_model.station_meshes, station)],
        collision_model.arm_capsules,
    )
    start_state = AssemblyState(slide_m=float(planning["type3"]["exchange_trajectory"]["slide_z_mag"]))
    p_done = AssemblyState(slide_m=start_state.slide_m, p_angle_rad=0.5 * np.pi)
    path = AssemblyPath.between(start_state, p_done)
    start_targets = planner.task_poses(
        AssemblyPath(start_state.as_array()[None, :]),
        station,
        planner.rolls,
    )
    solutions, valid = arm.solve_ik(start_targets, branches=(7,))
    start_index = int(np.flatnonzero(valid[:, 0])[0])

    q_step = float(joint_path["q_sweep"]["sample_step_deg"])
    q_angles = np.deg2rad(np.arange(0.0, 90.0 + 0.5 * q_step, q_step))
    plus_states = np.repeat(p_done.as_array()[None, :], len(q_angles), axis=0)
    minus_states = plus_states.copy()
    plus_states[:, 3] = q_angles
    minus_states[:, 3] = -q_angles
    result = planner.plan(
        path,
        station,
        initial_joints=solutions[start_index, 0],
        collision_checker=checker,
        terminal_continuations=(AssemblyPath(minus_states), AssemblyPath(plus_states)),
    )

    assert result.success
    assert result.waypoints.shape == (100, 6)
    assert result.continuation is not None
    assert len(result.continuation["states"]) == 37
    np.testing.assert_allclose(result.continuation["states"][:, 3], np.deg2rad(np.arange(-90, 95, 5)))
    continuation_joints = np.asarray(result.continuation["joints"])
    bounded = arm.joint_space.bounded
    assert np.all(continuation_joints[:, bounded] >= arm.joint_space.lower[bounded])
    assert np.all(continuation_joints[:, bounded] <= arm.joint_space.upper[bounded])


def test_runtime_config_has_no_deprecated_compatibility_fields() -> None:
    stage = CONFIG["planning"]["exchange"]["stage_path"]
    q_sweep = stage["joint_path"]["q_sweep"]

    assert "version" not in CONFIG
    assert "time_parameterization" not in stage
    assert "angle_step_deg" not in stage["q_manual"]
    assert not {
        "max_joint_step_rad",
        "motion_l2_weight",
        "relaxed_joint_index",
        "relaxed_limit_rad",
        "relaxed_limit_weight",
    } & q_sweep.keys()
    assert "release_slide_mag" not in CONFIG["planning"]["type3"]["exchange_trajectory"]


def test_approximate_ik_projects_only_the_initial_guess_into_hard_limits() -> None:
    planner = _type3_planner()
    q_ref = np.asarray([0.0, 1.0, 1.0, 0.0, 2.0, 0.0])
    bounded_reference = np.clip(
        q_ref,
        planner.arm.joint_space.lower,
        planner.arm.joint_space.upper,
    )
    target = planner.arm.forward_kinematics(bounded_reference[None, :]).tcp_transforms[0]

    solution, _ = planner._approximate_ik(target, q_ref)

    assert np.all(solution >= planner.arm.joint_space.lower)
    assert np.all(solution <= planner.arm.joint_space.upper)
    np.testing.assert_allclose(solution, bounded_reference, atol=1e-6)


def test_joint_space_distinguishes_periodic_and_bounded_joints() -> None:
    joint_space = ArmModel.from_config(CONFIG["arm"]).joint_space
    source = np.zeros((1, 6))
    target = np.zeros((1, 6))
    target[0, 0] = 2.0 * np.pi - 0.1
    target[0, 1] = 2.0 * np.pi - 0.1
    delta = joint_space.delta(target, source)
    expected_first = -0.1 if joint_space.continuous[0] else 2.0 * np.pi - 0.1
    expected_second = -0.1 if joint_space.continuous[1] else 2.0 * np.pi - 0.1
    np.testing.assert_allclose(delta[0, :2], [expected_first, expected_second])


def test_viterbi_returns_layered_minimum_cost_path() -> None:
    transition_costs = (
        np.asarray([[1.0, 4.0], [3.0, 1.0]]),
        np.asarray([[2.0, 5.0], [4.0, 1.0]]),
    )

    def step(layer_index: int, previous: np.ndarray):
        total = previous[None, :] + transition_costs[layer_index - 1]
        predecessors = np.argmin(total, axis=1)
        return total[np.arange(2), predecessors], predecessors

    result = solve_viterbi(np.asarray([0.0, 2.0]), 3, step)
    assert result.success
    assert result.path_indices == (0, 0, 0)
    assert result.cost == 3.0
