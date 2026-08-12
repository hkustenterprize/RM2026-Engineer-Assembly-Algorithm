from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from arm_exchange_interfaces.msg import (
    ApproachPlanningRequest,
    ApproachPlanningResult,
    RecoveryPlanningRequest,
    RecoveryPlanningResult,
    Type3PlanningRequest,
    Type3PlanningResult,
)
from arm_exchange_core.arm_model import ArmModel
from arm_exchange_core import load_config
from arm_exchange_core.transform import (
    quaternions_from_rotations,
    rotations_from_quaternions,
)
from arm_exchange_core.trajectory import FixedDurationParameterizer
from arm_exchange_core.planning.type2 import plan_joint_path_bitstar
from arm_exchange_core.planning.collision import CollisionChecker, CollisionModel
from arm_exchange_core.planning.type3 import (
    AssemblyPath,
    AssemblyState,
    Type3Planner,
)

from .ros_utils import (
    JOINT_NAMES,
    dense_trajectory_to_msg,
    joint_positions_from_joint_state,
    transform_from_pose_stamped,
    seconds_to_duration,
)


def _zyx_rpy_degrees(quaternion: np.ndarray) -> np.ndarray:
    rotation = rotations_from_quaternions(np.asarray(quaternion, dtype=float)[None, :])[0]
    pitch = -np.arcsin(np.clip(rotation[2, 0], -1.0, 1.0))
    roll = np.arctan2(rotation[2, 1], rotation[2, 2])
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    return np.rad2deg([roll, pitch, yaw])


def _step_counts_summary(values: np.ndarray) -> str:
    counts = np.asarray(values, dtype=np.int64)
    nonzero = np.flatnonzero(counts)
    if len(nonzero) == 0:
        return "all_zero"
    return (
        f"nonzero_steps={int(len(nonzero))}/{int(counts.size)} "
        f"first={int(nonzero[0])}:{int(counts[nonzero[0]])} "
        f"last={int(nonzero[-1])}:{int(counts[nonzero[-1]])} "
        f"min_nonzero={int(np.min(counts[nonzero]))} "
        f"max={int(np.max(counts))}"
    )


def _reason_counts_summary(values) -> str:
    counts = dict(values or {})
    if not counts:
        return "none"
    return ",".join(f"{key}:{int(counts[key])}" for key in sorted(counts))


class PlanningNode(Node):
    def __init__(self) -> None:
        super().__init__("arm_exchange_planning")
        system_cfg = load_config()
        self.arm = ArmModel.from_config(system_cfg["arm"])
        self.collision_model = CollisionModel.from_config(system_cfg["collision"])
        planning_cfg = system_cfg["planning"]
        geometry_cfg = planning_cfg["type3"]["exchange_trajectory"]
        exchange_cfg = planning_cfg["exchange"]
        stage_path_cfg = exchange_cfg["stage_path"]
        joint_path_cfg = stage_path_cfg["joint_path"]
        best_ik_cfg = dict(stage_path_cfg.get("best_ik", {}))
        trajectory_cfg = stage_path_cfg["trajectory"]
        approach_cfg = exchange_cfg["approach"]
        ready_adjust_cfg = stage_path_cfg["ready_adjust"]
        recovery_cfg = dict(stage_path_cfg["recovery"])
        self.motion_l1_weight = float(joint_path_cfg["motion_l1_weight"])
        self.motion_l2_weight = float(joint_path_cfg["motion_l2_weight"])
        self.time_parameterizer = FixedDurationParameterizer()
        self.approach_time_parameterizer = FixedDurationParameterizer()
        self.trajectory_duration_s = float(trajectory_cfg["duration_s"])
        self.trajectory_sample_dt = float(
            self.declare_parameter(
                "trajectory_sample_dt",
                float(trajectory_cfg["sample_dt"]),
            ).value
        )
        self.ik_branches = [
            int(v)
            for v in self.declare_parameter(
                "ik_branches",
                [int(v) for v in exchange_cfg["ik_branches"]],
            ).value
        ]
        if len(self.ik_branches) != 1:
            raise ValueError("Type III planning requires exactly one configured IK branch")
        self.insert_offset_m = float(geometry_cfg["insert_slide_mag"])
        self.slide_distance_m = float(geometry_cfg["slide_z_mag"])
        self.type3_planner = Type3Planner(
            self.arm,
            geometry=geometry_cfg,
            ik_branch=self.ik_branches[0],
            roll_sample_step_deg=float(joint_path_cfg["roll_sample_step_deg"]),
            bandwidth=int(joint_path_cfg["bandwidth"]),
            collision_soft_margin=float(joint_path_cfg["collision_soft_margin"]),
            collision_soft_weight=float(joint_path_cfg["collision_soft_weight"]),
            motion_l1_weight=self.motion_l1_weight,
            motion_l2_weight=self.motion_l2_weight,
            joint_limit_margin_rad=float(joint_path_cfg["joint_limit_margin_rad"]),
            joint_limit_weight=float(joint_path_cfg["joint_limit_weight"]),
            best_effort=best_ik_cfg,
            continuation=joint_path_cfg["q_sweep"],
        )
        self.approach_duration_s = float(approach_cfg["duration_s"])
        self.approach_sample_dt = float(approach_cfg["sample_dt"])
        self.approach_bit_timeout_s = float(approach_cfg["bit_timeout_s"])
        self.recovery_extract_distance_m = float(recovery_cfg["extract_distance_m"])
        self.recovery_extract_step_m = float(recovery_cfg["extract_step_m"])
        self.recovery_side_release_distance_m = float(recovery_cfg["side_release_distance_m"])
        self.recovery_side_release_step_m = float(recovery_cfg["side_release_step_m"])
        self.recovery_allow_degraded_release = bool(recovery_cfg["allow_degraded_release"])
        self.recovery_roll_search_max_rad = np.deg2rad(float(ready_adjust_cfg["roll_search_max_deg"]))
        self.recovery_roll_search_step_rad = np.deg2rad(float(ready_adjust_cfg["roll_search_step_deg"]))
        bit_timeout = recovery_cfg["bit_timeout_s"]
        self.recovery_bit_timeout_s = self.approach_bit_timeout_s if bit_timeout is None else float(bit_timeout)
        if not np.isfinite(self.recovery_roll_search_max_rad) or self.recovery_roll_search_max_rad < 0.0:
            raise ValueError("stage_path.ready_adjust.roll_search_max_deg must be a non-negative finite value")
        if self.recovery_roll_search_max_rad > 0.0 and (
            not np.isfinite(self.recovery_roll_search_step_rad)
            or self.recovery_roll_search_step_rad <= 0.0
        ):
            raise ValueError(
                "stage_path.ready_adjust.roll_search_step_deg must be positive "
                "when roll search is enabled"
            )
        for name, value in (
            ("recovery.extract_distance_m", self.recovery_extract_distance_m),
            ("recovery.extract_step_m", self.recovery_extract_step_m),
            ("recovery.side_release_distance_m", self.recovery_side_release_distance_m),
            ("recovery.side_release_step_m", self.recovery_side_release_step_m),
            ("recovery.bit_timeout_s", self.recovery_bit_timeout_s),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"stage_path.{name} must be a positive finite value")
        self.joint_lower = np.where(self.arm.joint_space.continuous, -np.pi, self.arm.joint_space.lower)
        self.joint_upper = np.where(self.arm.joint_space.continuous, np.pi, self.arm.joint_space.upper)
        self._type2_bound_tolerance = 1e-3
        self.get_logger().info(
            f"exchange ik_branch={self.ik_branches[0]} "
            f"type3_sample_dt={self.trajectory_sample_dt:.4f} "
            f"approach_sample_dt={self.approach_sample_dt:.4f} "
            f"motion_l1_weight={self.motion_l1_weight:.4f} "
            f"motion_l2_weight={self.motion_l2_weight:.4f} "
            f"roll_count={len(self.type3_planner.rolls)} "
            f"best_effort_ik={bool(best_ik_cfg.get('enabled', False))} "
            f"type3_time_parameterizer={type(self.time_parameterizer).__name__} "
            f"approach_time_parameterizer={type(self.approach_time_parameterizer).__name__}"
        )

        self.approach_result_pub = self.create_publisher(
            ApproachPlanningResult,
            "/host/planning/approach_result",
            10,
        )
        self.create_subscription(
            ApproachPlanningRequest,
            "/host/planning/approach_request",
            self._on_approach_request,
            10,
        )
        self.result_pub = self.create_publisher(
            Type3PlanningResult,
            "/host/planning/type3_result",
            10,
        )
        self.create_subscription(
            Type3PlanningRequest,
            "/host/planning/type3_request",
            self._on_request,
            10,
        )
        self.recovery_result_pub = self.create_publisher(
            RecoveryPlanningResult,
            "/host/planning/recovery_result",
            10,
        )
        self.create_subscription(
            RecoveryPlanningRequest,
            "/host/planning/recovery_request",
            self._on_recovery_request,
            10,
        )

    def _on_approach_request(self, msg: ApproachPlanningRequest) -> None:
        result = ApproachPlanningResult()
        result.header.stamp = self.get_clock().now().to_msg()
        result.header.frame_id = "arm_base"
        result.request_id = msg.request_id

        if msg.station_pose_arm_base.header.frame_id != "arm_base":
            raise ValueError(
                "station_pose_arm_base must be in arm_base, got "
                f"{msg.station_pose_arm_base.header.frame_id!r}"
            )
        t_arm_base_station = transform_from_pose_stamped(msg.station_pose_arm_base)

        initial_joints = joint_positions_from_joint_state(msg.initial_joint_state)
        initial_joints = self._normalize_for_type2_bounds(initial_joints[None, :])[0]
        ik_branches = tuple(self.ik_branches) if self.ik_branches else None
        self.get_logger().info(
            "approach request "
            f"id={msg.request_id} "
            f"station_pos={t_arm_base_station[:3, 3].tolist()} "
            f"initial_joints={initial_joints.tolist()} "
            f"ik_branches={list(ik_branches) if ik_branches is not None else 'all'} "
            f"bit_timeout_s={self.approach_bit_timeout_s:.3f} "
            f"local_exchange_meshes={len(self.collision_model.local_exchange_meshes)}"
        )

        collision_checker = self._make_approach_withdraw_checker(t_arm_base_station, t_arm_base_station)
        try:
            q_goal, goal_cost, goal_roll = self._select_approach_goal(
                t_arm_base_station,
                initial_joints,
                collision_checker,
                ik_branches,
            )
        except RuntimeError as exc:
            result.success = False
            result.message = str(exc)
            result.cost = float("inf")
            self.get_logger().warning(f"approach planning failed id={msg.request_id}: {result.message}")
            self.approach_result_pub.publish(result)
            return
        self.get_logger().info(
            "approach goal selected "
            f"id={msg.request_id} "
            f"goal_q={q_goal.tolist()} "
            f"goal_cost={goal_cost:.6f} "
            f"goal_roll={goal_roll:.6f}"
        )

        try:
            waypoints = plan_joint_path_bitstar(
                initial_joints,
                q_goal,
                validity_fn=lambda q: not bool(
                    collision_checker.check_configs(np.asarray(q, dtype=float)[None, :])[0]
                ),
                joint_lower=self.joint_lower,
                joint_upper=self.joint_upper,
                timeout_s=self.approach_bit_timeout_s,
            )
        except RuntimeError as exc:
            result.success = False
            result.message = str(exc)
            result.cost = float("inf")
            self.get_logger().warning(f"approach planning failed id={msg.request_id}: {result.message}")
            self.approach_result_pub.publish(result)
            return
        dense_trajectory = self._sample_trajectory(
            waypoints,
            duration_s=self.approach_duration_s,
            sample_dt=self.approach_sample_dt,
            label=f"approach id={msg.request_id}",
            parameterizer=self.approach_time_parameterizer,
        )
        result.success = True
        result.message = "ok"
        result.cost = float(goal_cost)
        result.trajectory = dense_trajectory_to_msg(
            dense_trajectory,
            stamp=result.header.stamp,
        )
        self.get_logger().info(
            "approach planning success "
            f"id={msg.request_id} "
            f"sparse_steps={int(waypoints.shape[0])} "
            f"dense_steps={int(dense_trajectory.positions.shape[0])} "
            f"duration_s={float(dense_trajectory.timestamps[-1]):.3f} "
            f"sample_dt={self.approach_sample_dt:.4f}"
        )
        self.approach_result_pub.publish(result)

    def _sample_trajectory(
        self,
        waypoints: np.ndarray,
        *,
        duration_s: float,
        sample_dt: float,
        label: str,
        parameterizer,
        **kwargs,
    ):
        trajectory = parameterizer.parameterize(
            waypoints,
            sample_dt=float(sample_dt),
            duration_s=float(duration_s),
            **kwargs,
        )
        self._log_trajectory_timing(label, trajectory)
        return trajectory

    def _sample_recovery_trajectory(self, waypoints: np.ndarray, *, label: str):
        return self._sample_trajectory(
            waypoints,
            duration_s=self.approach_duration_s,
            sample_dt=self.approach_sample_dt,
            label=label,
            parameterizer=self.approach_time_parameterizer,
        )

    def _log_trajectory_timing(self, label: str, trajectory) -> None:
        diagnostics = getattr(trajectory, "diagnostics", {}) or {}
        if diagnostics.get("mode") == "toppra":
            self.get_logger().info(
                "trajectory retime "
                f"{label} "
                f"mode=toppra "
                f"duration_s={float(diagnostics.get('duration_s', trajectory.timestamps[-1])):.3f} "
                f"samples={int(trajectory.timestamps.shape[0])} "
                f"gridpoints={int(diagnostics.get('gridpoints', 0))} "
                f"path_length={float(diagnostics.get('path_length', 0.0)):.3f} "
                f"limit_scale={float(diagnostics.get('limit_scale', 1.0)):.3f}"
            )
            return
        self.get_logger().info(
            "trajectory retime "
            f"{label} "
            f"mode={diagnostics.get('mode', type(trajectory).__name__)} "
            f"duration_s={float(trajectory.timestamps[-1]):.3f} "
            f"samples={int(trajectory.timestamps.shape[0])}"
        )

    def _select_approach_goal(self, station_pose, initial_joints, collision_checker, ik_branches):
        rolls = np.arange(360, dtype=float) * (np.pi / 180.0)
        approach_state = AssemblyState(axial_offset_m=self.insert_offset_m)
        targets = self.type3_planner.task_poses(
            AssemblyPath(approach_state.as_array()[None, :]),
            np.asarray(station_pose, dtype=float),
            rolls,
        )
        joints_batch, valid_batch = self.arm.solve_ik(targets, branches=ik_branches)
        valid_flat = valid_batch.ravel()
        joints_flat = joints_batch.reshape(-1, 6)
        valid_idx = np.flatnonzero(valid_flat)
        if len(valid_idx) == 0:
            raise RuntimeError("approach IK search found no valid candidates")

        candidate_joints = self._normalize_for_type2_bounds(joints_flat[valid_idx])
        collides = collision_checker.check_configs(candidate_joints)
        free_joints = candidate_joints[~collides]
        free_flat_idx = valid_idx[~collides]
        if len(free_flat_idx) == 0:
            raise RuntimeError("approach IK search found no collision-free candidates")

        delta = free_joints - initial_joints[None, :]
        costs = self.motion_l1_weight * np.sum(np.abs(delta), axis=1)
        costs += self.motion_l2_weight * np.sum(delta * delta, axis=1)
        best_local = int(np.argmin(costs))
        best_flat = int(free_flat_idx[best_local])
        roll_idx = best_flat // joints_batch.shape[1]
        return free_joints[best_local], float(costs[best_local]), float(rolls[roll_idx])

    def _normalize_for_type2_bounds(self, joints: np.ndarray) -> np.ndarray:
        joints = self.arm.joint_space.wrap(joints)
        bounded = self.arm.joint_space.bounded
        below = bounded & (joints < self.joint_lower - self._type2_bound_tolerance)
        above = bounded & (joints > self.joint_upper + self._type2_bound_tolerance)
        if np.any(below) or np.any(above):
            raise ValueError("Type II joint state is outside configured joint bounds")
        joints[..., bounded] = np.clip(joints[..., bounded], self.joint_lower[bounded], self.joint_upper[bounded])
        return joints

    def _on_recovery_request(self, msg: RecoveryPlanningRequest) -> None:
        result = RecoveryPlanningResult()
        result.header.stamp = self.get_clock().now().to_msg()
        result.header.frame_id = "arm_base"
        result.request_id = msg.request_id
        result.mode = int(msg.mode)
        result.degraded = False

        if msg.station_pose_arm_base.header.frame_id != "arm_base":
            raise ValueError("station_pose_arm_base must be in arm_base")
        if msg.local_pose_arm_base.header.frame_id != "arm_base":
            raise ValueError("local_pose_arm_base must be in arm_base")
        t_arm_base_station = transform_from_pose_stamped(msg.station_pose_arm_base)
        t_arm_base_local = transform_from_pose_stamped(msg.local_pose_arm_base)
        initial_joints = joint_positions_from_joint_state(msg.initial_joint_state)
        goal_joints = joint_positions_from_joint_state(msg.goal_joint_state)
        collision_checker = self._make_approach_withdraw_checker(t_arm_base_station, t_arm_base_local)
        local_rotation = t_arm_base_local[:3, :3]
        exchange_local_x_axis = local_rotation[:, 0]
        _, initial_tcp_rotation = self._tcp_pose(initial_joints)
        tcp_contact_minus_z_axis = -initial_tcp_rotation[:, 2]

        self.get_logger().info(
            "recovery request "
            f"id={msg.request_id} "
            f"mode={int(msg.mode)} "
            f"initial_joints={initial_joints.tolist()} "
            f"goal_joints={goal_joints.tolist()} "
            f"exchange_local_x_axis={np.round(exchange_local_x_axis, 6).tolist()} "
            f"tcp_contact_minus_z_axis={np.round(tcp_contact_minus_z_axis, 6).tolist()} "
            f"local_exchange_meshes={len(self.collision_model.local_exchange_meshes)} "
            f"bit_timeout_s={self.recovery_bit_timeout_s:.3f}"
        )
        try:
            if int(msg.mode) == RecoveryPlanningRequest.MODE_EXTRACT_FOR_RETRY:
                waypoints, degraded, message = self._plan_extract_for_retry(
                    t_arm_base_local,
                    initial_joints,
                )
            elif int(msg.mode) == RecoveryPlanningRequest.MODE_SIDE_RELEASE:
                waypoints = self._plan_side_release_segment(
                    initial_joints,
                )
                degraded = False
                message = "ok"
            elif int(msg.mode) == RecoveryPlanningRequest.MODE_RETRACT_TO_APPROACH_START:
                waypoints = self._plan_retract_to_approach_start(
                    initial_joints,
                    goal_joints,
                    collision_checker,
                )
                degraded = False
                message = "ok"
            else:
                raise ValueError(f"unsupported recovery mode: {int(msg.mode)}")
        except RuntimeError as exc:
            result.success = False
            result.message = str(exc)
            result.cost = float("inf")
            self.get_logger().warning(f"recovery planning failed id={msg.request_id}: {result.message}")
            self.recovery_result_pub.publish(result)
            return

        if waypoints is None:
            result.success = False
            result.degraded = bool(degraded)
            result.message = str(message)
            result.cost = float("inf")
            self.get_logger().warning(
                f"recovery planning degraded id={msg.request_id}: {result.message}"
            )
            self.recovery_result_pub.publish(result)
            return

        dense_trajectory = self._sample_recovery_trajectory(
            waypoints,
            label=f"recovery id={msg.request_id} mode={int(msg.mode)}",
        )
        result.success = True
        result.degraded = bool(degraded)
        result.message = str(message)
        result.cost = float(waypoints.shape[0])
        result.trajectory = dense_trajectory_to_msg(
            dense_trajectory,
            stamp=result.header.stamp,
        )
        self.get_logger().info(
            "recovery planning success "
            f"id={msg.request_id} "
            f"mode={int(msg.mode)} "
            f"degraded={bool(degraded)} "
            f"waypoint_steps={int(waypoints.shape[0])} "
            f"dense_steps={int(dense_trajectory.positions.shape[0])}"
        )
        self.recovery_result_pub.publish(result)

    def _make_station_checker(self, station_pose):
        return CollisionChecker(
            self.arm,
            [(self.collision_model.station_meshes, station_pose)],
            self.collision_model.arm_capsules,
        )

    def _make_approach_withdraw_checker(self, station_pose, local_pose):
        if not self.collision_model.local_exchange_meshes:
            return self._make_station_checker(station_pose)
        return CollisionChecker(
            self.arm,
            [
                (self.collision_model.station_meshes, station_pose),
                (self.collision_model.local_exchange_meshes, local_pose),
            ],
            self.collision_model.arm_capsules,
        )

    def _plan_extract_for_retry(
        self,
        local_pose,
        initial_joints: np.ndarray,
    ) -> tuple[np.ndarray | None, bool, str]:
        try:
            extract_waypoints = self._plan_extract_segment(local_pose, initial_joints)
        except RuntimeError as exc:
            if self.recovery_allow_degraded_release:
                self.get_logger().warning(f"extract-for-retry failed, requesting side-release fallback: {exc}")
                return None, True, f"recover extract failed; release required: {exc}"
            raise
        return extract_waypoints, False, "ok"

    def _plan_retract_to_approach_start(
        self,
        initial_joints: np.ndarray,
        goal_joints: np.ndarray,
        collision_checker,
    ) -> np.ndarray:
        return self._plan_bitstar_segment(
            initial_joints,
            goal_joints,
            collision_checker,
            timeout_s=self.recovery_bit_timeout_s,
        )

    def _plan_extract_segment(self, t_arm_base_local: np.ndarray, initial_joints: np.ndarray) -> np.ndarray:
        tcp_position, tcp_rotation = self._tcp_pose(initial_joints)
        local_rotation = t_arm_base_local[:3, :3]
        direction = local_rotation[:, 0]
        return self._plan_cartesian_offset_segment(
            initial_joints,
            tcp_position,
            tcp_rotation,
            direction,
            distance_m=self.recovery_extract_distance_m,
            step_m=self.recovery_extract_step_m,
            collision_checker=None,
            label="extract",
        )

    def _plan_side_release_segment(self, initial_joints: np.ndarray) -> np.ndarray:
        tcp_position, tcp_rotation = self._tcp_pose(initial_joints)
        direction = -tcp_rotation[:, 2]
        self.get_logger().info(
            "side-release uses TCP contact direction "
            f"tcp_contact_minus_z_axis={np.round(direction, 6).tolist()}"
        )
        return self._plan_cartesian_offset_segment(
            initial_joints,
            tcp_position,
            tcp_rotation,
            direction,
            distance_m=self.recovery_side_release_distance_m,
            step_m=self.recovery_side_release_step_m,
            collision_checker=None,
            label="side_release",
        )

    def _plan_cartesian_offset_segment(
        self,
        initial_joints: np.ndarray,
        start_position: np.ndarray,
        rotation: np.ndarray,
        direction: np.ndarray,
        *,
        distance_m: float,
        step_m: float,
        collision_checker,
        label: str,
    ) -> np.ndarray:
        direction = np.asarray(direction, dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            raise RuntimeError(f"{label} direction has zero length")
        direction = direction / norm
        steps = max(1, int(np.ceil(float(distance_m) / float(step_m))))
        distances = np.linspace(0.0, float(distance_m), steps + 1)
        positions = np.asarray(start_position, dtype=float)[None, :] + distances[:, None] * direction[None, :]
        rotations = np.repeat(np.asarray(rotation, dtype=float)[None, :, :], positions.shape[0], axis=0)
        return self._solve_cartesian_waypoints(
            positions,
            rotations,
            initial_joints,
            collision_checker,
            label=label,
        )

    def _solve_cartesian_waypoints(
        self,
        positions: np.ndarray,
        rotations: np.ndarray,
        reference_joints: np.ndarray,
        collision_checker,
        *,
        label: str,
    ) -> np.ndarray:
        waypoints = []
        reference = np.asarray(reference_joints, dtype=float)
        for idx in range(positions.shape[0]):
            best, roll_delta, cost = self._solve_cartesian_waypoint(
                positions[idx],
                rotations[idx],
                reference,
                collision_checker,
                label=label,
                waypoint_index=idx,
            )
            waypoints.append(best)
            reference = best
            self.get_logger().debug(
                f"{label} waypoint {idx} "
                f"roll_delta_deg={np.rad2deg(roll_delta):.3f} "
                f"cost={cost:.6f}"
            )
        return np.asarray(waypoints, dtype=float)

    def _solve_cartesian_waypoint(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
        reference: np.ndarray,
        collision_checker,
        *,
        label: str,
        waypoint_index: int,
    ) -> tuple[np.ndarray, float, float]:
        ik_branches = tuple(self.ik_branches) if self.ik_branches else None
        best_result = None
        saw_ik_solution = False
        saw_collision_free = False
        rejected_by_collision = 0
        for roll_shell in self._recovery_roll_search_shells():
            for roll_delta in roll_shell:
                candidate_rotation = np.asarray(rotation, dtype=float) @ self._roll_x_matrix(roll_delta)
                target = np.eye(4, dtype=float)[None, :, :]
                target[0, :3, :3] = candidate_rotation
                target[0, :3, 3] = position
                joints_batch, valid_batch = self.arm.solve_ik(target, branches=ik_branches)
                valid = valid_batch[0]
                if not np.any(valid):
                    continue
                saw_ik_solution = True
                candidates = joints_batch[0, valid, :]
                candidates = self.arm.joint_space.align_to_reference(
                    candidates,
                    np.broadcast_to(reference, candidates.shape),
                )
                if collision_checker is None:
                    free = candidates
                else:
                    collides = collision_checker.check_configs(candidates)
                    rejected_by_collision += int(np.count_nonzero(collides))
                    free = candidates[~collides]
                    if free.size == 0:
                        continue
                saw_collision_free = True
                costs = np.sum(
                    np.abs(self.arm.joint_space.delta(free, np.broadcast_to(reference, free.shape))),
                    axis=1,
                )
                local_idx = int(np.argmin(costs))
                result = (free[local_idx], float(roll_delta), float(costs[local_idx]))
                if best_result is None or result[2] < best_result[2]:
                    best_result = result
            if best_result is not None:
                return best_result
        reason = "no_ik_solution"
        if saw_ik_solution and not saw_collision_free:
            reason = f"all_ik_candidates_collide rejected={rejected_by_collision}"
        raise RuntimeError(
            f"{label} IK/collision failed at waypoint {waypoint_index} "
            f"reason={reason} "
            f"tcp_pos={np.round(np.asarray(position, dtype=float), 6).tolist()} "
            f"roll_search_max_deg={np.rad2deg(self.recovery_roll_search_max_rad):.3f}"
        )

    def _recovery_roll_search_shells(self):
        yield (0.0,)
        max_roll = self.recovery_roll_search_max_rad
        if max_roll <= 0.0:
            return

        step = self.recovery_roll_search_step_rad
        angle = min(step, max_roll)
        while angle <= max_roll + 1e-12:
            yield (angle, -angle)
            if np.isclose(angle, max_roll):
                break
            angle = min(angle + step, max_roll)

    def _roll_x_matrix(self, angle: float) -> np.ndarray:
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        return np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, c, -s],
                [0.0, s, c],
            ],
            dtype=float,
        )

    def _plan_bitstar_segment(
        self,
        start_joints: np.ndarray,
        goal_joints: np.ndarray,
        collision_checker,
        *,
        timeout_s: float,
    ) -> np.ndarray:
        start = self._normalize_for_type2_bounds(np.asarray(start_joints, dtype=float)[None, :])[0]
        goal = self._normalize_for_type2_bounds(np.asarray(goal_joints, dtype=float)[None, :])[0]
        return plan_joint_path_bitstar(
            start,
            goal,
            validity_fn=lambda q: not bool(collision_checker.check_configs(np.asarray(q, dtype=float)[None, :])[0]),
            joint_lower=self.joint_lower,
            joint_upper=self.joint_upper,
            timeout_s=float(timeout_s),
        )

    def _tcp_pose(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tcp = self.arm.forward_kinematics(np.asarray(joints, dtype=float)[None, :]).tcp_transforms[0]
        return tcp[:3, 3].copy(), tcp[:3, :3].copy()

    def _log_type3_graph_diagnostics(self, plan) -> None:
        diagnostics = plan.diagnostics
        self.get_logger().info(
            "graph built "
            f"valid_nodes={int(diagnostics['final_valid_nodes'])}/{int(diagnostics['total_nodes'])} "
            f"ik_valid_nodes={int(diagnostics['ik_valid_nodes'])}/{int(diagnostics['total_nodes'])} "
            f"collision_checked={int(diagnostics['collision_checked_nodes'])} "
            f"collision_rejected={int(diagnostics['collision_rejected_nodes'])} "
            f"continuation_checked={int(diagnostics.get('continuation_checked_nodes', 0))} "
            f"continuation_rejected={int(diagnostics.get('continuation_rejected_nodes', 0))} "
            f"path_samples={int(diagnostics['path_samples'])} "
            f"roll_count={int(diagnostics['roll_count'])} "
            f"ik_branch={int(diagnostics['ik_branch'])}"
        )
        self.get_logger().info(
            "graph step diagnostics "
            f"ik_valid={_step_counts_summary(diagnostics['ik_valid_by_step'])} "
            f"final_valid={_step_counts_summary(diagnostics['final_valid_by_step'])} "
            f"collision_rejected={_step_counts_summary(diagnostics['collision_rejected_by_step'])}"
        )
        self.get_logger().info(
            "graph continuation diagnostics "
            f"reasons={_reason_counts_summary(diagnostics.get('continuation_reject_reasons'))}"
        )

    def _log_type3_best_effort_repair(self, request_id: str, plan) -> None:
        repair = plan.diagnostics.get("repair")
        if repair is not None and repair["success"]:
            self.get_logger().warning(
                "strict Type III Viterbi search failed; using best-effort repair "
                f"id={request_id} "
                f"cost={float(plan.cost):.6f} "
                f"repaired_nodes={int(repair['repair_count'])} "
                f"max_pos_mm={float(repair['max_pos_error_m']) * 1000.0:.3f} "
                f"max_axis_deg={np.rad2deg(float(repair['max_axis_error_rad'])):.3f} "
                f"max_roll_deg={np.rad2deg(float(repair['max_roll_error_rad'])):.3f} "
                f"max_runtime_ms={float(repair['max_runtime_ms']):.3f}"
            )
        elif repair is not None:
            self.get_logger().warning(
                "best-effort repair failed "
                f"id={request_id} collisions={int(repair['collision_count'])}"
            )

    def _publish_type3_failure(self, result: Type3PlanningResult, plan, request_id: str) -> None:
        result.success = False
        result.message = plan.message
        result.cost = float(plan.cost)
        diagnostics = plan.diagnostics
        final_valid_by_step = np.asarray(diagnostics["final_valid_by_step"], dtype=np.int64)
        self.get_logger().warning(
            f"planning failed id={request_id} cost={float(plan.cost):.6f} "
            f"message={plan.message} "
            f"final_valid={_step_counts_summary(final_valid_by_step)}"
        )
        self.result_pub.publish(result)

    def _fill_q_sweep_result(self, result: Type3PlanningResult, q_sweep_cache, *, stamp) -> None:
        if q_sweep_cache is None:
            return
        states = np.asarray(q_sweep_cache["states"], dtype=float)
        result.q_sweep_s = [float(v) for v in states[:, 3] / (0.5 * np.pi)]
        result.q_sweep_trajectory = self._q_sweep_trajectory_to_msg(
            np.asarray(q_sweep_cache["joints"], dtype=float),
            stamp=stamp,
        )

    def _assembly_paths(self, path_name: str) -> tuple[AssemblyPath, tuple[AssemblyPath, ...]]:
        inserted = AssemblyState()
        slid = AssemblyState(slide_m=self.slide_distance_m)
        p_done = AssemblyState(slide_m=self.slide_distance_m, p_angle_rad=0.5 * np.pi)
        if path_name == "slide":
            return AssemblyPath.between(inserted, slid), ()
        if path_name == "p":
            step_deg = float(self.type3_planner.continuation_config.get("sample_step_deg", 5.0))
            q_angles = np.deg2rad(np.arange(0.0, 90.0 + 0.5 * step_deg, step_deg))
            plus_states = np.repeat(p_done.as_array()[None, :], len(q_angles), axis=0)
            minus_states = plus_states.copy()
            plus_states[:, 3] = q_angles
            minus_states[:, 3] = -q_angles
            return (
                AssemblyPath.between(slid, p_done),
                (AssemblyPath(minus_states), AssemblyPath(plus_states)),
            )
        raise ValueError(f"unsupported Type III path: {path_name!r}")

    def _on_request(self, msg: Type3PlanningRequest) -> None:
        result = Type3PlanningResult()
        result.header.stamp = self.get_clock().now().to_msg()
        result.header.frame_id = "arm_base"
        result.request_id = msg.request_id

        if msg.station_pose_arm_base.header.frame_id != "arm_base":
            raise ValueError("station_pose_arm_base must be in arm_base")
        t_arm_base_station = transform_from_pose_stamped(msg.station_pose_arm_base)

        path_name = str(msg.path_name).strip()
        path, continuations = self._assembly_paths(path_name)
        result.path_name = path_name
        result.selected_roll = 0.0
        result.selected_ik_branch = -1

        initial_joints = joint_positions_from_joint_state(msg.initial_joint_state)
        self.get_logger().info(
            "planning request "
            f"id={msg.request_id} "
            f"path={path_name} "
            f"station_pos={t_arm_base_station[:3, 3].tolist()} "
            "station_rpy_zyx_deg="
            f"{_zyx_rpy_degrees(quaternions_from_rotations(t_arm_base_station[None, :3, :3])[0]).tolist()} "
            f"initial_joints={initial_joints.tolist()} "
            f"ik_branch={self.ik_branches[0]} "
            f"motion_l1_weight={self.motion_l1_weight:.4f} "
            f"motion_l2_weight={self.motion_l2_weight:.4f}"
        )

        collision_checker = self._make_station_checker(t_arm_base_station)
        plan = self.type3_planner.plan(
            path,
            t_arm_base_station,
            initial_joints=initial_joints,
            collision_checker=collision_checker,
            terminal_continuations=continuations,
        )
        self._log_type3_graph_diagnostics(plan)
        self._log_type3_best_effort_repair(msg.request_id, plan)
        if not plan.success:
            self._publish_type3_failure(result, plan, msg.request_id)
            return

        path = plan.path
        if path is None:
            raise RuntimeError("successful TypeIII plan missing path")
        rolls = plan.rolls
        result.selected_roll = float(rolls[-1])
        result.selected_ik_branch = self.ik_branches[0]
        waypoints = plan.waypoints
        q_sweep_cache = plan.continuation
        dense_trajectory = self._sample_trajectory(
            waypoints,
            duration_s=self.trajectory_duration_s,
            sample_dt=self.trajectory_sample_dt,
            label=f"type3 id={msg.request_id}",
            parameterizer=self.time_parameterizer,
        )
        result.success = True
        result.message = plan.message
        result.cost = float(plan.cost)
        result.trajectory = dense_trajectory_to_msg(
            dense_trajectory,
            stamp=result.header.stamp,
        )
        self._fill_q_sweep_result(
            result,
            q_sweep_cache,
            stamp=result.header.stamp,
        )
        self.get_logger().info(
            "planning success "
            f"id={msg.request_id} "
            f"cost={float(plan.cost):.6f} "
            f"path_len={len(path)} "
            f"waypoint_steps={int(waypoints.shape[0])} "
            f"dense_steps={int(dense_trajectory.positions.shape[0])} "
            f"duration_s={float(dense_trajectory.timestamps[-1]):.3f} "
            f"sample_dt={self.trajectory_sample_dt:.4f} "
            f"q_sweep_points={len(result.q_sweep_s)} "
            f"best_effort={plan.diagnostics.get('repair') is not None} "
            "q_sweep_mode=strict "
            f"rolls={list(rolls)}"
        )
        self.result_pub.publish(result)

    def _q_sweep_trajectory_to_msg(self, joints: np.ndarray, *, stamp) -> JointTrajectory:
        joints = np.asarray(joints, dtype=float)
        if joints.ndim != 2 or joints.shape[1] != len(JOINT_NAMES):
            raise ValueError(f"Q sweep joints must have shape (N, {len(JOINT_NAMES)}), got {joints.shape}")
        msg = JointTrajectory()
        msg.header.stamp = stamp
        msg.joint_names = list(JOINT_NAMES)
        for i, q in enumerate(joints):
            point = JointTrajectoryPoint()
            point.positions = [float(v) for v in q]
            point.velocities = [0.0] * len(JOINT_NAMES)
            point.accelerations = [0.0] * len(JOINT_NAMES)
            point.time_from_start = seconds_to_duration(float(i))
            msg.points.append(point)
        return msg


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
