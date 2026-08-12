from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from arm_exchange_core.arm_model import ArmModel
from arm_exchange_core.planning.viterbi import solve_viterbi
from arm_exchange_core.transform import validate_transforms


@dataclass(frozen=True, slots=True)
class AssemblyState:
    """Physical coordinates that define one pose on the assembly manifold."""

    axial_offset_m: float = 0.0
    slide_m: float = 0.0
    p_angle_rad: float = 0.0
    q_angle_rad: float = 0.0
    release_m: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.asarray(
            (
                self.axial_offset_m,
                self.slide_m,
                self.p_angle_rad,
                self.q_angle_rad,
                self.release_m,
            ),
            dtype=float,
        )


@dataclass(frozen=True, slots=True)
class AssemblyPath:
    """Ordered task-space states; rows are physical assembly coordinates."""

    states: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.states, dtype=float)
        if values.ndim != 2 or values.shape[1] != 5 or len(values) < 1:
            raise ValueError(f"AssemblyPath.states must have shape (N, 5), N >= 1; got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("AssemblyPath.states must be finite")
        object.__setattr__(self, "states", values)

    @classmethod
    def between(
        cls,
        start: AssemblyState,
        goal: AssemblyState,
        samples: int = 100,
    ) -> "AssemblyPath":
        if samples < 2:
            raise ValueError("AssemblyPath requires at least two samples")
        fraction = np.linspace(0.0, 1.0, int(samples))[:, None]
        start_values = start.as_array()
        return cls(start_values + fraction * (goal.as_array() - start_values))


@dataclass(frozen=True, slots=True)
class Type3PlanResult:
    success: bool
    message: str
    cost: float
    waypoints: np.ndarray
    rolls: np.ndarray
    path: list[tuple[int, int]] | None
    continuation: dict[str, object] | None
    diagnostics: dict[str, object]


@dataclass(slots=True)
class _LayeredGraph:
    rolls: np.ndarray
    targets: np.ndarray
    joints: np.ndarray
    valid: np.ndarray
    penalties: np.ndarray


class Type3Planner:
    """Plan one IK branch over a roll-discretized assembly path."""

    def __init__(
        self,
        arm: ArmModel,
        *,
        geometry: dict,
        ik_branch: int,
        roll_sample_step_deg: float,
        bandwidth: int,
        collision_soft_margin: float,
        collision_soft_weight: float,
        motion_l1_weight: float,
        motion_l2_weight: float,
        joint_limit_margin_rad: float,
        joint_limit_weight: float,
        best_effort: dict | None = None,
        continuation: dict | None = None,
    ) -> None:
        self.arm = arm
        self.p_radius = float(geometry["r_p"])
        self.q_radius = float(geometry["r_q"])
        self.p_offset = float(geometry["phi"])
        self.ik_branch = int(ik_branch)
        if self.ik_branch not in range(8):
            raise ValueError("ik_branch must be in [0, 7]")
        self.rolls = np.deg2rad(
            np.arange(0.0, 360.0, float(roll_sample_step_deg), dtype=float)
        )
        if len(self.rolls) == 0:
            raise ValueError("roll_sample_step_deg produced an empty roll grid")
        self.bandwidth = int(bandwidth)
        if self.bandwidth < 0:
            raise ValueError("bandwidth must be non-negative")
        self.soft_margin = float(collision_soft_margin)
        self.soft_weight = float(collision_soft_weight)
        self.motion_l1_weight = float(motion_l1_weight)
        self.motion_l2_weight = float(motion_l2_weight)
        self.limit_margin = float(joint_limit_margin_rad)
        self.limit_weight = float(joint_limit_weight)
        self.best_effort = dict(best_effort or {})
        self.continuation_config = dict(continuation or {})

    def plan(
        self,
        path: AssemblyPath,
        station_transform: np.ndarray,
        initial_joints: np.ndarray,
        collision_checker,
        *,
        terminal_continuations: tuple[AssemblyPath, ...] = (),
    ) -> Type3PlanResult:
        station = validate_transforms(
            np.asarray(station_transform, dtype=float)[None, :, :],
            name="station_transform",
        )[0]
        initial = np.asarray(initial_joints, dtype=float)
        if initial.shape != (6,):
            raise ValueError(f"initial_joints must have shape (6,), got {initial.shape}")

        graph, diagnostics = self._build_graph(path, station, collision_checker)
        continuation_cache: dict[int, dict[str, object]] = {}
        if terminal_continuations:
            continuation_cache, continuation_diagnostics = self._filter_terminal_continuations(
                graph,
                terminal_continuations,
                station,
                collision_checker,
            )
            diagnostics.update(continuation_diagnostics)

        path_indices, cost = self._search(graph, initial, allow_invalid=False)
        repair_diagnostics = None
        waypoints = None
        if path_indices is None and bool(self.best_effort.get("enabled", False)):
            path_indices, cost = self._search(graph, initial, allow_invalid=True)
            if path_indices is not None:
                waypoints, repair_diagnostics = self._repair(
                    graph,
                    path_indices,
                    initial,
                    collision_checker,
                )
                if not bool(repair_diagnostics["success"]):
                    path_indices = None

        diagnostics["repair"] = repair_diagnostics
        if path_indices is None:
            return Type3PlanResult(
                success=False,
                message=self._failure_message(graph, diagnostics),
                cost=float(cost),
                waypoints=np.zeros((0, 6), dtype=float),
                rolls=np.zeros(0, dtype=float),
                path=None,
                continuation=None,
                diagnostics=diagnostics,
            )

        if waypoints is None:
            raw = np.asarray([graph.joints[i, j] for i, j in path_indices])
            waypoints = self.arm.joint_space.align_trajectory(raw, initial[None, :])
        selected_rolls = np.unwrap(np.asarray([graph.rolls[j] for _, j in path_indices]))
        continuation = None
        if terminal_continuations and repair_diagnostics is not None:
            branches = tuple(
                self._evaluate_continuation(
                    candidate,
                    station,
                    float(graph.rolls[path_indices[-1][1]]),
                    waypoints[-1],
                    collision_checker,
                )[0]
                for candidate in terminal_continuations
            )
            if all(branch is not None for branch in branches):
                continuation = self._combine_continuations(
                    terminal_continuations,
                    branches,
                )
        elif terminal_continuations:
            continuation = continuation_cache.get(path_indices[-1][1])
        if terminal_continuations:
            if continuation is None:
                return Type3PlanResult(
                    success=False,
                    message="selected terminal has no feasible continuation",
                    cost=float(cost),
                    waypoints=waypoints,
                    rolls=selected_rolls,
                    path=path_indices,
                    continuation=None,
                    diagnostics=diagnostics,
                )
        repair_count = 0 if repair_diagnostics is None else int(repair_diagnostics["repair_count"])
        message = "ok" if repair_count == 0 else f"ok best_effort repaired_nodes={repair_count}"
        return Type3PlanResult(
            success=True,
            message=message,
            cost=float(cost),
            waypoints=waypoints,
            rolls=selected_rolls,
            path=path_indices,
            continuation=continuation,
            diagnostics=diagnostics,
        )

    def task_poses(
        self,
        path: AssemblyPath,
        station_transform: np.ndarray,
        rolls: np.ndarray,
    ) -> np.ndarray:
        states = np.asarray(path.states, dtype=float)
        roll = np.asarray(rolls, dtype=float).reshape(-1)
        batch_size = max(len(states), len(roll))
        if len(states) not in (1, batch_size) or len(roll) not in (1, batch_size):
            raise ValueError("path states and rolls must have equal batch sizes or a singleton batch")
        states = np.broadcast_to(states, (batch_size, 5))
        roll = np.broadcast_to(roll, (batch_size,))
        station = validate_transforms(
            np.asarray(station_transform, dtype=float)[None, :, :],
            name="station_transform",
        )[0]
        return self._poses(states, roll, station)

    def _build_graph(self, path: AssemblyPath, station: np.ndarray, collision_checker):
        n_steps = len(path.states)
        n_rolls = len(self.rolls)
        states = np.repeat(path.states, n_rolls, axis=0)
        rolls = np.tile(self.rolls, n_steps)
        targets = self._poses(states, rolls, station)
        joints, valid = self.arm.solve_ik(targets, branches=(self.ik_branch,))
        joints = joints[:, 0].reshape(n_steps, n_rolls, 6)
        ik_valid = valid[:, 0].reshape(n_steps, n_rolls)
        valid = ik_valid.copy()
        penalties = np.zeros((n_steps, n_rolls), dtype=float)

        valid_flat = valid.reshape(-1)
        selected = np.flatnonzero(valid_flat)
        collision_rejected = np.zeros(valid_flat.shape, dtype=bool)
        if collision_checker is not None and len(selected):
            collides, collision_penalty = collision_checker.check_configs_with_penalty(
                joints.reshape(-1, 6)[selected],
                self.soft_margin,
                self.soft_weight,
            )
            collision_rejected[selected[collides]] = True
            valid_flat[selected[collides]] = False
            penalties.reshape(-1)[selected[~collides]] = collision_penalty[~collides]
        valid = valid_flat.reshape(n_steps, n_rolls)
        penalties += self.arm.joint_space.limit_penalty(
            joints,
            self.limit_margin,
            self.limit_weight,
        )
        graph = _LayeredGraph(
            rolls=self.rolls,
            targets=targets.reshape(n_steps, n_rolls, 4, 4),
            joints=joints,
            valid=valid,
            penalties=penalties,
        )
        diagnostics = {
            "total_nodes": int(valid.size),
            "ik_valid_nodes": int(np.count_nonzero(ik_valid)),
            "final_valid_nodes": int(np.count_nonzero(valid)),
            "collision_checked_nodes": int(len(selected)),
            "collision_rejected_nodes": int(np.count_nonzero(collision_rejected)),
            "ik_valid_by_step": np.count_nonzero(ik_valid, axis=1),
            "final_valid_by_step": np.count_nonzero(valid, axis=1),
            "collision_rejected_by_step": np.count_nonzero(
                collision_rejected.reshape(n_steps, n_rolls), axis=1
            ),
            "roll_count": n_rolls,
            "path_samples": n_steps,
            "ik_branch": self.ik_branch,
        }
        return graph, diagnostics

    def _search(self, graph: _LayeredGraph, initial: np.ndarray, *, allow_invalid: bool):
        n_steps, n_rolls = graph.valid.shape
        offsets = np.arange(-self.bandwidth, self.bandwidth + 1)
        previous_indices = (np.arange(n_rolls)[:, None] + offsets[None, :]) % n_rolls
        invalid_penalty = float(self.best_effort.get("invalid_node_penalty", 1000.0))
        roll_weight = float(self.best_effort.get("roll_motion_weight", 1.0))

        def node_cost(layer: int) -> np.ndarray:
            if allow_invalid:
                return np.where(graph.valid[layer], graph.penalties[layer], invalid_penalty)
            return np.where(graph.valid[layer], graph.penalties[layer], np.inf)

        initial_delta = self.arm.joint_space.delta(
            graph.joints[0],
            np.broadcast_to(initial, graph.joints[0].shape),
        )
        initial_motion = self._motion_cost(initial_delta)
        initial_cost = node_cost(0) + np.where(graph.valid[0], initial_motion, 0.0)
        if np.any(graph.valid[0]):
            nearest = int(np.argmin(np.where(graph.valid[0], initial_motion, np.inf)))
            locked = np.full(n_rolls, np.inf)
            locked[nearest] = initial_cost[nearest]
            initial_cost = locked

        def step(layer: int, previous_cost: np.ndarray):
            previous_joints = graph.joints[layer - 1][previous_indices]
            both_valid = graph.valid[layer, :, None] & graph.valid[layer - 1][previous_indices]
            delta = self.arm.joint_space.delta(
                graph.joints[layer, :, None, :],
                previous_joints,
            )
            transition = np.where(both_valid, self._motion_cost(delta), 0.0)
            if allow_invalid:
                roll_delta = np.abs(np.arange(n_rolls)[:, None] - previous_indices)
                transition += roll_weight * np.minimum(roll_delta, n_rolls - roll_delta)
            total = previous_cost[previous_indices] + transition + node_cost(layer)[:, None]
            best = np.argmin(total, axis=1)
            return total[np.arange(n_rolls), best], previous_indices[np.arange(n_rolls), best]

        result = solve_viterbi(initial_cost, n_steps, step)
        if not result.success:
            return None, float("inf")
        return [(i, int(j)) for i, j in enumerate(result.path_indices)], float(result.cost)

    def _repair(self, graph, path, initial, collision_checker):
        repaired = []
        q_ref = np.asarray(initial, dtype=float)
        invalid_mask = []
        metrics = []
        for layer, roll_index in path:
            if graph.valid[layer, roll_index]:
                q = graph.joints[layer, roll_index]
                metric = None
            else:
                q, metric = self._approximate_ik(graph.targets[layer, roll_index], q_ref)
            q = self.arm.joint_space.align_to_reference(q[None, :], q_ref[None, :])[0]
            repaired.append(q)
            invalid_mask.append(not bool(graph.valid[layer, roll_index]))
            if metric is not None:
                metrics.append(metric)
            q_ref = q
        joints = np.asarray(repaired)
        collisions = (
            np.zeros(len(joints), dtype=bool)
            if collision_checker is None
            else collision_checker.check_configs(joints)
        )
        return joints, {
            "success": not bool(np.any(collisions)),
            "repair_count": int(np.count_nonzero(invalid_mask)),
            "collision_count": int(np.count_nonzero(collisions)),
            "max_pos_error_m": max((m[0] for m in metrics), default=0.0),
            "max_axis_error_rad": max((m[1] for m in metrics), default=0.0),
            "max_roll_error_rad": max((m[2] for m in metrics), default=0.0),
            "max_runtime_ms": max((m[3] for m in metrics), default=0.0),
        }

    def _approximate_ik(self, target: np.ndarray, q_ref: np.ndarray):
        config = self.best_effort
        position_sigma = float(config.get("position_sigma_m", 0.005))
        axis_sigma = np.deg2rad(float(config.get("axis_sigma_deg", 5.0)))
        reference_sigma = np.deg2rad(float(config.get("joint_reference_sigma_deg", 30.0)))
        roll_free = bool(config.get("roll_free", True))

        def errors(q):
            tcp = self.arm.forward_kinematics(np.asarray(q)[None, :]).tcp_transforms[0]
            rotation_error = Rotation.from_matrix(target[:3, :3].T @ tcp[:3, :3]).as_rotvec()
            roll_error = float(rotation_error[0])
            axis_error = rotation_error.copy()
            axis_error[0] = 0.0
            return tcp, axis_error, roll_error

        def residual(q):
            tcp, axis_error, roll_error = errors(q)
            terms = [(tcp[:3, 3] - target[:3, 3]) / position_sigma, axis_error / axis_sigma]
            if not roll_free:
                terms.append(np.asarray([roll_error / np.deg2rad(90.0)]))
            delta = self.arm.joint_space.delta(q[None, :], q_ref[None, :])[0]
            terms.append(delta / reference_sigma)
            return np.concatenate(terms)

        initial = np.clip(
            np.asarray(q_ref, dtype=float),
            self.arm.joint_space.lower,
            self.arm.joint_space.upper,
        )
        started = time.perf_counter()
        solution = least_squares(
            residual,
            initial,
            bounds=(self.arm.joint_space.lower, self.arm.joint_space.upper),
            method="trf",
            max_nfev=int(config.get("max_nfev", 40)),
            x_scale="jac",
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        q = np.asarray(solution.x)
        tcp, axis_error, roll_error = errors(q)
        metric = (
            float(np.linalg.norm(tcp[:3, 3] - target[:3, 3])),
            float(np.linalg.norm(axis_error)),
            float(abs(roll_error)),
            (time.perf_counter() - started) * 1000.0,
        )
        return q, metric

    def _filter_terminal_continuations(
        self,
        graph,
        continuations: tuple[AssemblyPath, ...],
        station,
        collision_checker,
    ):
        cache = {}
        checked = 0
        rejected = 0
        reasons: dict[str, int] = {}
        terminal = len(graph.valid) - 1
        for roll_index in np.flatnonzero(graph.valid[terminal]):
            checked += 1
            branches = []
            branch_reasons = []
            for continuation in continuations:
                result, reason = self._evaluate_continuation(
                    continuation,
                    station,
                    float(graph.rolls[roll_index]),
                    graph.joints[terminal, roll_index],
                    collision_checker,
                )
                branches.append(result)
                branch_reasons.append(reason)
            if any(result is None for result in branches):
                graph.valid[terminal, roll_index] = False
                rejected += 1
                for index, (result, reason) in enumerate(zip(branches, branch_reasons, strict=True)):
                    if result is None:
                        key = f"branch_{index}_{reason}"
                        reasons[key] = reasons.get(key, 0) + 1
            else:
                combined = self._combine_continuations(
                    continuations,
                    tuple(branches),
                )
                cache[int(roll_index)] = combined
                graph.penalties[terminal, roll_index] += float(combined["cost"])
        return cache, {
            "continuation_checked_nodes": checked,
            "continuation_rejected_nodes": rejected,
            "continuation_reject_reasons": reasons,
            "final_valid_nodes": int(np.count_nonzero(graph.valid)),
            "final_valid_by_step": np.count_nonzero(graph.valid, axis=1),
        }

    @staticmethod
    def _combine_continuations(paths, results):
        if len(results) == 1:
            return dict(results[0])
        if len(results) != 2:
            raise ValueError("combining more than two terminal continuation branches is unsupported")
        first_path, second_path = paths
        first, second = results
        return {
            "states": np.vstack((first_path.states[:0:-1], second_path.states)),
            "joints": np.vstack((first["joints"][:0:-1], second["joints"])),
            "cost": max(float(first["cost"]), float(second["cost"])),
        }

    def _evaluate_continuation(self, path, station, roll, start, collision_checker):
        config = self.continuation_config
        targets = self._poses(path.states, np.full(len(path.states), roll), station)
        joints, valid = self.arm.solve_ik(targets, branches=(self.ik_branch,))
        joints = joints[:, 0]
        valid = valid[:, 0]
        if not np.all(valid) and not bool(self.best_effort.get("enabled", False)):
            return None, "ik"
        if not np.all(valid):
            q_ref = np.asarray(start)
            for index in range(len(joints)):
                if not valid[index]:
                    joints[index] = self._approximate_ik(targets[index], q_ref)[0]
                joints[index] = self.arm.joint_space.align_to_reference(
                    joints[index][None, :], q_ref[None, :]
                )[0]
                q_ref = joints[index]
        joints = self.arm.joint_space.align_trajectory(joints, np.asarray(start)[None, :])
        if collision_checker is None:
            collision_penalty = np.zeros(len(joints))
        else:
            collides, collision_penalty = collision_checker.check_configs_with_penalty(
                joints,
                float(config.get("collision_soft_margin", 0.0)),
                float(config.get("collision_soft_weight", 0.0)),
            )
            if np.any(collides):
                return None, "collision"
        deltas = self.arm.joint_space.delta(joints[1:], joints[:-1])
        start_delta = self.arm.joint_space.delta(joints[:1], np.asarray(start)[None, :])
        motion = float(np.sum(self._motion_cost(np.vstack((start_delta, deltas)))))
        limit = float(
            np.sum(
                self.arm.joint_space.limit_penalty(
                    joints,
                    float(config.get("limit_margin_rad", 0.0)),
                    float(config.get("limit_weight", 0.0)),
                )
            )
        )
        return {
            "states": path.states,
            "joints": joints,
            "cost": float(config.get("motion_weight", 1.0)) * motion
            + float(np.sum(collision_penalty))
            + limit,
        }, "ok"

    def _poses(self, states: np.ndarray, rolls: np.ndarray, station: np.ndarray) -> np.ndarray:
        axial, slide, p_angle, q_angle, release = np.asarray(states, dtype=float).T
        cos_q, sin_q = np.cos(q_angle), np.sin(q_angle)
        cos_p, sin_p = np.cos(p_angle), np.sin(p_angle)
        x_axis = np.column_stack((cos_p, -sin_q * sin_p, cos_q * sin_p))
        y_axis = np.column_stack((np.zeros(len(states)), cos_q, sin_q))
        z_axis = np.column_stack((-sin_p, -sin_q * cos_p, cos_q * cos_p))
        local_position = np.column_stack(
            (
                axial + self.p_radius * (np.cos(p_angle - self.p_offset) - np.cos(self.p_offset)),
                -self.q_radius * np.sin(q_angle),
                slide
                + self.p_radius * (np.sin(p_angle - self.p_offset) + np.sin(self.p_offset))
                + self.q_radius * (np.cos(q_angle) - 1.0),
            )
        )
        cos_roll, sin_roll = np.cos(rolls), np.sin(rolls)
        local_position += release[:, None] * (
            sin_roll[:, None] * y_axis - cos_roll[:, None] * z_axis
        )
        local_rotation = np.empty((len(states), 3, 3))
        local_rotation[:, :, 0] = x_axis
        local_rotation[:, :, 1] = cos_roll[:, None] * y_axis + sin_roll[:, None] * z_axis
        local_rotation[:, :, 2] = -sin_roll[:, None] * y_axis + cos_roll[:, None] * z_axis
        poses = np.broadcast_to(np.eye(4), (len(states), 4, 4)).copy()
        poses[:, :3, :3] = station[:3, :3] @ local_rotation
        poses[:, :3, 3] = local_position @ station[:3, :3].T + station[:3, 3]
        return poses

    def _motion_cost(self, delta: np.ndarray) -> np.ndarray:
        return self.motion_l1_weight * np.sum(np.abs(delta), axis=-1) + self.motion_l2_weight * np.sum(
            delta * delta, axis=-1
        )

    @staticmethod
    def _failure_message(graph, diagnostics):
        valid_counts = np.count_nonzero(graph.valid, axis=1)
        first_zero = np.flatnonzero(valid_counts == 0)
        first_zero_text = "none" if len(first_zero) == 0 else str(int(first_zero[0]))
        reasons = diagnostics.get("continuation_reject_reasons", {})
        reason_text = ",".join(f"{key}:{value}" for key, value in sorted(reasons.items())) or "none"
        return (
            "type3 planner found no path "
            f"terminal_valid={int(valid_counts[-1])} "
            f"first_zero_step={first_zero_text} "
            f"continuation_reasons={reason_text}"
        )
