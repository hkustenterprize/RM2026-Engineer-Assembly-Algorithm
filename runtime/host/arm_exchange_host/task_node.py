from __future__ import annotations

import threading
import time
import uuid

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from arm_exchange_interfaces.msg import (
    ArmCtrlAdvanceMsg,
    ArmCtrlEnterMsg,
    ArmCtrlQMsg,
    ArmCtrlStartMsg,
    ArmCtrlWithdrawMsg,
    ArmCtrlXYZMsg,
    ArmFeedforwardWrenchMsg,
    ArmHost2MCUMsg,
    ArmMCU2HostMsg,
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
from arm_exchange_core.planning.collision import CollisionChecker, CollisionModel
from arm_exchange_core.planning.type3 import AssemblyState

from .ros_utils import (
    duration_to_seconds,
    host_output_from_trajectory_point,
    joint_positions_from_joint_state,
    transform_from_pose_stamped,
)


HOST_STATE_BY_TASK_STATE = {
    "IDLE": ArmHost2MCUMsg.HOST_IDLE,
    "WAITING_PERCEPTION": ArmHost2MCUMsg.HOST_WAITING_PERCEPTION,
    "PLANNING_APPROACH": ArmHost2MCUMsg.HOST_PLANNING_APPROACH,
    "EXECUTING_APPROACH": ArmHost2MCUMsg.HOST_EXECUTING_APPROACH,
    "READY_TO_EXCHANGE": ArmHost2MCUMsg.HOST_READY_TO_EXCHANGE,
    "PLANNING_TYPE3": ArmHost2MCUMsg.HOST_PLANNING_TYPE3,
    "SLIDE_READY": ArmHost2MCUMsg.HOST_SLIDE_READY,
    "SLIDE_EXECUTING": ArmHost2MCUMsg.HOST_SLIDE_EXECUTING,
    "SLIDE_ADJUST": ArmHost2MCUMsg.HOST_SLIDE_ADJUST,
    "P_READY": ArmHost2MCUMsg.HOST_P_READY,
    "P_EXECUTING": ArmHost2MCUMsg.HOST_P_EXECUTING,
    "P_ADJUST": ArmHost2MCUMsg.HOST_P_ADJUST,
    "Q_MANUAL_ADJUST": ArmHost2MCUMsg.HOST_Q_MANUAL,
    "ERROR": ArmHost2MCUMsg.HOST_ERROR,
}

RECOVERY_HOST_STATE_BY_MODE = {
    RecoveryPlanningRequest.MODE_EXTRACT_FOR_RETRY: (
        ArmHost2MCUMsg.HOST_RECOVERY_EXTRACT_PLANNING,
        ArmHost2MCUMsg.HOST_RECOVERY_EXTRACT_EXECUTING,
    ),
    RecoveryPlanningRequest.MODE_SIDE_RELEASE: (
        ArmHost2MCUMsg.HOST_RECOVERY_SIDE_RELEASE_PLANNING,
        ArmHost2MCUMsg.HOST_RECOVERY_SIDE_RELEASE_EXECUTING,
    ),
    RecoveryPlanningRequest.MODE_RETRACT_TO_APPROACH_START: (
        ArmHost2MCUMsg.HOST_RECOVERY_RETRACT_PLANNING,
        ArmHost2MCUMsg.HOST_RECOVERY_RETRACT_EXECUTING,
    ),
}

FEEDFORWARD_WRENCH_STATES = {
    "SLIDE_EXECUTING",
    "SLIDE_ADJUST",
    "P_EXECUTING",
    "P_ADJUST",
    "Q_MANUAL_ADJUST",
}


class TaskNode(Node):
    def __init__(self) -> None:
        super().__init__("arm_exchange_task")
        system_cfg = load_config()
        self.arm = ArmModel.from_config(system_cfg["arm"])
        self.collision_model = CollisionModel.from_config(system_cfg["collision"])

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._tick_period_s = 0.001
        planning_cfg = system_cfg["planning"]
        exchange_cfg = planning_cfg["exchange"]
        stage_path_cfg = exchange_cfg["stage_path"]
        q_manual_cfg = stage_path_cfg["q_manual"]
        ready_adjust_cfg = stage_path_cfg["ready_adjust"]
        semi_auto_cfg = stage_path_cfg["semi_auto"]
        feedforward_wrench_cfg = exchange_cfg.get("feedforward_wrench", {})
        exchange_trajectory_cfg = planning_cfg["type3"]["exchange_trajectory"]

        self.state = "IDLE"
        self.selected_exchange_level = int(stage_path_cfg["default_level"])
        self.mcu_enabled = False
        self.latest_mcu_state: ArmMCU2HostMsg | None = None
        self.pending_start_level: int | None = None
        self.latest_joint_state: JointState | None = None
        self.latest_station_pose: PoseStamped | None = None
        self.active_station_pose: PoseStamped | None = None
        self.approach_start_joint_state: JointState | None = None
        self.pending_request_id = ""
        self.pending_request_kind = ""
        self.recovery_mode = 0
        self.recovery_exchange_local_pose: PoseStamped | None = None
        self.trajectory: JointTrajectory | None = None
        self.trajectory_times: np.ndarray | None = None
        self.hold_point: JointTrajectoryPoint | None = None
        self.q_manual_point: JointTrajectoryPoint | None = None
        self.advance_active = False
        self.advance_last_wall_time_s: float | None = None
        self.stage_playback_elapsed_s = 0.0
        self.stage_name = ""
        self.confirmed_exchange_stage = "insert"
        self.stage_adjust_extra = 0.0
        self.stage_collision_checker = None
        self.q_manual_angle = 0.0
        self.q_manual_target_angle = 0.0
        self.q_sweep_s: np.ndarray | None = None
        self.q_sweep_positions: np.ndarray | None = None
        self.q_manual_min_angle = np.deg2rad(float(q_manual_cfg["min_angle_deg"]))
        self.q_manual_max_angle = np.deg2rad(float(q_manual_cfg["max_angle_deg"]))
        self.q_manual_rate_limit = np.deg2rad(float(q_manual_cfg["rate_limit_deg_s"]))
        self.q_manual_command_sample_dt = float(q_manual_cfg.get("command_sample_dt", self._tick_period_s))
        self._last_q_manual_solve_wall_time_s = 0.0
        self._last_q_manual_step_wall_time_s = 0.0
        self.ready_adjust_max_delta_m = float(ready_adjust_cfg["max_delta_m"])
        if not np.isfinite(self.ready_adjust_max_delta_m) or self.ready_adjust_max_delta_m <= 0.0:
            raise ValueError("ready_adjust.max_delta_m must be a positive finite value")
        self.ready_adjust_max_rate_m_s = float(ready_adjust_cfg["max_rate_m_s"])
        if not np.isfinite(self.ready_adjust_max_rate_m_s) or self.ready_adjust_max_rate_m_s <= 0.0:
            raise ValueError("ready_adjust.max_rate_m_s must be a positive finite value")
        self.ready_adjust_roll_search_max_rad = np.deg2rad(float(ready_adjust_cfg["roll_search_max_deg"]))
        self.ready_adjust_roll_search_step_rad = np.deg2rad(float(ready_adjust_cfg["roll_search_step_deg"]))
        if not np.isfinite(self.ready_adjust_roll_search_max_rad) or self.ready_adjust_roll_search_max_rad < 0.0:
            raise ValueError("ready_adjust.roll_search_max_deg must be a non-negative finite value")
        if self.ready_adjust_roll_search_max_rad > 0.0 and (
            not np.isfinite(self.ready_adjust_roll_search_step_rad)
            or self.ready_adjust_roll_search_step_rad <= 0.0
        ):
            raise ValueError("ready_adjust.roll_search_step_deg must be positive when roll search is enabled")
        self.ready_adjust_last_wall_time_s: float | None = None
        configured_ready_branches = ready_adjust_cfg["ik_branches"]
        if configured_ready_branches is None:
            configured_ready_branches = exchange_cfg["ik_branches"]
        self.ready_adjust_ik_branches = self._normalize_ik_branches(configured_ready_branches)
        configured_adjust_branches = semi_auto_cfg["ik_branches"]
        if configured_adjust_branches is None:
            configured_adjust_branches = exchange_cfg["ik_branches"]
        self.adjust_ik_branches = self._normalize_ik_branches(configured_adjust_branches)
        self.slide_forward_rate_m_s = float(semi_auto_cfg["slide_forward_rate_m_s"])
        self.slide_forward_max_extra_m = float(semi_auto_cfg["slide_forward_max_extra_m"])
        self.p_forward_rate_rad_s = np.deg2rad(float(semi_auto_cfg["p_forward_rate_deg_s"]))
        self.p_forward_max_extra_rad = np.deg2rad(float(semi_auto_cfg["p_forward_max_extra_deg"]))
        self.adjust_roll_search_max_rad = np.deg2rad(float(semi_auto_cfg["adjust_roll_search_max_deg"]))
        self.adjust_roll_search_step_rad = np.deg2rad(float(semi_auto_cfg["adjust_roll_search_step_deg"]))
        for name, value in (
            ("semi_auto.slide_forward_rate_m_s", self.slide_forward_rate_m_s),
            ("semi_auto.slide_forward_max_extra_m", self.slide_forward_max_extra_m),
            ("semi_auto.p_forward_rate_deg_s", self.p_forward_rate_rad_s),
            ("semi_auto.p_forward_max_extra_deg", self.p_forward_max_extra_rad),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"stage_path.{name} must be positive")
        if not np.isfinite(self.adjust_roll_search_max_rad) or self.adjust_roll_search_max_rad < 0.0:
            raise ValueError("stage_path.semi_auto.adjust_roll_search_max_deg must be non-negative")
        if self.adjust_roll_search_max_rad > 0.0 and (
            not np.isfinite(self.adjust_roll_search_step_rad)
            or self.adjust_roll_search_step_rad <= 0.0
        ):
            raise ValueError(
                "stage_path.semi_auto.adjust_roll_search_step_deg must be "
                "positive when roll search is enabled"
            )
        self.stage_slide_z_mag = float(exchange_trajectory_cfg["slide_z_mag"])
        self.stage_r_p = float(exchange_trajectory_cfg["r_p"])
        self.stage_r_q = float(exchange_trajectory_cfg["r_q"])
        self.stage_phi = float(exchange_trajectory_cfg["phi"])
        for name, value in (
            ("planning.type3.exchange_trajectory.slide_z_mag", self.stage_slide_z_mag),
            ("planning.type3.exchange_trajectory.r_p", self.stage_r_p),
            ("planning.type3.exchange_trajectory.r_q", self.stage_r_q),
            ("planning.type3.exchange_trajectory.phi", self.stage_phi),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        self.feedforward_wrench_enabled = bool(feedforward_wrench_cfg.get("enabled", False))
        self.feedforward_force_magnitude_n = float(feedforward_wrench_cfg.get("force_magnitude_n", 3.0))
        if (
            not np.isfinite(self.feedforward_force_magnitude_n)
            or self.feedforward_force_magnitude_n < 0.0
        ):
            raise ValueError("planning.exchange.feedforward_wrench.force_magnitude_n must be non-negative")
        self.feedforward_pressure_direction_local = np.asarray(
            feedforward_wrench_cfg.get("pressure_direction_local", [-1.0, 0.0, 0.0]),
            dtype=float,
        )
        if (
            self.feedforward_pressure_direction_local.shape != (3,)
            or np.any(~np.isfinite(self.feedforward_pressure_direction_local))
            or float(np.linalg.norm(self.feedforward_pressure_direction_local)) <= 1e-12
        ):
            raise ValueError(
                "planning.exchange.feedforward_wrench.pressure_direction_local "
                "must be a non-zero 3-vector"
            )
        self.feedforward_pressure_direction_local /= float(
            np.linalg.norm(self.feedforward_pressure_direction_local)
        )
        self.feedforward_contact_point_tcp_m = np.asarray(
            feedforward_wrench_cfg.get("contact_point_tcp_m", [0.0, 0.0, 0.0]),
            dtype=float,
        )
        if self.feedforward_contact_point_tcp_m.shape != (3,) or np.any(
            ~np.isfinite(self.feedforward_contact_point_tcp_m)
        ):
            raise ValueError(
                "planning.exchange.feedforward_wrench.contact_point_tcp_m "
                "must be a finite 3-vector"
            )
        self.feedforward_min_projected_force_ratio = float(
            feedforward_wrench_cfg.get("min_projected_force_ratio", 0.25)
        )
        if (
            not np.isfinite(self.feedforward_min_projected_force_ratio)
            or self.feedforward_min_projected_force_ratio < 0.0
        ):
            raise ValueError(
                "planning.exchange.feedforward_wrench.min_projected_force_ratio "
                "must be non-negative"
            )
        self.feedforward_max_force_n = float(feedforward_wrench_cfg.get("max_force_n", 8.0))
        self.feedforward_max_torque_nm = float(feedforward_wrench_cfg.get("max_torque_nm", 1.0))
        for name, value in (
            ("planning.exchange.feedforward_wrench.max_force_n", self.feedforward_max_force_n),
            ("planning.exchange.feedforward_wrench.max_torque_nm", self.feedforward_max_torque_nm),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive")
        self.playback_start_s = 0.0
        self.execution_done_state = "IDLE"
        self._last_logged_task_status = ""

        self.approach_request_pub = self.create_publisher(
            ApproachPlanningRequest,
            "/host/planning/approach_request",
            10,
        )
        self.type3_request_pub = self.create_publisher(
            Type3PlanningRequest,
            "/host/planning/type3_request",
            10,
        )
        self.recovery_request_pub = self.create_publisher(
            RecoveryPlanningRequest,
            "/host/planning/recovery_request",
            10,
        )
        self.command_pub = self.create_publisher(ArmHost2MCUMsg, "/host/arm/host_output", 10)
        self.feedforward_wrench_pub = self.create_publisher(
            ArmFeedforwardWrenchMsg,
            "/host/arm/feedforward_wrench",
            10,
        )
        self.state_pub = self.create_publisher(String, "/host/task_state", 10)

        latest_control_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(ArmMCU2HostMsg, "/mcu/arm/state", self._on_mcu_state, 10)
        self.create_subscription(ArmCtrlStartMsg, "/mcu/arm/ctrl_start", self._on_ctrl_start, 10)
        self.create_subscription(ArmCtrlEnterMsg, "/mcu/arm/ctrl_enter", self._on_ctrl_enter, 10)
        self.create_subscription(ArmCtrlAdvanceMsg, "/mcu/arm/ctrl_advance", self._on_ctrl_advance, latest_control_qos)
        self.create_subscription(ArmCtrlQMsg, "/mcu/arm/ctrl_q", self._on_ctrl_q, latest_control_qos)
        self.create_subscription(ArmCtrlWithdrawMsg, "/mcu/arm/ctrl_withdraw", self._on_ctrl_withdraw, 10)
        self.create_subscription(ArmCtrlXYZMsg, "/mcu/arm/ctrl_xyz", self._on_ctrl_xyz, latest_control_qos)
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)
        self.create_subscription(PoseStamped, "/host/perception/exchange_station_pose", self._on_station_pose, 10)
        self.create_subscription(
            ApproachPlanningResult,
            "/host/planning/approach_result",
            self._on_approach_result,
            10,
        )
        self.create_subscription(Type3PlanningResult, "/host/planning/type3_result", self._on_type3_result, 10)
        self.create_subscription(
            RecoveryPlanningResult,
            "/host/planning/recovery_result",
            self._on_recovery_result,
            10,
        )

        self.create_timer(0.1, self._publish_state)
        self._tick_thread = threading.Thread(target=self._tick_loop, name="task_node_tick", daemon=False)
        self._tick_thread.start()

    def _on_mcu_state(self, msg: ArmMCU2HostMsg) -> None:
        with self._lock:
            self.latest_mcu_state = msg
            if bool(msg.enabled) == self.mcu_enabled:
                return
            self.mcu_enabled = bool(msg.enabled)
            if not self.mcu_enabled:
                self.pending_start_level = None
                self._reset_task("mcu disabled", target_state="IDLE")
            elif self.pending_start_level is not None:
                level = self.pending_start_level
                self.pending_start_level = None
                self._handle_start(level)
            self._log_task_status(force=True)

    def _on_ctrl_start(self, msg: ArmCtrlStartMsg) -> None:
        with self._lock:
            level = int(msg.level)
            if not self._accept_mcu_control():
                self.pending_start_level = level
                return
            self._handle_start(level)

    def _handle_start(self, level: int) -> None:
        if self.state != "IDLE":
            return
        self.selected_exchange_level = int(level)
        if self.selected_exchange_level not in (1, 2, 3):
            self.get_logger().warn(f"ignore invalid exchange level: {self.selected_exchange_level}")
            return
        self.hold_point = None
        if self.latest_station_pose is None or self.latest_joint_state is None:
            self.state = "WAITING_PERCEPTION"
            self._log_task_status(force=True)
        else:
            self._request_approach_plan()

    def _on_ctrl_enter(self, _msg: ArmCtrlEnterMsg) -> None:
        with self._lock:
            if not self._accept_mcu_control():
                return
            if self.state == "READY_TO_EXCHANGE":
                if self.latest_joint_state is None or self.active_station_pose is None:
                    self._release_to_operator("READY enter missing joint state or active exchange pose", error=True)
                    return
                self._reconstruct_active_station_pose(AssemblyState())
                self.confirmed_exchange_stage = "insert"
                if self.selected_exchange_level == 1:
                    self.get_logger().info("level 1 insert confirmed; use withdraw recovery to leave exchange")
                else:
                    self._request_type3_plan("slide")
            elif self.state == "SLIDE_ADJUST":
                self._reconstruct_active_station_pose(
                    AssemblyState(slide_m=self.stage_slide_z_mag)
                )
                self.confirmed_exchange_stage = "slide"
                if self.selected_exchange_level == 2:
                    self.get_logger().info("level 2 slide confirmed; use withdraw recovery to leave exchange")
                else:
                    self._request_type3_plan("p")
            elif self.state == "P_ADJUST":
                self._reconstruct_active_station_pose(
                    AssemblyState(
                        slide_m=self.stage_slide_z_mag,
                        p_angle_rad=0.5 * np.pi,
                    )
                )
                self._rebase_q_sweep_to_current_hold()
                self.confirmed_exchange_stage = "p"
                self.state = "Q_MANUAL_ADJUST"
                self.q_manual_point = self.hold_point
                self.q_manual_angle = 0.0
                self.q_manual_target_angle = 0.0
                self._log_task_status(force=True)
            elif self.state == "Q_MANUAL_ADJUST":
                self.get_logger().info("Q manual confirmed; use withdraw recovery to leave exchange")

    def _on_ctrl_advance(self, msg: ArmCtrlAdvanceMsg) -> None:
        with self._lock:
            next_active = bool(msg.active) if self._accept_mcu_control() else False
            if next_active != self.advance_active:
                self.advance_last_wall_time_s = None
            self.advance_active = next_active

    def _on_ctrl_q(self, msg: ArmCtrlQMsg) -> None:
        with self._lock:
            if not self._accept_mcu_control() or self.state != "Q_MANUAL_ADJUST":
                return
            if msg.mode == ArmCtrlQMsg.MODE_TARGET:
                target_angle = np.deg2rad(float(msg.q_deg10) * 0.1)
            elif msg.mode == ArmCtrlQMsg.MODE_DELTA:
                target_angle = self.q_manual_target_angle + np.deg2rad(float(msg.q_deg10) * 0.1)
            else:
                self.get_logger().warn(f"ignore invalid Q control mode: {int(msg.mode)}")
                return
            self._update_q_manual_target(target_angle)

    def _on_ctrl_xyz(self, msg: ArmCtrlXYZMsg) -> None:
        with self._lock:
            if not self._accept_mcu_control() or self.state != "READY_TO_EXCHANGE":
                return
            self._apply_ready_xyz_delta(np.asarray([msg.dx_m, msg.dy_m, msg.dz_m], dtype=float))

    def _on_ctrl_withdraw(self, msg: ArmCtrlWithdrawMsg) -> None:
        with self._lock:
            if not self._accept_mcu_control():
                return
            mode = int(msg.mode) if int(msg.mode) != 0 else ArmCtrlWithdrawMsg.MODE_EXTRACT_FOR_RETRY
            mode_name = self._withdraw_mode_name(mode)
            self.get_logger().warn(f"withdraw command state={self.state} mode={mode_name}({mode})")
            if self.state in ("RECOVERY_PLANNING", "RECOVERY_EXECUTING"):
                self.get_logger().warn("ignore withdraw while recovery is active", throttle_duration_sec=1.0)
                return
            if not self._can_recover_for_retry():
                self.get_logger().warn(f"ignore withdraw outside recoverable states: {self.state}")
                return
            self.recovery_exchange_local_pose = self._current_exchange_local_pose()
            if mode == ArmCtrlWithdrawMsg.MODE_EXTRACT_FOR_RETRY:
                self.get_logger().warn(f"request extract-for-retry recovery from state={self.state}")
                self._request_recovery_plan(RecoveryPlanningRequest.MODE_EXTRACT_FOR_RETRY)
            elif mode == ArmCtrlWithdrawMsg.MODE_SIDE_RELEASE:
                self.get_logger().warn(f"request side-release recovery from state={self.state}")
                self._request_recovery_plan(RecoveryPlanningRequest.MODE_SIDE_RELEASE)
            else:
                self.get_logger().warn(f"ignore invalid withdraw mode: {mode}")

    def _accept_mcu_control(self) -> bool:
        return bool(self.mcu_enabled)

    def _withdraw_mode_name(self, mode: int) -> str:
        if mode == ArmCtrlWithdrawMsg.MODE_EXTRACT_FOR_RETRY:
            return "EXTRACT_FOR_RETRY"
        if mode == ArmCtrlWithdrawMsg.MODE_SIDE_RELEASE:
            return "SIDE_RELEASE"
        return "UNKNOWN"

    def _on_joint_state(self, msg: JointState) -> None:
        with self._lock:
            self.latest_joint_state = msg
            self._request_waiting_approach_if_ready()

    def _on_station_pose(self, msg: PoseStamped) -> None:
        if msg.header.frame_id != "arm_base":
            raise ValueError(f"perception station pose must be in arm_base, got {msg.header.frame_id!r}")
        with self._lock:
            self.latest_station_pose = msg
            self._request_waiting_approach_if_ready()

    def _request_waiting_approach_if_ready(self) -> None:
        if (
            self.state == "WAITING_PERCEPTION"
            and self.latest_station_pose is not None
            and self.latest_joint_state is not None
        ):
            self._request_approach_plan()

    def _on_approach_result(self, msg: ApproachPlanningResult) -> None:
        with self._lock:
            if msg.request_id != self.pending_request_id or self.pending_request_kind != "approach":
                return
            if not msg.success:
                self.get_logger().warn(f"approach planning failed: {msg.message}")
                self._release_to_operator("approach planning failed", error=True)
                return
            self._start_execution(msg.trajectory, state="EXECUTING_APPROACH", done_state="READY_TO_EXCHANGE")

    def _on_type3_result(self, msg: Type3PlanningResult) -> None:
        with self._lock:
            if msg.request_id != self.pending_request_id or self.pending_request_kind != "type3":
                return
            if not msg.success:
                self.get_logger().warn(f"type3 planning failed: {msg.message}")
                self._release_to_operator("type3 planning failed", error=True)
                return
            path_name = str(msg.path_name).strip()
            if path_name == "slide":
                self._start_stage_execution(msg.trajectory, stage_name="slide")
                return
            if path_name == "p":
                self._clear_q_manual_context()
                if not self._cache_q_sweep(msg):
                    self._release_to_operator("type3 result missing Q sweep cache", error=True)
                    return
                self._start_stage_execution(msg.trajectory, stage_name="p")
                return

            self._release_to_operator(f"unsupported Type III path: {path_name}", error=True)

    def _on_recovery_result(self, msg: RecoveryPlanningResult) -> None:
        with self._lock:
            if msg.request_id != self.pending_request_id or self.pending_request_kind != "recovery":
                return
            if not msg.success:
                if (
                    msg.degraded
                    and int(msg.mode) == RecoveryPlanningRequest.MODE_EXTRACT_FOR_RETRY
                    and self.state == "RECOVERY_PLANNING"
                ):
                    self.get_logger().warn(
                        f"extract-for-retry degraded, fallback to side-release: {msg.message}"
                    )
                    self._request_recovery_plan(RecoveryPlanningRequest.MODE_SIDE_RELEASE)
                    return
                self.get_logger().warn(f"recovery planning failed: {msg.message}")
                self._release_to_operator("recovery planning failed", error=True)
                return
            self.recovery_mode = int(msg.mode)
            self._start_execution(msg.trajectory, state="RECOVERY_EXECUTING", done_state="RECOVERY_STAGE_DONE")

    def _tick_loop(self) -> None:
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            with self._lock:
                self._tick()
            next_tick += self._tick_period_s
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0.0:
                self._stop_event.wait(sleep_s)
            else:
                next_tick = time.monotonic()

    def _tick(self) -> None:
        if (
            self.state in ("EXECUTING_APPROACH", "RECOVERY_EXECUTING")
            and self.trajectory is not None
            and self.trajectory_times is not None
        ):
            elapsed = self.get_clock().now().nanoseconds * 1e-9 - self.playback_start_s
            point = self._sample_trajectory_point(elapsed)
            self._publish_host_output(point)
            if elapsed >= float(self.trajectory_times[-1]):
                self._finish_execution()
        elif self.state in ("SLIDE_READY", "SLIDE_EXECUTING", "P_READY", "P_EXECUTING"):
            self._tick_stage_playback()
        elif self.state in ("SLIDE_ADJUST", "P_ADJUST"):
            self._tick_stage_adjust()
        elif self.state == "READY_TO_EXCHANGE" and self.hold_point is not None:
            self._publish_host_output(self.hold_point)
        elif self.state == "Q_MANUAL_ADJUST" and self.q_manual_point is not None:
            self._step_q_manual_target()
            self._publish_host_output(self.q_manual_point)
        else:
            self._publish_idle_host_output()
        self._publish_feedforward_wrench()

    def _request_approach_plan(self) -> None:
        if self.state not in ["WAITING_PERCEPTION", "IDLE"]:
            self.get_logger().warn(f"cannot request approach plan in state {self.state}")
            return
        self.active_station_pose = self.latest_station_pose
        self.approach_start_joint_state = self.latest_joint_state
        station_position = self._pose_position(self.active_station_pose)
        station_rotation = self._pose_rotation(self.active_station_pose)
        station_quat = quaternions_from_rotations(station_rotation[None, :, :])[0]
        station_stamp = self.active_station_pose.header.stamp
        self.get_logger().info(
            "approach locked perception station pose "
            f"stamp={station_stamp.sec}.{station_stamp.nanosec:09d} "
            f"position={np.round(station_position, 6).tolist()} "
            f"quat_wxyz={np.round(station_quat, 6).tolist()}"
        )
        request = ApproachPlanningRequest()
        request.header.stamp = self.get_clock().now().to_msg()
        request.header.frame_id = "arm_base"
        request.request_id = uuid.uuid4().hex
        request.station_pose_arm_base = self.active_station_pose
        request.initial_joint_state = self.latest_joint_state
        self.pending_request_id = request.request_id
        self.pending_request_kind = "approach"
        self.state = "PLANNING_APPROACH"
        self._log_task_status(force=True)
        self.approach_request_pub.publish(request)

    def _request_type3_plan(self, path_name: str) -> None:
        if self.state not in ["READY_TO_EXCHANGE", "SLIDE_ADJUST"]:
            self.get_logger().warn(f"cannot request type3 plan in state {self.state}")
            return

        request = Type3PlanningRequest()
        request.header.stamp = self.get_clock().now().to_msg()
        request.header.frame_id = "arm_base"
        request.request_id = uuid.uuid4().hex
        request.station_pose_arm_base = self.active_station_pose
        request.initial_joint_state = self.latest_joint_state
        request.path_name = str(path_name)
        self.pending_request_id = request.request_id
        self.pending_request_kind = "type3"
        self.stage_name = str(path_name)
        self.state = "PLANNING_TYPE3"
        self._log_task_status(force=True)
        self.type3_request_pub.publish(request)

    def _request_recovery_plan(self, mode: int) -> None:
        if (
            self.active_station_pose is None
            or self.latest_joint_state is None
            or self.approach_start_joint_state is None
        ):
            self._release_to_operator(
                "recovery requires active station pose, current joint state, and approach start",
                error=True,
            )
            return
        if self.recovery_exchange_local_pose is None:
            self.recovery_exchange_local_pose = self._current_exchange_local_pose()
        self._hold_latest_joint_state()
        self.trajectory = None
        self.trajectory_times = None
        request = RecoveryPlanningRequest()
        request.header.stamp = self.get_clock().now().to_msg()
        request.header.frame_id = "arm_base"
        request.request_id = uuid.uuid4().hex
        request.mode = int(mode)
        request.station_pose_arm_base = self.active_station_pose
        # Wire name kept for compatibility; semantic is exchange_local_frame in arm_base.
        request.local_pose_arm_base = self.recovery_exchange_local_pose
        request.initial_joint_state = self.latest_joint_state
        request.goal_joint_state = self.approach_start_joint_state
        self.pending_request_id = request.request_id
        self.pending_request_kind = "recovery"
        self.recovery_mode = int(mode)
        self.state = "RECOVERY_PLANNING"
        self._log_task_status(force=True)
        self.recovery_request_pub.publish(request)

    def _current_exchange_local_pose(self) -> PoseStamped:
        position, rotation = self._current_exchange_local_frame()
        return self._pose_from_position_rotation(position, rotation)

    def _pose_from_position_rotation(self, position: np.ndarray, rotation: np.ndarray) -> PoseStamped:
        quat = quaternions_from_rotations(np.asarray(rotation, dtype=float)[None, :, :])[0]
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "arm_base"
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        msg.pose.orientation.w = float(quat[0])
        msg.pose.orientation.x = float(quat[1])
        msg.pose.orientation.y = float(quat[2])
        msg.pose.orientation.z = float(quat[3])
        return msg

    def _current_exchange_local_frame(self) -> tuple[np.ndarray, np.ndarray]:
        if self.state in ("SLIDE_READY", "SLIDE_EXECUTING"):
            alpha = self._stage_playback_alpha()
            return self._local_exchange_frame("slide", extra=(alpha - 1.0) * self.stage_slide_z_mag)
        if self.state == "SLIDE_ADJUST":
            return self._local_exchange_frame("slide", extra=float(self.stage_adjust_extra))
        if self.state in ("P_READY", "P_EXECUTING"):
            alpha = self._stage_playback_alpha()
            return self._local_exchange_frame("p", extra=(alpha - 1.0) * 0.5 * np.pi)
        if self.state == "P_ADJUST":
            return self._local_exchange_frame("p", extra=float(self.stage_adjust_extra))
        if self.state == "Q_MANUAL_ADJUST":
            return self._local_exchange_frame("q", extra=float(self.q_manual_angle))
        if self.state == "PLANNING_TYPE3":
            return self._local_exchange_frame(self.confirmed_exchange_stage, extra=0.0)
        if self.state in ("RECOVERY_PLANNING", "RECOVERY_EXECUTING"):
            if self.recovery_exchange_local_pose is not None:
                return (
                    self._pose_position(self.recovery_exchange_local_pose),
                    self._pose_rotation(self.recovery_exchange_local_pose),
                )
        return self._local_exchange_frame("insert", extra=0.0)

    def _feedforward_stage_extra(self) -> tuple[str, float] | None:
        if self.state in ("SLIDE_EXECUTING", "SLIDE_ADJUST"):
            if self.state == "SLIDE_EXECUTING":
                alpha = self._stage_playback_alpha()
                return "slide", (alpha - 1.0) * self.stage_slide_z_mag
            return "slide", float(self.stage_adjust_extra)
        if self.state in ("P_EXECUTING", "P_ADJUST"):
            if self.state == "P_EXECUTING":
                alpha = self._stage_playback_alpha()
                return "p", (alpha - 1.0) * 0.5 * np.pi
            return "p", float(self.stage_adjust_extra)
        if self.state == "Q_MANUAL_ADJUST":
            return "q", float(self.q_manual_angle)
        return None

    def _stage_playback_alpha(self) -> float:
        if self.trajectory_times is None or self.trajectory_times.size == 0:
            return 0.0
        duration_s = max(float(self.trajectory_times[-1]), 1e-12)
        return float(np.clip(self.stage_playback_elapsed_s / duration_s, 0.0, 1.0))

    def _publish_feedforward_wrench(self) -> None:
        msg = self._zero_feedforward_wrench_msg()
        if not self.feedforward_wrench_enabled or self.state not in FEEDFORWARD_WRENCH_STATES:
            self.feedforward_wrench_pub.publish(msg)
            return
        if self.active_station_pose is None or self.latest_joint_state is None:
            self.feedforward_wrench_pub.publish(msg)
            return
        if self.feedforward_force_magnitude_n <= 0.0:
            self.feedforward_wrench_pub.publish(msg)
            return
        stage_extra = self._feedforward_stage_extra()
        if stage_extra is None:
            self.feedforward_wrench_pub.publish(msg)
            return

        stage, extra = stage_extra
        current_joints = joint_positions_from_joint_state(self.latest_joint_state)
        _, tcp_rotation = self._tcp_pose_from_joints(current_joints)
        _, local_rotation = self._local_exchange_frame(stage, extra=extra)

        raw_force_arm = (
            local_rotation
            @ self.feedforward_pressure_direction_local
            * self.feedforward_force_magnitude_n
        )
        tangent = self._feedforward_stage_tangent(stage, extra)
        safe_force_arm, ratio = self._project_force_orthogonal_to_tangent(raw_force_arm, tangent)
        safe_force_arm = self._clamp_vector_norm(safe_force_arm, self.feedforward_max_force_n)
        if ratio < self.feedforward_min_projected_force_ratio:
            self.get_logger().warn(
                "feedforward wrench projected force too small "
                f"stage={stage} ratio={ratio:.3f} min={self.feedforward_min_projected_force_ratio:.3f}",
                throttle_duration_sec=1.0,
            )
            self.feedforward_wrench_pub.publish(msg)
            return

        force_tcp = tcp_rotation.T @ safe_force_arm
        torque_tcp = np.cross(self.feedforward_contact_point_tcp_m, force_tcp)
        torque_tcp = self._clamp_vector_norm(torque_tcp, self.feedforward_max_torque_nm)

        msg.enabled = True
        msg.host_state = self._host_state_code()
        msg.projected_force_ratio = float(ratio)
        msg.force_tcp_n = [float(value) for value in force_tcp]
        msg.torque_tcp_nm = [float(value) for value in torque_tcp]
        self.feedforward_wrench_pub.publish(msg)

    def _zero_feedforward_wrench_msg(self) -> ArmFeedforwardWrenchMsg:
        msg = ArmFeedforwardWrenchMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "tcp"
        msg.enabled = False
        msg.host_state = self._host_state_code()
        msg.projected_force_ratio = 0.0
        msg.force_tcp_n = [0.0, 0.0, 0.0]
        msg.torque_tcp_nm = [0.0, 0.0, 0.0]
        return msg

    def _feedforward_stage_tangent(self, stage: str, extra: float) -> np.ndarray:
        eps = 1e-5
        if stage == "p" or stage == "q":
            eps = 1e-5
        p0 = self._feedforward_contact_position(stage, extra)
        p1 = self._feedforward_contact_position(stage, extra + eps)
        return (p1 - p0) / eps

    def _feedforward_contact_position(self, stage: str, extra: float) -> np.ndarray:
        local_position, local_rotation = self._local_exchange_frame(stage, extra=float(extra))
        tcp_position = local_position
        return tcp_position + local_rotation @ self.feedforward_contact_point_tcp_m

    def _project_force_orthogonal_to_tangent(
        self,
        force: np.ndarray,
        tangent: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        tangent_norm = float(np.linalg.norm(tangent))
        force_norm = float(np.linalg.norm(force))
        if tangent_norm <= 1e-12 or force_norm <= 1e-12:
            return np.asarray(force, dtype=float).copy(), 0.0 if force_norm <= 1e-12 else 1.0
        unit = np.asarray(tangent, dtype=float) / tangent_norm
        projected = np.asarray(force, dtype=float) - unit * float(np.dot(force, unit))
        return projected, float(np.linalg.norm(projected) / force_norm)

    def _clamp_vector_norm(self, vector: np.ndarray, max_norm: float) -> np.ndarray:
        value = np.asarray(vector, dtype=float)
        norm = float(np.linalg.norm(value))
        if norm <= max_norm or norm <= 1e-12:
            return value
        return value * (max_norm / norm)

    def _hold_latest_joint_state(self) -> None:
        if self.latest_joint_state is None:
            raise RuntimeError("hold latest joint state requires joint state")
        self.hold_point = self._zero_velocity_point(joint_positions_from_joint_state(self.latest_joint_state))

    def _current_joint_reference(self) -> tuple[np.ndarray, str] | None:
        if self.latest_joint_state is not None:
            return joint_positions_from_joint_state(self.latest_joint_state), "latest_joint_state"
        if self.hold_point is not None:
            return np.asarray(self.hold_point.positions, dtype=float), "hold_point"
        return None

    def _can_recover_for_retry(self) -> bool:
        return self.state in (
            "READY_TO_EXCHANGE",
            "PLANNING_TYPE3",
            "SLIDE_READY",
            "SLIDE_EXECUTING",
            "SLIDE_ADJUST",
            "P_READY",
            "P_EXECUTING",
            "P_ADJUST",
            "Q_MANUAL_ADJUST",
        )

    def _apply_ready_xyz_delta(self, delta_exchange: np.ndarray) -> None:
        if self.latest_joint_state is None or self.hold_point is None or self.active_station_pose is None:
            self._release_to_operator("READY xyz missing joint state, hold point, or active exchange pose", error=True)
            return
        if delta_exchange.shape != (3,) or np.any(~np.isfinite(delta_exchange)):
            self.get_logger().warn(f"ignore invalid READY xyz delta: {delta_exchange}")
            return
        delta_norm = float(np.linalg.norm(delta_exchange))
        if delta_norm <= 0.0:
            return
        if delta_norm > self.ready_adjust_max_delta_m:
            self.get_logger().warn(
                f"ignore READY xyz delta norm={delta_norm:.6f} > max={self.ready_adjust_max_delta_m:.6f}"
            )
            return
        delta_exchange = self._clamp_ready_xyz_rate(delta_exchange)
        if not np.any(delta_exchange):
            return

        command_joints = np.asarray(self.hold_point.positions, dtype=float)
        position, tcp_rotation = self._tcp_pose_from_joints(command_joints)
        exchange_rotation = self._pose_rotation(self.active_station_pose)
        target_position = position + exchange_rotation @ delta_exchange
        result = self._solve_ready_xyz_ik(target_position, tcp_rotation, command_joints)
        if result is None:
            self.get_logger().warn(
                "READY xyz IK has no valid solution "
                f"delta={np.round(delta_exchange, 6).tolist()} "
                f"roll_search_max_deg={np.rad2deg(self.ready_adjust_roll_search_max_rad):.3f}"
            )
            return
        best, roll_delta, cost = result

        self.hold_point = self._zero_velocity_point(best)
        self.get_logger().debug(
            "READY xyz adjusted "
            f"delta_exchange={np.round(delta_exchange, 6).tolist()} "
            f"roll_delta_deg={np.rad2deg(roll_delta):.3f} "
            f"cost={cost:.6f} "
            f"target={np.round(best, 6).tolist()}"
        )

    def _solve_ready_xyz_ik(
        self,
        target_position: np.ndarray,
        tcp_rotation: np.ndarray,
        command_joints: np.ndarray,
    ) -> tuple[np.ndarray, float, float] | None:
        return self._solve_roll_search_ik(
            target_position,
            tcp_rotation,
            command_joints,
            roll_shells=self._roll_search_shells(
                self.ready_adjust_roll_search_max_rad,
                self.ready_adjust_roll_search_step_rad,
            ),
            ik_branches=self.ready_adjust_ik_branches,
            collision_checker=None,
        )

    def _roll_search_shells(self, max_roll: float, step: float):
        yield (0.0,)
        if max_roll <= 0.0:
            return

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

    def _clamp_ready_xyz_rate(self, delta_exchange: np.ndarray) -> np.ndarray:
        now_s = time.monotonic()
        last_s = self.ready_adjust_last_wall_time_s
        self.ready_adjust_last_wall_time_s = now_s
        if last_s is None:
            return delta_exchange

        dt_s = max(0.0, now_s - last_s)
        max_delta_m = self.ready_adjust_max_rate_m_s * dt_s
        delta_norm = float(np.linalg.norm(delta_exchange))
        if delta_norm <= max_delta_m:
            return delta_exchange
        if max_delta_m <= 0.0:
            return np.zeros(3, dtype=float)

        clamped = delta_exchange * (max_delta_m / delta_norm)
        self.get_logger().debug(
            "READY xyz rate clamped "
            f"dt_s={dt_s:.6f} "
            f"norm={delta_norm:.6f} "
            f"max_norm={max_delta_m:.6f} "
            f"delta={np.round(clamped, 6).tolist()}"
        )
        return clamped

    def _reconstruct_active_station_pose(self, assembly_state: AssemblyState) -> None:
        if self.active_station_pose is None or self.latest_joint_state is None:
            raise RuntimeError("READY exchange reconstruction requires station pose and joint state")
        current_joints = joint_positions_from_joint_state(self.latest_joint_state)
        tcp_position, tcp_rotation = self._tcp_pose_from_joints(current_joints)
        reference_rotation = self._pose_rotation(self.active_station_pose)
        stage_rotation = self._assembly_state_rotation(assembly_state)
        stage_translation = self._assembly_state_translation(assembly_state)

        exchange_rotation = self._align_reference_axis(
            reference_rotation,
            stage_rotation[:, 0],
            tcp_rotation[:, 0],
        )
        exchange_position = tcp_position - exchange_rotation @ stage_translation

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "arm_base"
        msg.pose.position.x = float(exchange_position[0])
        msg.pose.position.y = float(exchange_position[1])
        msg.pose.position.z = float(exchange_position[2])
        quat = quaternions_from_rotations(exchange_rotation[None, :, :])[0]
        msg.pose.orientation.w = float(quat[0])
        msg.pose.orientation.x = float(quat[1])
        msg.pose.orientation.y = float(quat[2])
        msg.pose.orientation.z = float(quat[3])
        self.active_station_pose = msg
        self.get_logger().info(
            "reconstructed exchange pose "
            f"state={assembly_state} "
            f"position={np.round(exchange_position, 6).tolist()} "
            f"quat_wxyz={np.round(quat, 6).tolist()}"
        )

    def _assembly_state_translation(self, state: AssemblyState) -> np.ndarray:
        p = float(state.p_angle_rad)
        q = float(state.q_angle_rad)
        position = self._p_curve(p)
        position += np.asarray(
            [
                state.axial_offset_m,
                -self.stage_r_q * np.sin(q),
                state.slide_m + self.stage_r_q * (np.cos(q) - 1.0),
            ]
        )
        return position

    @staticmethod
    def _assembly_state_rotation(state: AssemblyState) -> np.ndarray:
        p = float(state.p_angle_rad)
        q = float(state.q_angle_rad)
        cp, sp = np.cos(p), np.sin(p)
        cq, sq = np.cos(q), np.sin(q)
        return np.asarray(
            [
                [cp, 0.0, -sp],
                [-sq * sp, cq, -sq * cp],
                [cq * sp, sq, cq * cp],
            ],
            dtype=float,
        )

    def _align_reference_axis(
        self,
        reference_rotation: np.ndarray,
        source_axis: np.ndarray,
        target_axis: np.ndarray,
    ) -> np.ndarray:
        source = np.asarray(source_axis, dtype=float)
        target = np.asarray(target_axis, dtype=float)
        source /= max(float(np.linalg.norm(source)), 1e-12)
        target /= max(float(np.linalg.norm(target)), 1e-12)
        source_world = np.asarray(reference_rotation, dtype=float) @ source
        delta = self._rotation_between_vectors(source_world, target, reference_rotation[:, 1])
        return delta @ reference_rotation

    def _rotation_between_vectors(
        self,
        source: np.ndarray,
        target: np.ndarray,
        fallback_axis: np.ndarray,
    ) -> np.ndarray:
        source = np.asarray(source, dtype=float)
        target = np.asarray(target, dtype=float)
        source /= max(float(np.linalg.norm(source)), 1e-12)
        target /= max(float(np.linalg.norm(target)), 1e-12)
        dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
        if dot > 1.0 - 1e-10:
            return np.eye(3)
        if dot < -1.0 + 1e-10:
            axis = np.asarray(fallback_axis, dtype=float)
            axis = axis - source * float(np.dot(axis, source))
            if float(np.linalg.norm(axis)) < 1e-9:
                basis = np.eye(3)[int(np.argmin(np.abs(source)))]
                axis = np.asarray(basis, dtype=float)
                axis = axis - source * float(np.dot(axis, source))
            return self._axis_angle_rotation(axis, np.pi)
        cross = np.cross(source, target)
        skew = self._skew(cross)
        return np.eye(3) + skew + skew @ skew * ((1.0 - dot) / max(float(np.dot(cross, cross)), 1e-12))

    def _axis_angle_rotation(self, axis: np.ndarray, angle: float) -> np.ndarray:
        axis = np.asarray(axis, dtype=float)
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        skew = self._skew(axis)
        return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)

    def _skew(self, vector: np.ndarray) -> np.ndarray:
        x, y, z = np.asarray(vector, dtype=float)
        return np.asarray(
            [
                [0.0, -z, y],
                [z, 0.0, -x],
                [-y, x, 0.0],
            ],
            dtype=float,
        )

    def _rebase_q_sweep_to_current_hold(self) -> None:
        if self.q_sweep_s is None or self.q_sweep_positions is None or self.hold_point is None:
            return
        center_idx = int(np.argmin(np.abs(self.q_sweep_s)))
        center = np.asarray(self.q_sweep_positions[center_idx], dtype=float)
        current = np.asarray(self.hold_point.positions, dtype=float)
        offset = self.arm.joint_space.delta(current[None, :], center[None, :])[0]
        self.q_sweep_positions = self.q_sweep_positions + offset[None, :]
        self.get_logger().info(
            "rebased Q sweep to current P-adjust hold "
            f"center_idx={center_idx} "
            f"offset_max={float(np.max(np.abs(offset))):.6f}"
        )

    def _tcp_pose_from_joints(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tcp = self.arm.forward_kinematics(np.asarray(joints, dtype=float)[None, :]).tcp_transforms[0]
        return tcp[:3, 3].copy(), tcp[:3, :3].copy()

    def _pose_rotation(self, msg: PoseStamped) -> np.ndarray:
        q = msg.pose.orientation
        return rotations_from_quaternions(np.asarray([[q.w, q.x, q.y, q.z]]))[0]

    def _pose_position(self, msg: PoseStamped) -> np.ndarray:
        p = msg.pose.position
        return np.asarray([p.x, p.y, p.z], dtype=float)

    def _rotation_y_matrix(self, angle: float) -> np.ndarray:
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        return np.asarray(
            [
                [c, 0.0, s],
                [0.0, 1.0, 0.0],
                [-s, 0.0, c],
            ],
            dtype=float,
        )

    def _start_execution(self, trajectory: JointTrajectory, *, state: str, done_state: str) -> None:
        self._set_trajectory(trajectory)
        self._log_execution_continuity(trajectory, state=state)
        self.playback_start_s = self.get_clock().now().nanoseconds * 1e-9
        self.pending_request_id = ""
        self.pending_request_kind = ""
        self.state = state
        self.execution_done_state = done_state
        self._log_task_status(force=True)

    def _start_stage_execution(self, trajectory: JointTrajectory, *, stage_name: str) -> None:
        stage = str(stage_name).strip().lower()
        if stage not in ("slide", "p"):
            raise ValueError(f"unsupported stage execution: {stage_name!r}")
        if self.active_station_pose is None:
            raise RuntimeError("stage execution requires active station pose")
        self._set_trajectory(trajectory)
        self._log_execution_continuity(trajectory, state=f"{stage.upper()}_READY")
        self.stage_name = stage
        self.stage_playback_elapsed_s = 0.0
        self.stage_adjust_extra = 0.0
        self.stage_collision_checker = CollisionChecker(
            self.arm,
            [
                (
                    self.collision_model.station_meshes,
                    transform_from_pose_stamped(self.active_station_pose),
                )
            ],
            self.collision_model.arm_capsules,
        )
        self.advance_last_wall_time_s = None
        self.pending_request_id = ""
        self.pending_request_kind = ""
        self.state = "SLIDE_READY" if stage == "slide" else "P_READY"
        self.hold_point = trajectory.points[0]
        self._log_task_status(force=True)

    def _advance_dt(self) -> float:
        now_s = time.monotonic()
        last_s = self.advance_last_wall_time_s
        self.advance_last_wall_time_s = now_s
        if last_s is None:
            return 0.0
        return max(0.0, now_s - last_s)

    def _tick_stage_playback(self) -> None:
        if self.trajectory is None or self.trajectory_times is None:
            self._release_to_operator("stage playback missing trajectory", error=True)
            return
        if not self.advance_active:
            self.advance_last_wall_time_s = None
            self._publish_host_output(self.hold_point or self.trajectory.points[0])
            return

        self.stage_playback_elapsed_s = min(
            float(self.trajectory_times[-1]),
            self.stage_playback_elapsed_s + self._advance_dt(),
        )
        point = self._sample_trajectory_point(self.stage_playback_elapsed_s)
        self.hold_point = point
        self._publish_host_output(point)
        self.state = "SLIDE_EXECUTING" if self.stage_name == "slide" else "P_EXECUTING"
        if self.stage_playback_elapsed_s >= float(self.trajectory_times[-1]):
            self.trajectory = None
            self.trajectory_times = None
            self.advance_last_wall_time_s = None
            self.state = "SLIDE_ADJUST" if self.stage_name == "slide" else "P_ADJUST"
            self._log_task_status(force=True)

    def _tick_stage_adjust(self) -> None:
        if self.hold_point is None:
            self._publish_idle_host_output()
            return
        if not self.advance_active:
            self.advance_last_wall_time_s = None
            self._publish_host_output(self.hold_point)
            return
        dt_s = self._advance_dt()
        if dt_s <= 0.0:
            self._publish_host_output(self.hold_point)
            return
        if self.state == "SLIDE_ADJUST":
            self._apply_stage_forward_adjust(
                "slide",
                dt_s,
                rate=self.slide_forward_rate_m_s,
                max_extra=self.slide_forward_max_extra_m,
            )
        elif self.state == "P_ADJUST":
            self._apply_stage_forward_adjust(
                "p",
                dt_s,
                rate=self.p_forward_rate_rad_s,
                max_extra=self.p_forward_max_extra_rad,
            )
        self._publish_host_output(self.hold_point)

    def _apply_stage_forward_adjust(self, stage: str, dt_s: float, *, rate: float, max_extra: float) -> None:
        next_extra = min(max_extra, self.stage_adjust_extra + rate * float(dt_s))
        if next_extra <= self.stage_adjust_extra + 1e-12:
            self.get_logger().warn(f"{stage} forward adjust reached configured max", throttle_duration_sec=1.0)
            return
        delta = next_extra - self.stage_adjust_extra
        if self._apply_stage_adjust_step(stage, delta):
            self.stage_adjust_extra = next_extra

    def _apply_stage_adjust_step(self, stage: str, delta: float) -> bool:
        if self.active_station_pose is None or self.hold_point is None or self.stage_collision_checker is None:
            self._release_to_operator(
                "stage adjust missing station pose, collision checker, or hold point",
                error=True,
            )
            return False
        command_joints = np.asarray(self.hold_point.positions, dtype=float)
        position, rotation = self._tcp_pose_from_joints(command_joints)
        target_position, target_rotation = self._stage_forward_target(stage, position, rotation, float(delta))
        result = self._solve_roll_search_ik(
            target_position,
            target_rotation,
            command_joints,
            roll_shells=self._roll_search_shells(
                self.adjust_roll_search_max_rad,
                self.adjust_roll_search_step_rad,
            ),
            ik_branches=self.adjust_ik_branches,
            collision_checker=self.stage_collision_checker,
        )
        if result is None:
            self.get_logger().warn(f"{stage} forward adjust IK/collision failed", throttle_duration_sec=1.0)
            return False
        best, roll_delta, cost = result
        self.hold_point = self._zero_velocity_point(best)
        self.get_logger().debug(
            f"{stage} forward adjusted roll_delta_deg={np.rad2deg(roll_delta):.3f} cost={cost:.6f}"
        )
        return True

    def _solve_roll_search_ik(
        self,
        target_position: np.ndarray,
        tcp_rotation: np.ndarray,
        command_joints: np.ndarray,
        *,
        roll_shells,
        ik_branches: tuple[int, ...] | None,
        collision_checker: CollisionChecker | None,
    ) -> tuple[np.ndarray, float, float] | None:
        best, reason, roll_delta, cost, _diagnostics = self._solve_roll_search_ik_detailed(
            target_position,
            tcp_rotation,
            command_joints,
            roll_shells=roll_shells,
            ik_branch_sets=(ik_branches,),
            collision_checker=collision_checker,
        )
        if reason != "ok" or best is None:
            return None
        return best, roll_delta, cost

    def _dedupe_ik_branch_sets(
        self,
        ik_branch_sets,
    ) -> tuple[tuple[int, ...] | None, ...]:
        deduped: list[tuple[int, ...] | None] = []
        seen = set()
        for branches in ik_branch_sets:
            key = None if branches is None else tuple(int(v) for v in branches)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        return tuple(deduped)

    def _branch_set_label(self, branches: tuple[int, ...] | None) -> str:
        return "all" if branches is None else "[" + ",".join(str(v) for v in branches) + "]"

    def _solve_roll_search_ik_detailed(
        self,
        target_position: np.ndarray,
        tcp_rotation: np.ndarray,
        command_joints: np.ndarray,
        *,
        roll_shells,
        ik_branch_sets,
        collision_checker: CollisionChecker | None,
    ) -> tuple[np.ndarray | None, str, float, float, str]:
        target = np.asarray(target_position, dtype=float)
        reference = np.asarray(command_joints, dtype=float)
        deduped_branch_sets = self._dedupe_ik_branch_sets(ik_branch_sets)
        shell_tuple = tuple(roll_shells)
        diagnostic_parts = []
        total_ik = 0
        total_free = 0
        best: np.ndarray | None = None
        best_roll_delta = 0.0
        best_cost = float("inf")

        for branches in deduped_branch_sets:
            branch_ik = 0
            branch_free = 0
            branch_best_roll: float | None = None
            branch_best_cost = float("inf")
            for roll_shell in shell_tuple:
                roll_deltas = np.asarray(tuple(roll_shell), dtype=float)
                if roll_deltas.size == 0:
                    continue
                rotations = np.asarray(
                    [tcp_rotation @ self._roll_x_matrix(delta) for delta in roll_deltas],
                    dtype=float,
                )
                tcp_positions = np.repeat(target[None, :], roll_deltas.size, axis=0)
                targets = np.broadcast_to(np.eye(4), (len(rotations), 4, 4)).copy()
                targets[:, :3, :3] = rotations
                targets[:, :3, 3] = tcp_positions
                joints_batch, valid_batch = self.arm.solve_ik(targets, branches=branches)
                roll_indices, ik_indices = np.nonzero(valid_batch)
                branch_ik += int(roll_indices.size)
                total_ik += int(roll_indices.size)
                if roll_indices.size == 0:
                    continue

                candidates = joints_batch[roll_indices, ik_indices, :]
                candidates = self.arm.joint_space.align_to_reference(
                    candidates,
                    np.broadcast_to(reference, candidates.shape),
                )
                candidate_rolls = roll_deltas[roll_indices]
                if collision_checker is not None:
                    free_mask = ~collision_checker.check_configs(candidates)
                    candidates = candidates[free_mask]
                    candidate_rolls = candidate_rolls[free_mask]
                branch_free += int(candidates.shape[0])
                total_free += int(candidates.shape[0])
                if candidates.size == 0:
                    continue

                costs = np.sum(
                    np.abs(self.arm.joint_space.delta(candidates, np.broadcast_to(reference, candidates.shape))),
                    axis=1,
                )
                local_best_idx = int(np.argmin(costs))
                local_cost = float(costs[local_best_idx])
                local_roll_delta = float(candidate_rolls[local_best_idx])
                if local_cost < branch_best_cost:
                    branch_best_cost = local_cost
                    branch_best_roll = local_roll_delta
                if local_cost < best_cost:
                    best_cost = local_cost
                    best = candidates[local_best_idx]
                    best_roll_delta = local_roll_delta

            best_roll_text = "none" if branch_best_roll is None else f"{np.rad2deg(branch_best_roll):.2f}"
            diagnostic_parts.append(
                f"branches={self._branch_set_label(branches)}:"
                f"ik={branch_ik}:free={branch_free}:best_roll_deg={best_roll_text}"
            )

        diagnostics = "branch_sets=" + ";".join(diagnostic_parts)
        if best is None:
            if total_ik <= 0:
                return None, "ik", 0.0, float("inf"), diagnostics
            if total_free <= 0:
                return None, "collision", 0.0, float("inf"), diagnostics
            return None, "no_free_candidate", 0.0, float("inf"), diagnostics
        return best, "ok", best_roll_delta, best_cost, diagnostics

    def _stage_forward_target(
        self,
        stage: str,
        tcp_position: np.ndarray,
        tcp_rotation: np.ndarray,
        delta: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        local_position, local_rotation = self._local_exchange_frame(stage, extra=float(self.stage_adjust_extra))
        tcp_offset_local = local_rotation.T @ (np.asarray(tcp_position, dtype=float) - local_position)
        tcp_rotation_local = local_rotation.T @ np.asarray(tcp_rotation, dtype=float)
        next_position, next_rotation = self._local_exchange_frame(
            stage,
            extra=float(self.stage_adjust_extra) + float(delta),
        )
        return next_position + next_rotation @ tcp_offset_local, next_rotation @ tcp_rotation_local

    def _local_exchange_frame(self, stage: str, *, extra: float) -> tuple[np.ndarray, np.ndarray]:
        if self.active_station_pose is None:
            raise RuntimeError("local exchange frame requires active station pose")
        station_rotation = self._pose_rotation(self.active_station_pose)
        station_position = self._pose_position(self.active_station_pose)
        stage_name = str(stage).strip().lower()
        if stage_name == "insert":
            translation = np.zeros(3, dtype=float)
            rotation = np.eye(3)
        elif stage_name == "slide":
            translation = np.asarray([0.0, 0.0, self.stage_slide_z_mag + float(extra)], dtype=float)
            rotation = np.eye(3)
        elif stage_name == "p":
            theta_p = 0.5 * np.pi + float(extra)
            translation = np.asarray([0.0, 0.0, self.stage_slide_z_mag], dtype=float) + self._p_curve(theta_p)
            rotation = self._rotation_y_matrix(-theta_p)
        elif stage_name == "q":
            theta_p = 0.5 * np.pi
            theta_q = float(extra)
            translation = (
                np.asarray([0.0, 0.0, self.stage_slide_z_mag], dtype=float)
                + self._p_curve(theta_p)
                + self._q_curve(theta_q)
            )
            rotation = self._roll_x_matrix(theta_q) @ self._rotation_y_matrix(-theta_p)
        else:
            raise ValueError(f"unsupported local exchange stage: {stage!r}")
        return station_position + station_rotation @ translation, station_rotation @ rotation

    def _p_curve(self, theta_p: float) -> np.ndarray:
        return np.asarray(
            [
                self.stage_r_p * np.cos(theta_p - self.stage_phi) - self.stage_r_p * np.cos(self.stage_phi),
                0.0,
                self.stage_r_p * np.sin(theta_p - self.stage_phi) + self.stage_r_p * np.sin(self.stage_phi),
            ],
            dtype=float,
        )

    def _q_curve(self, theta_q: float) -> np.ndarray:
        return self._roll_x_matrix(theta_q) @ np.asarray([0.0, 0.0, self.stage_r_q], dtype=float) + np.asarray(
            [0.0, 0.0, -self.stage_r_q],
            dtype=float,
        )

    def _finish_execution(self) -> None:
        if self.trajectory is None:
            raise RuntimeError("finish execution requires an active trajectory")
        was_recovery = self.state == "RECOVERY_EXECUTING"
        recovery_mode = self.recovery_mode
        done_state = self.execution_done_state
        final_point = self.trajectory.points[-1]
        self.trajectory = None
        self.trajectory_times = None
        if was_recovery and done_state == "RECOVERY_STAGE_DONE":
            self.hold_point = final_point
            if recovery_mode in (
                RecoveryPlanningRequest.MODE_EXTRACT_FOR_RETRY,
                RecoveryPlanningRequest.MODE_SIDE_RELEASE,
            ):
                self._request_recovery_plan(RecoveryPlanningRequest.MODE_RETRACT_TO_APPROACH_START)
            elif recovery_mode == RecoveryPlanningRequest.MODE_RETRACT_TO_APPROACH_START:
                self.state = "IDLE"
                self._clear_exchange_context()
                self._log_task_status(force=True)
            else:
                self._release_to_operator(f"unknown recovery mode finished: {recovery_mode}", error=True)
            return

        self.state = done_state
        if done_state == "IDLE":
            self.hold_point = None
            if was_recovery:
                self._clear_exchange_context()
        else:
            self.hold_point = final_point
            if done_state == "READY_TO_EXCHANGE":
                self.ready_adjust_last_wall_time_s = None
            if done_state == "Q_MANUAL_ADJUST":
                self.q_manual_point = final_point
                self.q_manual_angle = 0.0
                self.q_manual_target_angle = 0.0
        self._log_task_status(force=True)

    def _release_to_operator(self, reason: str, *, error: bool) -> None:
        self.get_logger().info(f"release to operator: {reason}")
        self.state = "ERROR" if error else "IDLE"
        self._clear_exchange_context()
        self._log_task_status(force=True)

    def _reset_task(self, reason: str, *, target_state: str) -> None:
        self.get_logger().info(f"reset task: {reason}")
        self.state = target_state
        self._clear_exchange_context()

    def _clear_exchange_context(self) -> None:
        self.pending_start_level = None
        self.pending_request_id = ""
        self.pending_request_kind = ""
        self.recovery_mode = 0
        self.recovery_exchange_local_pose = None
        self.trajectory = None
        self.trajectory_times = None
        self.hold_point = None
        self.advance_active = False
        self.advance_last_wall_time_s = None
        self.stage_playback_elapsed_s = 0.0
        self.stage_name = ""
        self.confirmed_exchange_stage = "insert"
        self.stage_adjust_extra = 0.0
        self.stage_collision_checker = None
        self.active_station_pose = None
        self.approach_start_joint_state = None
        self.ready_adjust_last_wall_time_s = None
        self._clear_q_manual_context()

    def _normalize_ik_branches(self, value) -> tuple[int, ...] | None:
        if value is None:
            return None
        branches = tuple(int(v) for v in value)
        if not branches:
            return None
        invalid = [branch for branch in branches if branch < 0 or branch >= 8]
        if invalid:
            raise ValueError(f"ready_adjust.ik_branches must be in [0, 7], got {invalid}")
        if len(set(branches)) != len(branches):
            raise ValueError(f"ready_adjust.ik_branches must not contain duplicates, got {branches}")
        return branches

    def _clear_q_manual_context(self) -> None:
        self.q_manual_point = None
        self.q_sweep_s = None
        self.q_sweep_positions = None
        self.q_manual_angle = 0.0
        self.q_manual_target_angle = 0.0
        self._last_q_manual_solve_wall_time_s = 0.0
        self._last_q_manual_step_wall_time_s = 0.0

    def _log_execution_continuity(self, trajectory: JointTrajectory, *, state: str) -> None:
        if not trajectory.points:
            return
        first = np.asarray(trajectory.points[0].positions, dtype=float)
        fields = [f"start_state={state}"]
        if self.hold_point is not None:
            hold = np.asarray(self.hold_point.positions, dtype=float)
            delta = self.arm.joint_space.delta(first[None, :], hold[None, :])[0]
            fields.append(f"first_minus_hold_max={float(np.max(np.abs(delta))):.6f}")
        if self.latest_joint_state is not None:
            current = joint_positions_from_joint_state(self.latest_joint_state)
            delta = self.arm.joint_space.delta(first[None, :], current[None, :])[0]
            fields.append(f"first_minus_current_max={float(np.max(np.abs(delta))):.6f}")
        fields.append(f"first={np.round(first, 6).tolist()}")
        self.get_logger().info("execution continuity " + " ".join(fields))

    def _publish_state(self) -> None:
        with self._lock:
            state = (
                f"{self.state}:level={self.selected_exchange_level}:"
                f"q={np.rad2deg(self.q_manual_angle):.1f}:"
                f"q_target={np.rad2deg(self.q_manual_target_angle):.1f}"
            )
            self._log_task_status()
        self.state_pub.publish(String(data=state))

    def _log_task_status(self, *, force: bool = False) -> None:
        status = (
            f"task state={self.state} "
            f"level={self.selected_exchange_level} "
            f"q_manual_deg={np.rad2deg(self.q_manual_angle):.1f} "
            f"q_target_deg={np.rad2deg(self.q_manual_target_angle):.1f} "
            f"pending={self.pending_request_kind or '-'} "
            f"recovery_mode={self.recovery_mode} "
            f"stage={self.stage_name or '-'} "
            f"advance={int(self.advance_active)} "
            f"stage_extra={self.stage_adjust_extra:.4f}"
        )
        if force or status != self._last_logged_task_status:
            self._last_logged_task_status = status
            self.get_logger().info(status)

    def _update_q_manual_target(self, target_angle: float) -> None:
        if self.q_sweep_s is None or self.q_sweep_positions is None:
            self.get_logger().warn("ignore Q target: missing planned Q sweep cache")
            return
        candidate_angle = float(np.clip(
            target_angle,
            self.q_manual_min_angle,
            self.q_manual_max_angle,
        ))
        self.q_manual_target_angle = candidate_angle
        self._log_task_status(force=True)

    def _step_q_manual_target(self) -> None:
        if self.q_sweep_s is None or self.q_sweep_positions is None:
            return
        now = time.monotonic()
        if (
            self.q_manual_command_sample_dt > self._tick_period_s
            and now - self._last_q_manual_solve_wall_time_s < self.q_manual_command_sample_dt
        ):
            return
        step_dt = self.q_manual_command_sample_dt
        if self._last_q_manual_step_wall_time_s > 0.0:
            step_dt = max(now - self._last_q_manual_step_wall_time_s, self._tick_period_s)
        self._last_q_manual_solve_wall_time_s = now
        self._last_q_manual_step_wall_time_s = now
        target = float(np.clip(
            self.q_manual_target_angle,
            self.q_manual_min_angle,
            self.q_manual_max_angle,
        ))
        if self.q_manual_rate_limit > 0.0:
            max_step = self.q_manual_rate_limit * step_dt
            delta = float(np.clip(target - self.q_manual_angle, -max_step, max_step))
            candidate_angle = self.q_manual_angle + delta
        else:
            candidate_angle = target
        self._set_q_manual_command_angle(candidate_angle)

    def _set_q_manual_command_angle(self, candidate_angle: float) -> None:
        candidate_angle = float(np.clip(
            candidate_angle,
            self.q_manual_min_angle,
            self.q_manual_max_angle,
        ))
        s_target = float(np.clip(candidate_angle / (0.5 * np.pi), -1.0, 1.0))
        q_target = np.asarray(
            [
                np.interp(s_target, self.q_sweep_s, self.q_sweep_positions[:, j])
                for j in range(self.q_sweep_positions.shape[1])
            ],
            dtype=float,
        )
        candidate = self._zero_velocity_point(q_target)
        self.q_manual_angle = candidate_angle
        self.q_manual_point = candidate
        self.hold_point = candidate

    def _cache_q_sweep(self, msg: Type3PlanningResult) -> bool:
        s = np.asarray(msg.q_sweep_s, dtype=float)
        positions = np.asarray(
            [point.positions for point in msg.q_sweep_trajectory.points],
            dtype=float,
        )
        if s.ndim != 1 or s.size == 0:
            self.get_logger().warn("type3 result Q sweep has empty s grid")
            return False
        if positions.shape != (s.size, 6):
            self.get_logger().warn(
                f"type3 result Q sweep positions shape mismatch: s={s.shape} positions={positions.shape}"
            )
            return False
        if np.any(~np.isfinite(s)) or np.any(~np.isfinite(positions)):
            self.get_logger().warn("type3 result Q sweep contains NaN/Inf")
            return False
        if np.any(np.diff(s) <= 0.0):
            self.get_logger().warn("type3 result Q sweep s grid must be strictly increasing")
            return False
        self.q_sweep_s = s
        self.q_sweep_positions = positions
        self.get_logger().info(
            f"cached Q sweep points={int(s.size)} s_range=[{float(s[0]):.3f}, {float(s[-1]):.3f}]"
        )
        return True

    def _host_state_code(self) -> int:
        if self.state in ("RECOVERY_PLANNING", "RECOVERY_EXECUTING"):
            states = RECOVERY_HOST_STATE_BY_MODE.get(int(self.recovery_mode))
            if states is None:
                return int(
                    ArmHost2MCUMsg.HOST_RECOVERY_PLANNING
                    if self.state == "RECOVERY_PLANNING"
                    else ArmHost2MCUMsg.HOST_RECOVERY_EXECUTING
                )
            return int(states[0] if self.state == "RECOVERY_PLANNING" else states[1])
        return int(HOST_STATE_BY_TASK_STATE[self.state])

    def _zero_velocity_point(self, positions: np.ndarray) -> JointTrajectoryPoint:
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in np.asarray(positions, dtype=float)]
        point.velocities = [0.0] * len(point.positions)
        point.accelerations = [0.0] * len(point.positions)
        return point

    def _publish_host_output(self, point: JointTrajectoryPoint) -> None:
        msg = host_output_from_trajectory_point(
            point,
            self.get_clock().now().to_msg(),
            self._host_state_code(),
        )
        self.command_pub.publish(msg)

    def _publish_idle_host_output(self) -> None:
        point = self.hold_point or self.q_manual_point
        if point is None:
            point = JointTrajectoryPoint()
            point.positions = [0.0] * 6
            point.velocities = [0.0] * 6
            point.accelerations = [0.0] * 6
        self._publish_host_output(point)

    def _set_trajectory(self, trajectory: JointTrajectory) -> None:
        if not trajectory.points:
            raise ValueError("planning result trajectory has no points")
        times = np.asarray([duration_to_seconds(point.time_from_start) for point in trajectory.points], dtype=float)
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("planning result trajectory times must be strictly increasing")
        self.trajectory = trajectory
        self.trajectory_times = times

    def _sample_trajectory_point(self, elapsed_s: float) -> JointTrajectoryPoint:
        if self.trajectory is None or self.trajectory_times is None:
            raise RuntimeError("trajectory sample requires an active trajectory")
        index = int(np.searchsorted(self.trajectory_times, float(elapsed_s), side="right") - 1)
        index = int(np.clip(index, 0, len(self.trajectory.points) - 1))
        return self.trajectory.points[index]

    def destroy_node(self) -> bool:
        self._stop_event.set()
        self._tick_thread.join()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TaskNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
