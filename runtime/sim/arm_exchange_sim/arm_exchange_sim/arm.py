from __future__ import annotations

import mujoco
import numpy as np
from arm_exchange_interfaces.msg import ArmFeedforwardWrenchMsg, ArmHost2MCUMsg, ArmMCU2HostMsg
from arm_exchange_core import load_config
from arm_exchange_core.arm_model import ArmModel
from mujoco_engine.plugin_base import BasePlugin, PluginContext, PluginSetupContext
from sensor_msgs.msg import JointState
from std_msgs.msg import String


class ArmPlugin(BasePlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.joint_names = tuple(str(name) for name in kwargs.get("joint_names", ()))
        self.motor_names = tuple(
            str(name) for name in kwargs.get(
                "motor_names",
                tuple(f"motor_joint{i + 1}" for i in range(len(self.joint_names))),
            )
        )
        self.topic = str(kwargs.get("topic", "/host/arm/host_output"))
        self.joint_state_topic = str(kwargs.get("joint_state_topic", "/joint_states"))
        self.joint_state_frame = str(kwargs.get("joint_state_frame", "arm_base"))
        self.wrench_topic = str(kwargs.get("wrench_topic", "/host/arm/feedforward_wrench"))
        self.wrench_timeout_s = float(kwargs.get("wrench_timeout_s", 0.1))
        self.host_enabled_topic = str(kwargs.get("host_enabled_topic", "/sim/mcu/arm_enabled"))
        self.host_enabled_default = bool(kwargs.get("host_enabled_default", False))
        self.mcu_state_topic = str(kwargs.get("mcu_state_topic", "/mcu/arm/state"))
        self.authority_state_topic = str(kwargs.get("authority_state_topic", "/sim/state/control_authority"))
        self.home_positions_cfg = kwargs.get("home_positions")
        self.use_acceleration_feedforward = bool(kwargs.get("use_acceleration_feedforward", False))
        self.include_link_inertia_moments = bool(kwargs.get("include_link_inertia_moments", True))
        self.feedforward_update_period_s = float(kwargs.get("feedforward_update_period_s", 0.01))
        self.debug_log_period_s = float(kwargs.get("debug_log_period_s", 0.0))
        self.torque_sign = self._array_cfg(
            kwargs.get("torque_sign", [1.0] * len(self.joint_names)),
            "torque_sign",
        )
        self.pid_kp = self._array_cfg(
            kwargs.get("pid_kp", [60.0, 80.0, 60.0, 20.0, 15.0, 10.0]),
            "pid_kp",
        )
        self.pid_kd = self._array_cfg(
            kwargs.get("pid_kd", [8.0, 10.0, 8.0, 3.0, 2.0, 1.0]),
            "pid_kd",
        )
        self.pid_ki = self._array_cfg(kwargs.get("pid_ki", [0.0] * len(self.joint_names)), "pid_ki")
        self.integral_limit = self._array_cfg(
            kwargs.get("integral_limit", [0.2] * len(self.joint_names)),
            "integral_limit",
        )
        self.torque_limit = self._array_cfg(
            kwargs.get("torque_limit", [80.0, 80.0, 80.0, 30.0, 20.0, 20.0]),
            "torque_limit",
        )
        self.control_authority = "host" if self.host_enabled_default else "operator"
        self.host_enabled = self.host_enabled_default
        self.arm_state = "HOMING"
        self.last_command_time = None
        self._last_step_time = None
        self.home_positions = None
        self.host_target_position = None
        self.host_target_velocity = None
        self.host_target_acceleration = None
        self.wrench_enabled = False
        self.wrench_host_state = int(ArmHost2MCUMsg.HOST_IDLE)
        self.wrench_force_tcp = np.zeros(3, dtype=float)
        self.wrench_torque_tcp = np.zeros(3, dtype=float)
        self.last_wrench_time = None
        self._cached_tau_ff = np.zeros(len(self.joint_names), dtype=float)
        self._last_feedforward_time = None
        self._last_debug_log_time = None
        self.integral_error = np.zeros(len(self.joint_names), dtype=float)
        self.arm = ArmModel.from_config(load_config()["arm"])
        self.ros_subscriptions[self.topic] = "arm_exchange_interfaces.msg.ArmHost2MCUMsg"
        self.ros_subscriptions[self.host_enabled_topic] = "std_msgs.msg.Bool"
        self.ros_subscriptions[self.wrench_topic] = "arm_exchange_interfaces.msg.ArmFeedforwardWrenchMsg"

        if len(self.joint_names) != 6:
            raise ValueError("ArmPlugin expects exactly 6 arm joints")
        if len(self.motor_names) != len(self.joint_names):
            raise ValueError("ArmPlugin motor_names must match joint_names length")

    def _array_cfg(self, value, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        if self.joint_names and arr.shape != (len(self.joint_names),):
            raise ValueError(f"ArmPlugin {name} must match joint_names length")
        return arr

    def setup(self, context: PluginSetupContext):
        self.authority_pub = context.node.create_publisher(String, self.authority_state_topic, 10)
        self.mcu_state_pub = context.node.create_publisher(ArmMCU2HostMsg, self.mcu_state_topic, 10)
        self.joint_state_pub = context.node.create_publisher(JointState, self.joint_state_topic, 10)

    def on_compile_callback(self, context: PluginContext):
        model = context.model
        self.qpos_addr = {}
        self.qvel_addr = {}
        self.actuator_ids = {}
        for name in self.joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"ArmPlugin missing joint: {name}")
            self.qpos_addr[name] = int(model.jnt_qposadr[joint_id])
            self.qvel_addr[name] = int(model.jnt_dofadr[joint_id])
        for name in self.motor_names:
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id < 0:
                raise ValueError(f"ArmPlugin missing actuator: {name}")
            self.actuator_ids[name] = int(actuator_id)
        if self.home_positions_cfg is None:
            self.home_positions = np.asarray(
                [context.data.qpos[self.qpos_addr[name]] for name in self.joint_names],
                dtype=float,
            )
        else:
            self.home_positions = np.asarray(self.home_positions_cfg, dtype=float)
            if self.home_positions.shape != (len(self.joint_names),):
                raise ValueError(
                    "ArmPlugin home_positions must match joint_names length, "
                    f"got {self.home_positions.shape} and {len(self.joint_names)}"
                )
        zeros = np.zeros(len(self.joint_names), dtype=float)
        self._cached_tau_ff = self._compute_feedforward_torque(self.home_positions, zeros, zeros)

    def on_message_callback(self, context: PluginContext, topic: str, msg):
        if topic == self.host_enabled_topic:
            self._set_enabled(bool(msg.data))
            self.authority_pub.publish(String(data=self.control_authority))
            return
        if topic == self.wrench_topic:
            self._handle_feedforward_wrench(context, msg)
            return
        if topic != self.topic:
            raise ValueError(f"ArmPlugin received unexpected topic: {topic}")
        if not self.host_enabled:
            return
        if not self._host_state_tracks(int(msg.host_state)):
            return
        if len(msg.position) != len(self.joint_names):
            raise ValueError("ArmHost2MCUMsg position length must match joint_names")
        if len(msg.velocity) != len(self.joint_names):
            raise ValueError("ArmHost2MCUMsg velocity length must match joint_names")
        self.host_target_position = np.asarray(msg.position, dtype=float)
        self.host_target_velocity = np.asarray(msg.velocity, dtype=float)
        self.host_target_acceleration = np.zeros(len(self.joint_names), dtype=float)
        self.arm_state = "HOST_TRACKING"
        self.last_command_time = float(context.data.time)
        self.authority_pub.publish(String(data=self.control_authority))

    def _handle_feedforward_wrench(self, context: PluginContext, msg: ArmFeedforwardWrenchMsg) -> None:
        self.wrench_enabled = bool(msg.enabled)
        self.wrench_host_state = int(msg.host_state)
        self.wrench_force_tcp = np.asarray(msg.force_tcp_n, dtype=float)
        self.wrench_torque_tcp = np.asarray(msg.torque_tcp_nm, dtype=float)
        self.last_wrench_time = float(context.data.time)

    def on_step_callback(self, context: PluginContext):
        now = float(context.data.time)
        if self._last_step_time is None:
            self._last_step_time = now
            return
        dt = max(0.0, now - self._last_step_time)
        self._last_step_time = now
        self._run_controller(context, dt)
        self._publish_mcu_state(context)

    def on_timer_callback(self, context: PluginContext, alias: str):
        q, qd = self._read_arm_state(context)
        msg = JointState()
        msg.header.stamp = context.node.get_clock().now().to_msg()
        msg.header.frame_id = self.joint_state_frame
        msg.name = list(self.joint_names)
        msg.position = q.tolist()
        msg.velocity = qd.tolist()
        self.joint_state_pub.publish(msg)

    def _set_enabled(self, enabled: bool) -> None:
        self.host_enabled = bool(enabled)
        authority = "host" if self.host_enabled else "operator"
        if self.host_enabled and self.control_authority != "host":
            self._capture_current_as_host_target()
            self.arm_state = "HOST_HOLD"
        elif not self.host_enabled:
            self.arm_state = "HOMING"
        if authority != self.control_authority:
            self.integral_error[:] = 0.0
        self.control_authority = authority

    def _host_state_tracks(self, host_state: int) -> bool:
        return host_state in (
            ArmHost2MCUMsg.HOST_EXECUTING_APPROACH,
            ArmHost2MCUMsg.HOST_READY_TO_EXCHANGE,
            ArmHost2MCUMsg.HOST_EXECUTING_EXCHANGE,
            ArmHost2MCUMsg.HOST_SLIDE_READY,
            ArmHost2MCUMsg.HOST_SLIDE_EXECUTING,
            ArmHost2MCUMsg.HOST_SLIDE_ADJUST,
            ArmHost2MCUMsg.HOST_P_READY,
            ArmHost2MCUMsg.HOST_P_EXECUTING,
            ArmHost2MCUMsg.HOST_P_ADJUST,
            ArmHost2MCUMsg.HOST_RECOVERY_EXECUTING,
            ArmHost2MCUMsg.HOST_RECOVERY_EXTRACT_EXECUTING,
            ArmHost2MCUMsg.HOST_RECOVERY_SIDE_RELEASE_EXECUTING,
            ArmHost2MCUMsg.HOST_RECOVERY_RETRACT_EXECUTING,
            ArmHost2MCUMsg.HOST_Q_MANUAL,
        )

    def _capture_current_as_host_target(self) -> None:
        # The simulated lower controller owns holding once host has authority,
        # even before the host sends a new trajectory point.
        self.host_target_position = None
        self.host_target_velocity = None
        self.host_target_acceleration = None

    def _run_controller(self, context: PluginContext, dt: float) -> None:
        q, qd = self._read_arm_state(context)
        q_ref, qd_ref, qdd_ref = self._select_reference(q)
        error = self._joint_error(q_ref, q)
        if dt > 0.0:
            self.integral_error += error * dt
            self.integral_error = np.clip(
                self.integral_error,
                -self.integral_limit,
                self.integral_limit,
            )

        qdd_ff = qdd_ref if self.use_acceleration_feedforward else np.zeros(6, dtype=float)
        tau_ff = self._feedforward_torque(context, q_ref, qd_ref, qdd_ff)
        tau_virtual = self._virtual_wrench_torque(context, q)
        tau_pid = self.pid_kp * error + self.pid_kd * (qd_ref - qd) + self.pid_ki * self.integral_error
        tau_cmd = self.torque_sign * (tau_ff + tau_virtual + tau_pid)
        tau_cmd = np.clip(tau_cmd, -self.torque_limit, self.torque_limit)
        for motor_name, torque in zip(self.motor_names, tau_cmd, strict=True):
            context.data.ctrl[self.actuator_ids[motor_name]] = float(torque)
        self._maybe_log_diagnostics(context, error, tau_ff + tau_virtual, tau_pid, tau_cmd)

    def _virtual_wrench_torque(self, context: PluginContext, q: np.ndarray) -> np.ndarray:
        if not self.host_enabled or not self.wrench_enabled:
            return np.zeros(len(self.joint_names), dtype=float)
        if self.last_wrench_time is None:
            return np.zeros(len(self.joint_names), dtype=float)
        if (
            self.wrench_timeout_s > 0.0
            and float(context.data.time) - self.last_wrench_time > self.wrench_timeout_s
        ):
            return np.zeros(len(self.joint_names), dtype=float)
        if not self._host_state_accepts_wrench(self.wrench_host_state):
            return np.zeros(len(self.joint_names), dtype=float)

        wrench = np.concatenate((self.wrench_force_tcp, self.wrench_torque_tcp))[None, :]
        return self.arm.external_wrench_torque(q[None, :], wrench)[0]

    def _host_state_accepts_wrench(self, host_state: int) -> bool:
        return host_state in (
            ArmHost2MCUMsg.HOST_SLIDE_EXECUTING,
            ArmHost2MCUMsg.HOST_SLIDE_ADJUST,
            ArmHost2MCUMsg.HOST_P_EXECUTING,
            ArmHost2MCUMsg.HOST_P_ADJUST,
            ArmHost2MCUMsg.HOST_Q_MANUAL,
        )

    def _publish_mcu_state(self, context: PluginContext) -> None:
        q, qd = self._read_arm_state(context)
        msg = ArmMCU2HostMsg()
        msg.header.stamp = context.node.get_clock().now().to_msg()
        msg.enabled = bool(self.host_enabled)
        msg.position = [float(value) for value in q]
        msg.velocity = [float(value) for value in qd]
        self.mcu_state_pub.publish(msg)

    def _feedforward_torque(
        self,
        context: PluginContext,
        q_ref: np.ndarray,
        qd_ref: np.ndarray,
        qdd_ref: np.ndarray,
    ) -> np.ndarray:
        now = float(context.data.time)
        if (
            self._last_feedforward_time is None
            or self.feedforward_update_period_s <= 0.0
            or now - self._last_feedforward_time >= self.feedforward_update_period_s
        ):
            self._cached_tau_ff = self._compute_feedforward_torque(q_ref, qd_ref, qdd_ref)
            self._last_feedforward_time = now
        return self._cached_tau_ff

    def _compute_feedforward_torque(
        self,
        q_ref: np.ndarray,
        qd_ref: np.ndarray,
        qdd_ref: np.ndarray,
    ) -> np.ndarray:
        return self.arm.inverse_dynamics(
            q_ref[None, :],
            qd_ref[None, :],
            qdd_ref[None, :],
            include_link_inertia=self.include_link_inertia_moments,
        )[0]

    def _read_arm_state(self, context: PluginContext) -> tuple[np.ndarray, np.ndarray]:
        data = context.data
        q = np.asarray([data.qpos[self.qpos_addr[name]] for name in self.joint_names], dtype=float)
        qd = np.asarray([data.qvel[self.qvel_addr[name]] for name in self.joint_names], dtype=float)
        return q, qd

    def _maybe_log_diagnostics(
        self,
        context: PluginContext,
        error: np.ndarray,
        tau_ff: np.ndarray,
        tau_pid: np.ndarray,
        tau_cmd: np.ndarray,
    ) -> None:
        if self.debug_log_period_s <= 0.0:
            return
        now = float(context.data.time)
        if (
            self._last_debug_log_time is not None
            and now - self._last_debug_log_time < self.debug_log_period_s
        ):
            return
        self._last_debug_log_time = now
        saturation = np.abs(tau_cmd) >= (self.torque_limit - 1e-6)
        max_error = float(np.max(np.abs(error)))
        max_tau = float(np.max(np.abs(tau_cmd)))
        sat_count = int(np.count_nonzero(saturation))
        context.node.get_logger().info(
            "arm torque ctrl "
            f"authority={self.control_authority} state={self.arm_state} "
            f"max_abs_error={max_error:.4f} max_abs_tau={max_tau:.3f} "
            f"saturated_joints={sat_count} "
            f"tau_ff={np.round(tau_ff, 3).tolist()} "
            f"tau_pid={np.round(tau_pid, 3).tolist()} "
            f"tau_cmd={np.round(tau_cmd, 3).tolist()}"
        )

    def _select_reference(self, current_q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        zeros = np.zeros(len(self.joint_names), dtype=float)
        if not self.host_enabled:
            self.arm_state = "HOMING"
            return self.home_positions, zeros, zeros
        if self.host_target_position is None:
            self.host_target_position = np.asarray(current_q, dtype=float).copy()
            self.host_target_velocity = zeros.copy()
            self.host_target_acceleration = zeros.copy()
            self.arm_state = "HOST_HOLD"
        qd_ref = self.host_target_velocity if self.host_target_velocity is not None else zeros
        qdd_ref = self.host_target_acceleration if self.host_target_acceleration is not None else zeros
        return self.host_target_position, qd_ref, qdd_ref

    def _joint_error(self, target: np.ndarray, current: np.ndarray) -> np.ndarray:
        return self.arm.joint_space.delta(target[None, :], current[None, :])[0]
