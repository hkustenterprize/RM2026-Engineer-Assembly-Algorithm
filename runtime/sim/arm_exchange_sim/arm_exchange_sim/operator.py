from __future__ import annotations

import math

from arm_exchange_interfaces.msg import (
    ArmCtrlAdvanceMsg,
    ArmCtrlGimbalControlMsg,
    ArmCtrlQMsg,
    ArmCtrlStartMsg,
    ArmCtrlWithdrawMsg,
    ArmCtrlXYZMsg,
    ArmHost2MCUMsg,
    OperatorInputState,
)
import mujoco
from geometry_msgs.msg import Vector3
from mujoco_engine.plugin_base import BasePlugin, PluginContext, PluginSetupContext
from std_msgs.msg import Bool, Empty, String


def _wrap_pi(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


class OperatorLogicPlugin(BasePlugin):
    MODES = ("teleop", "arm_manual", "exchange_manual_q")
    CAMERAS = ("left", "right")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_state_topic = str(kwargs.get("input_state_topic", "/operator/input_state"))
        self.host_enabled_topic = str(kwargs.get("host_enabled_topic", "/sim/mcu/arm_enabled"))
        self.ctrl_start_topic = str(kwargs.get("ctrl_start_topic", "/mcu/arm/ctrl_start"))
        self.ctrl_enter_topic = str(kwargs.get("ctrl_enter_topic", "/mcu/arm/ctrl_enter"))
        self.ctrl_advance_topic = str(kwargs.get("ctrl_advance_topic", "/mcu/arm/ctrl_advance"))
        self.ctrl_gimbal_topic = str(kwargs.get("ctrl_gimbal_topic", "/mcu/gimbal/control"))
        self.ctrl_q_topic = str(kwargs.get("ctrl_q_topic", "/mcu/arm/ctrl_q"))
        self.ctrl_xyz_topic = str(kwargs.get("ctrl_xyz_topic", "/mcu/arm/ctrl_xyz"))
        self.ctrl_withdraw_topic = str(kwargs.get("ctrl_withdraw_topic", "/mcu/arm/ctrl_withdraw"))
        self.host_output_topic = str(kwargs.get("host_output_topic", "/host/arm/host_output"))
        self.mode_state_topic = str(kwargs.get("mode_state_topic", "/sim/state/operator_mode"))
        self.authority_state_topic = str(kwargs.get("authority_state_topic", "/sim/state/control_authority"))
        self.active_camera_state_topic = str(kwargs.get("active_camera_state_topic", "/sim/state/active_camera"))
        self.gimbal_state_topic = str(kwargs.get("gimbal_state_topic", "/sim/state/gimbal"))
        self.chassis_state_topic = str(kwargs.get("chassis_state_topic", "/sim/state/chassis"))
        self.chassis_speed = float(kwargs.get("chassis_speed", 0.4))
        self.chassis_yaw_speed = float(kwargs.get("chassis_yaw_speed", 1.0))
        self.ready_xyz_speed = float(kwargs.get("ready_xyz_speed", 0.02))
        self.key_gimbal_speed = float(kwargs.get("key_gimbal_speed", 1.0))
        self.blocked_chassis_regions = [
            {
                "x": tuple(region.get("x", ())),
                "y": tuple(region.get("y", ())),
            }
            for region in kwargs.get("blocked_chassis_regions", [])
        ]
        self.pitch_min = float(kwargs.get("pitch_min", -1.2))
        self.pitch_max = float(kwargs.get("pitch_max", 1.2))
        self.joint_names = dict(
            kwargs.get(
                "joint_names",
                {
                    "chassis_x": "chassis_x",
                    "chassis_y": "chassis_y",
                    "chassis_yaw": "chassis_yaw",
                    "camera_left_yaw": "second_camera_left_yaw",
                    "camera_left_pitch": "second_camera_left_pitch",
                    "camera_right_yaw": "second_camera_right_yaw",
                    "camera_right_pitch": "second_camera_right_pitch",
                },
            )
        )
        self.actuator_names = dict(
            kwargs.get(
                "actuator_names",
                {
                    "chassis_x": "chassis_x",
                    "chassis_y": "chassis_y",
                    "chassis_yaw": "chassis_yaw",
                    "camera_left_yaw": "cam_left_yaw",
                    "camera_left_pitch": "cam_left_pitch",
                    "camera_right_yaw": "cam_right_yaw",
                    "camera_right_pitch": "cam_right_pitch",
                },
            )
        )
        self.ros_subscriptions[self.input_state_topic] = (
            "arm_exchange_interfaces.msg.OperatorInputState"
        )
        self.ros_subscriptions[self.ctrl_gimbal_topic] = (
            "arm_exchange_interfaces.msg.ArmCtrlGimbalControlMsg"
        )
        self.ros_subscriptions[self.host_output_topic] = (
            "arm_exchange_interfaces.msg.ArmHost2MCUMsg"
        )

        self.control_authority = "operator"
        self.host_enabled = False
        self.operator_mode = "teleop"
        self.active_camera = "left"
        self.pressed: set[str] = set()
        self.input_state = OperatorInputState()
        self.previous_input_state = OperatorInputState()
        self.chassis_x = 0.0
        self.chassis_y = 0.0
        self.chassis_yaw = 0.0
        self.q_manual_deg = 0.0
        self.gimbal_by_camera = {
            "left": {"yaw": 0.0, "pitch": 0.0},
            "right": {"yaw": 0.0, "pitch": 0.0},
        }
        self.host_session_started = False
        self._last_step_time: float | None = None
        self._last_logged_status: tuple[str, str, str] | None = None

    def setup(self, context: PluginSetupContext):
        node = context.node
        self.mode_pub = node.create_publisher(String, self.mode_state_topic, 10)
        self.authority_pub = node.create_publisher(String, self.authority_state_topic, 10)
        self.active_camera_pub = node.create_publisher(String, self.active_camera_state_topic, 10)
        self.gimbal_pub = node.create_publisher(Vector3, self.gimbal_state_topic, 10)
        self.chassis_pub = node.create_publisher(Vector3, self.chassis_state_topic, 10)
        self.host_enabled_pub = node.create_publisher(Bool, self.host_enabled_topic, 10)
        self.ctrl_start_pub = node.create_publisher(ArmCtrlStartMsg, self.ctrl_start_topic, 10)
        self.ctrl_enter_pub = node.create_publisher(Empty, self.ctrl_enter_topic, 10)
        self.ctrl_advance_pub = node.create_publisher(ArmCtrlAdvanceMsg, self.ctrl_advance_topic, 10)
        self.ctrl_gimbal_pub = node.create_publisher(ArmCtrlGimbalControlMsg, self.ctrl_gimbal_topic, 10)
        self.ctrl_q_pub = node.create_publisher(ArmCtrlQMsg, self.ctrl_q_topic, 10)
        self.ctrl_xyz_pub = node.create_publisher(ArmCtrlXYZMsg, self.ctrl_xyz_topic, 10)
        self.ctrl_withdraw_pub = node.create_publisher(ArmCtrlWithdrawMsg, self.ctrl_withdraw_topic, 10)

    def on_compile_callback(self, context: PluginContext):
        model = context.model
        self.qpos_addr = {}
        self.qvel_addr = {}
        for logical_name, mujoco_name in self.joint_names.items():
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(mujoco_name))
            if joint_id < 0:
                raise ValueError(f"OperatorLogicPlugin missing joint: {mujoco_name}")
            self.qpos_addr[logical_name] = int(model.jnt_qposadr[joint_id])
            self.qvel_addr[logical_name] = int(model.jnt_dofadr[joint_id])
        self.ctrl_addr = {}
        for logical_name, mujoco_name in self.actuator_names.items():
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, str(mujoco_name))
            if actuator_id < 0:
                raise ValueError(f"OperatorLogicPlugin missing actuator: {mujoco_name}")
            self.ctrl_addr[logical_name] = int(actuator_id)
        self._read_initial_state(context)
        self._publish_state(context)
        self._log_status_if_changed(context)

    def on_message_callback(self, context: PluginContext, topic: str, msg):
        if topic == self.input_state_topic:
            self._handle_input_state(context, msg)
        elif topic == self.ctrl_gimbal_topic:
            self._handle_gimbal_control(msg)
        elif topic == self.host_output_topic:
            self._handle_host_output(context, msg)
        else:
            raise ValueError(f"OperatorLogicPlugin received unexpected topic: {topic}")
        self._apply_state(context)
        self._publish_state(context)

    def on_step_callback(self, context: PluginContext):
        now = float(context.data.time)
        if self._last_step_time is None:
            self._last_step_time = now
            return
        dt = max(0.0, now - self._last_step_time)
        self._last_step_time = now
        self._publish_gimbal_control_from_keys(context, dt)
        if self.host_enabled:
            self._publish_ready_xyz_from_keys(context, dt)
        if self.control_authority == "operator" and self.operator_mode == "teleop":
            self._integrate_chassis(dt)
        self._apply_state(context)

    def on_timer_callback(self, context: PluginContext, alias: str):
        self._publish_state(context)

    def _read_initial_state(self, context: PluginContext) -> None:
        data = context.data
        self.chassis_x = float(data.qpos[self.qpos_addr["chassis_x"]])
        self.chassis_y = float(data.qpos[self.qpos_addr["chassis_y"]])
        self.chassis_yaw = float(data.qpos[self.qpos_addr["chassis_yaw"]])
        for camera in self.CAMERAS:
            self.gimbal_by_camera[camera]["yaw"] = float(
                data.qpos[self.qpos_addr[f"camera_{camera}_yaw"]]
            )
            self.gimbal_by_camera[camera]["pitch"] = float(
                data.qpos[self.qpos_addr[f"camera_{camera}_pitch"]]
            )

    def _handle_input_state(self, context: PluginContext, msg: OperatorInputState) -> None:
        self.previous_input_state = self.input_state
        self.input_state = msg
        self.pressed = {
            key
            for key in ("w", "a", "s", "d", "q", "e", "h", "j", "k", "l")
            if bool(getattr(msg, key))
        }
        self._handle_input_edges(context, msg, self.previous_input_state)
        self._log_status_if_changed(context)

    def _handle_input_edges(
        self,
        context: PluginContext,
        current: OperatorInputState,
        previous: OperatorInputState,
    ) -> None:
        if self._pressed_now(current, previous, "esc"):
            self._set_host_enabled(context, False)
            return
        if self._pressed_now(current, previous, "c"):
            self._cycle_camera()
            self._publish_gimbal_control(context)
            self._log_status_if_changed(context)
            return
        if self.host_enabled:
            self._publish_advance(context, bool(current.i))
            if self._pressed_now(current, previous, "enter"):
                self._publish_enter(context)
            if self._pressed_now(current, previous, "key_4"):
                self._publish_withdraw(context, ArmCtrlWithdrawMsg.MODE_EXTRACT_FOR_RETRY)
            if self._pressed_now(current, previous, "key_5"):
                self._publish_withdraw(context, ArmCtrlWithdrawMsg.MODE_SIDE_RELEASE)
            if self._pressed_now(current, previous, "q"):
                self._publish_q_delta(context, -1)
            if self._pressed_now(current, previous, "e"):
                self._publish_q_delta(context, 1)
            return
        if (
            self._pressed_now(current, previous, "key_1")
            or self._pressed_now(current, previous, "key_2")
            or self._pressed_now(current, previous, "key_3")
        ):
            self.operator_mode = "teleop"
            if self._pressed_now(current, previous, "key_1"):
                level = 1
            elif self._pressed_now(current, previous, "key_2"):
                level = 2
            else:
                level = 3
            self._set_host_enabled(context, True)
            self._publish_start(context, level)
        elif self._pressed_now(current, previous, "enter"):
            self._set_host_enabled(context, True)
            self._publish_start(context, 3)
        elif self._pressed_now(current, previous, "key_4"):
            self.operator_mode = "exchange_manual_q"
        elif self._pressed_now(current, previous, "tab"):
            self._cycle_mode()

    def _pressed_now(
        self,
        current: OperatorInputState,
        previous: OperatorInputState,
        field: str,
    ) -> bool:
        return bool(getattr(current, field)) and not bool(getattr(previous, field))

    def _set_host_enabled(self, context: PluginContext, enabled: bool) -> None:
        self.host_enabled = bool(enabled)
        self.control_authority = "host" if self.host_enabled else "operator"
        self.host_session_started = False
        if not self.host_enabled:
            self.operator_mode = "teleop"
            self.q_manual_deg = 0.0
        if self.host_enabled:
            self.pressed.clear()
        self.host_enabled_pub.publish(Bool(data=self.host_enabled))
        self._log_status_if_changed(context)

    def _handle_host_output(self, context: PluginContext, msg: ArmHost2MCUMsg) -> None:
        if not self.host_enabled:
            return
        if int(msg.host_state) == ArmHost2MCUMsg.HOST_IDLE:
            if self.host_session_started:
                self._set_host_enabled(context, False)
            return
        self.host_session_started = True

    def _publish_start(self, context: PluginContext, level: int) -> None:
        msg = ArmCtrlStartMsg()
        msg.header.stamp = context.node.get_clock().now().to_msg()
        msg.level = int(level)
        self.ctrl_start_pub.publish(msg)

    def _publish_enter(self, context: PluginContext) -> None:
        self.ctrl_enter_pub.publish(Empty())

    def _publish_advance(self, context: PluginContext, active: bool) -> None:
        msg = ArmCtrlAdvanceMsg()
        msg.header.stamp = context.node.get_clock().now().to_msg()
        msg.active = bool(active)
        self.ctrl_advance_pub.publish(msg)

    def _publish_q_delta(self, context: PluginContext, direction: int) -> None:
        self.q_manual_deg = min(90.0, max(-90.0, self.q_manual_deg + float(direction)))
        msg = ArmCtrlQMsg()
        msg.header.stamp = context.node.get_clock().now().to_msg()
        msg.mode = ArmCtrlQMsg.MODE_TARGET
        msg.q_deg10 = int(round(self.q_manual_deg * 10.0))
        self.ctrl_q_pub.publish(msg)

    def _publish_ready_xyz_from_keys(self, context: PluginContext, dt: float) -> None:
        dx = float("w" in self.pressed) - float("s" in self.pressed)
        dy = float("a" in self.pressed) - float("d" in self.pressed)
        dz = float("q" in self.pressed) - float("e" in self.pressed)
        if dx == 0.0 and dy == 0.0 and dz == 0.0:
            return
        scale = self.ready_xyz_speed * float(dt)
        msg = ArmCtrlXYZMsg()
        msg.header.stamp = context.node.get_clock().now().to_msg()
        msg.dx_m = float(dx * scale)
        msg.dy_m = float(dy * scale)
        msg.dz_m = float(dz * scale)
        self.ctrl_xyz_pub.publish(msg)

    def _publish_withdraw(self, context: PluginContext, mode: int) -> None:
        msg = ArmCtrlWithdrawMsg()
        msg.header.stamp = context.node.get_clock().now().to_msg()
        msg.mode = int(mode)
        self.ctrl_withdraw_pub.publish(msg)

    def _cycle_mode(self) -> None:
        index = self.MODES.index(self.operator_mode)
        self.operator_mode = self.MODES[(index + 1) % len(self.MODES)]

    def _cycle_camera(self) -> None:
        index = self.CAMERAS.index(self.active_camera)
        self.active_camera = self.CAMERAS[(index + 1) % len(self.CAMERAS)]

    def _integrate_chassis(self, dt: float) -> None:
        old_x = self.chassis_x
        old_y = self.chassis_y
        forward = float("w" in self.pressed) - float("s" in self.pressed)
        left = float("a" in self.pressed) - float("d" in self.pressed)
        yaw = float("q" in self.pressed) - float("e" in self.pressed)
        vx = forward * self.chassis_speed
        vy = left * self.chassis_speed
        operator_yaw = self.chassis_yaw + math.pi
        cos_yaw = math.cos(operator_yaw)
        sin_yaw = math.sin(operator_yaw)
        self.chassis_x += (cos_yaw * vx - sin_yaw * vy) * dt
        self.chassis_y += (sin_yaw * vx + cos_yaw * vy) * dt
        if self._is_chassis_blocked(self.chassis_x, self.chassis_y):
            self.chassis_x = old_x
            self.chassis_y = old_y
        self.chassis_yaw = _wrap_pi(self.chassis_yaw + yaw * self.chassis_yaw_speed * dt)

    def _is_chassis_blocked(self, x: float, y: float) -> bool:
        for region in self.blocked_chassis_regions:
            x_range = region["x"]
            y_range = region["y"]
            if len(x_range) != 2 or len(y_range) != 2:
                continue
            if min(x_range) <= x <= max(x_range) and min(y_range) <= y <= max(y_range):
                return True
        return False

    def _publish_gimbal_control_from_keys(self, context: PluginContext, dt: float) -> None:
        yaw = float("l" in self.pressed) - float("h" in self.pressed)
        pitch = float("k" in self.pressed) - float("j" in self.pressed)
        if yaw == 0.0 and pitch == 0.0:
            return
        current = self.gimbal_by_camera[self.active_camera]
        target_yaw = _wrap_pi(current["yaw"] + yaw * self.key_gimbal_speed * dt)
        target_pitch = min(
            self.pitch_max,
            max(self.pitch_min, current["pitch"] + pitch * self.key_gimbal_speed * dt),
        )
        self._publish_gimbal_control(context, pitch=target_pitch, yaw=target_yaw)

    def _publish_gimbal_control(
        self,
        context: PluginContext,
        *,
        pitch: float | None = None,
        yaw: float | None = None,
    ) -> None:
        current = self.gimbal_by_camera[self.active_camera]
        current["pitch"] = min(
            self.pitch_max,
            max(self.pitch_min, float(current["pitch"] if pitch is None else pitch)),
        )
        current["yaw"] = _wrap_pi(float(current["yaw"] if yaw is None else yaw))
        msg = ArmCtrlGimbalControlMsg()
        msg.header.stamp = context.node.get_clock().now().to_msg()
        msg.camera = (
            ArmCtrlGimbalControlMsg.CAMERA_RIGHT
            if self.active_camera == "right"
            else ArmCtrlGimbalControlMsg.CAMERA_LEFT
        )
        msg.pitch = float(current["pitch"])
        msg.yaw = float(current["yaw"])
        self.ctrl_gimbal_pub.publish(msg)

    def _handle_gimbal_control(self, msg: ArmCtrlGimbalControlMsg) -> None:
        if int(msg.camera) == ArmCtrlGimbalControlMsg.CAMERA_RIGHT:
            self.active_camera = "right"
        else:
            self.active_camera = "left"
        self.gimbal_by_camera[self.active_camera]["yaw"] = _wrap_pi(float(msg.yaw))
        self.gimbal_by_camera[self.active_camera]["pitch"] = min(
            self.pitch_max,
            max(self.pitch_min, float(msg.pitch)),
        )

    def _apply_state(self, context: PluginContext) -> None:
        data = context.data
        data.qpos[self.qpos_addr["chassis_x"]] = self.chassis_x
        data.qpos[self.qpos_addr["chassis_y"]] = self.chassis_y
        data.qpos[self.qpos_addr["chassis_yaw"]] = self.chassis_yaw
        for camera in self.CAMERAS:
            state = self.gimbal_by_camera[camera]
            data.qpos[self.qpos_addr[f"camera_{camera}_yaw"]] = state["yaw"]
            data.qpos[self.qpos_addr[f"camera_{camera}_pitch"]] = state["pitch"]
        data.ctrl[self.ctrl_addr["chassis_x"]] = self.chassis_x
        data.ctrl[self.ctrl_addr["chassis_y"]] = self.chassis_y
        data.ctrl[self.ctrl_addr["chassis_yaw"]] = self.chassis_yaw
        for camera in self.CAMERAS:
            state = self.gimbal_by_camera[camera]
            data.ctrl[self.ctrl_addr[f"camera_{camera}_yaw"]] = state["yaw"]
            data.ctrl[self.ctrl_addr[f"camera_{camera}_pitch"]] = state["pitch"]
        for name in self.qvel_addr:
            data.qvel[self.qvel_addr[name]] = 0.0
        mujoco.mj_forward(context.model, data)

    def _publish_state(self, context: PluginContext) -> None:
        self.mode_pub.publish(String(data=self.operator_mode))
        self.authority_pub.publish(String(data=self.control_authority))
        self.active_camera_pub.publish(String(data=self.active_camera))
        current = self.gimbal_by_camera[self.active_camera]
        self.gimbal_pub.publish(Vector3(x=current["yaw"], y=current["pitch"], z=0.0))
        self.chassis_pub.publish(Vector3(x=self.chassis_x, y=self.chassis_y, z=self.chassis_yaw))

    def _log_status_if_changed(self, context: PluginContext) -> None:
        status = (self.control_authority, self.operator_mode, self.active_camera)
        if status == self._last_logged_status:
            return
        self._last_logged_status = status
        context.node.get_logger().info(
            "operator control "
            f"authority={self.control_authority} "
            f"mode={self.operator_mode} "
            f"camera={self.active_camera}"
        )
