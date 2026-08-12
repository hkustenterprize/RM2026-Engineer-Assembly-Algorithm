from __future__ import annotations

import mujoco
from arm_exchange_interfaces.msg import ExchangeStationState
from mujoco_engine.plugin_base import BasePlugin, PluginContext


_STATION_JOINTS = {
    "x": "level_x",
    "y": "level_y",
    "z": "level_z",
    "alpha": "level_alpha",
    "theta": "level_theta",
    "phi": "level_phi",
}
_STATION_ACTUATORS = {name: f"ctrl_{joint}" for name, joint in _STATION_JOINTS.items()}
_DEFAULT_STATE = {
    "x": -0.05,
    "y": 0.2,
    "z": 0.6,
    "alpha": 0.0,
    "theta": 0.0,
    "phi": 0.0,
}


class StationPlugin(BasePlugin):
    """Set the simulated exchange-station pose from a ROS message."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.topic = str(kwargs.get("topic", "/debug/scene/exchange_station/set_state"))
        self.initial_state = _DEFAULT_STATE | {
            name: float(value) for name, value in kwargs.get("initial_state", {}).items()
        }
        self.limits = {
            name: tuple(float(value) for value in bounds)
            for name, bounds in kwargs["limits"].items()
        }
        if self.limits.keys() != _STATION_JOINTS.keys():
            raise ValueError(f"station limits must define {tuple(_STATION_JOINTS)}")
        self.ros_subscriptions[self.topic] = "arm_exchange_interfaces.msg.ExchangeStationState"

    def on_compile_callback(self, context: PluginContext):
        self.qpos_addr = self._addresses(
            context.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            _STATION_JOINTS,
        )
        self.ctrl_addr = self._addresses(
            context.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            _STATION_ACTUATORS,
        )
        self._apply(context, self.initial_state)

    def on_message_callback(self, context: PluginContext, topic: str, msg):
        if topic != self.topic or not isinstance(msg, ExchangeStationState):
            raise ValueError(f"StationPlugin received invalid message on {topic}")
        self._apply(context, {name: float(getattr(msg, name)) for name in _STATION_JOINTS})

    @staticmethod
    def _addresses(model, object_type, names: dict[str, str]) -> dict[str, int]:
        addresses = {}
        for logical_name, mujoco_name in names.items():
            object_id = mujoco.mj_name2id(model, object_type, mujoco_name)
            if object_id < 0:
                raise ValueError(f"StationPlugin missing MuJoCo object: {mujoco_name}")
            addresses[logical_name] = int(
                model.jnt_qposadr[object_id]
                if object_type == mujoco.mjtObj.mjOBJ_JOINT
                else object_id
            )
        return addresses

    def _apply(self, context: PluginContext, values: dict[str, float]) -> None:
        for name, value in values.items():
            lower, upper = self.limits[name]
            if not lower <= value <= upper:
                raise ValueError(f"station {name}={value:.6f} outside [{lower:.6f}, {upper:.6f}]")
            context.data.qpos[self.qpos_addr[name]] = value
            context.data.ctrl[self.ctrl_addr[name]] = value
        mujoco.mj_forward(context.model, context.data)
