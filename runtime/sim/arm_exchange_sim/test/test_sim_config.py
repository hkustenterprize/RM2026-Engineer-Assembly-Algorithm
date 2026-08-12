from pathlib import Path

import numpy as np
import yaml
from mujoco_engine.factory import make_object_from_config

from arm_exchange_sim.camera_mount import ParallelMountKinematics


CONFIG_DIR = Path(__file__).parents[1] / "config"


def _load_config(filename: str):
    with (CONFIG_DIR / filename).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _load_plugins(filename: str):
    config = _load_config(filename)
    plugin_configs = config["plugins"]
    assert all("ros_subscriptions" not in item for item in plugin_configs)
    return [make_object_from_config(item) for item in plugin_configs]


def test_camera_config_contains_only_consumed_fields() -> None:
    allowed = {
        "name",
        "topic",
        "info_topic",
        "frame_id",
        "width",
        "height",
        "render_scale",
        "fps",
        "enabled",
    }
    assert all(
        set(camera) <= allowed
        for camera in _load_config("simulation_config.yaml")["cameras"]
    )


def test_simulation_plugins_can_be_constructed() -> None:
    plugins = _load_plugins("simulation_config.yaml")
    names = {type(plugin).__name__ for plugin in plugins}
    assert names == {
        "CameraMountPlugin",
        "ArmExchangeTfPlugin",
        "ArmPlugin",
        "OperatorLogicPlugin",
        "StationPlugin",
    }
    arm = next(plugin for plugin in plugins if type(plugin).__name__ == "ArmPlugin")
    assert set(arm.ros_subscriptions) == {
        "/host/arm/host_output",
        "/host/arm/feedforward_wrench",
        "/sim/mcu/arm_enabled",
    }
    assert arm.ros_events == {"joint_states": 100}


def test_parallel_mount_uses_arm_model_link_transforms() -> None:
    mount = ParallelMountKinematics.from_config()
    joints = np.asarray(
        [
            [0.0, 0.4, 0.0, 0.0, 0.0, 0.0],
            [0.3, 1.2, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    links = mount.arm.forward_kinematics(joints).link_transforms
    linkage = mount.t_arm_linkage(joints)

    expected_origins = links[:, 2, :3, 3] + np.einsum(
        "bij,j->bi",
        links[:, 2, :3, :3],
        np.asarray([mount.arm.a[2], 0.0, 0.0]),
    )
    np.testing.assert_allclose(linkage[:, :3, :3], links[:, 1, :3, :3])
    np.testing.assert_allclose(linkage[:, :3, 3], expected_origins)
