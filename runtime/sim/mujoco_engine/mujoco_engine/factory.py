import importlib
import os
from typing import Any

import mujoco
import numpy as np
from ament_index_python.packages import get_package_share_directory


# =============================================================================
# Configuration & Object Instantiation
# =============================================================================

def resolve_path(path: str) -> str:
    """Resolve an absolute path or a ROS package resource URI."""
    if not path:
        return ""

    if path.startswith("package://"):
        pkg_part = path[10:]
        if "/" not in pkg_part:
            return get_package_share_directory(pkg_part)
        pkg_name, relative_path = pkg_part.split("/", 1)
        return os.path.join(get_package_share_directory(pkg_name), relative_path)

    if not os.path.isabs(path):
        raise ValueError(f"path must be absolute or use package://, got {path!r}")
    return path


def import_and_get(class_path: str):
    """
    Dynamically import a module and get a class or function from it.
    Example: "sensor_msgs.msg.Imu" -> returns the Imu class.
    """
    try:
        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError, ValueError) as e:
        raise ImportError(f"Failed to import {class_path}: {e}")


def make_object(class_path: str, args: dict[str, Any] | None = None):
    """Instantiate an object from a class path and arguments."""
    _class = import_and_get(class_path)
    return _class() if args is None else _class(**args)


def make_object_from_config(config: Any) -> Any:
    """
    Recursively parse a configuration dictionary to instantiate objects or functions.
    Supports:
    - _function_name: "module.func" -> returns the function object.
    - _class_name: "module.Class" -> instantiates the class with remaining keys as kwargs.
    """
    if isinstance(config, list):
        return [make_object_from_config(i) for i in config]
    if isinstance(config, dict):
        if "_function_name" in config:
            return import_and_get(config["_function_name"])
        if "_class_name" in config:
            class_path = config["_class_name"]
            kwargs = {
                key: make_object_from_config(value)
                for key, value in config.items()
                if key != "_class_name"
            }
            return make_object(class_path, kwargs)
        return {key: make_object_from_config(value) for key, value in config.items()}
    return config


class MujocoDataHandler:

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

        self.extractors = {
            "sensor": (mujoco.mjtObj.mjOBJ_SENSOR, lambda id: data.sensordata[model.sensor_adr[id]: model.sensor_adr[id] + model.sensor_dim[id]]),
            "qpos": (mujoco.mjtObj.mjOBJ_JOINT, lambda id: data.qpos[model.jnt_qposadr[id]]),
            "qvel": (mujoco.mjtObj.mjOBJ_JOINT, lambda id: data.qvel[model.jnt_dofadr[id]]),
            "joint": (mujoco.mjtObj.mjOBJ_JOINT, lambda id: data.qpos[model.jnt_qposadr[id]]),
            "actuator": (mujoco.mjtObj.mjOBJ_ACTUATOR, lambda id: data.ctrl[id]),
            "site": (mujoco.mjtObj.mjOBJ_SITE, lambda id: data.site_xpos[id]),
            "site_xpos": (mujoco.mjtObj.mjOBJ_SITE, lambda id: data.site_xpos[id]),
            "site_xmat": (mujoco.mjtObj.mjOBJ_SITE, lambda id: data.site_xmat[id]),
            "site_xvel": (mujoco.mjtObj.mjOBJ_SITE, self._get_site_xvel),
            "body": (mujoco.mjtObj.mjOBJ_BODY, lambda id: data.xpos[id]),
            "body_xpos": (mujoco.mjtObj.mjOBJ_BODY, lambda id: data.xpos[id]),
            "body_xquat": (mujoco.mjtObj.mjOBJ_BODY, lambda id: data.xquat[id]),
            "geom": (mujoco.mjtObj.mjOBJ_GEOM, lambda id: data.geom_xpos[id]),
            "geom_xmat": (mujoco.mjtObj.mjOBJ_GEOM, lambda id: data.geom_xmat[id]),
            "geom_xpos": (mujoco.mjtObj.mjOBJ_GEOM, lambda id: data.geom_xpos[id]),
            "camera": (mujoco.mjtObj.mjOBJ_CAMERA, lambda id: None),
        }

    def _get_site_xvel(self, id: int):
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, id, vel, 0)
        return vel

    def get_value_and_id(self, type: str, name: str):
        if type == "time":
            return self.data.time, -1

        if type not in self.extractors:
            return None, -1

        obj_type, func = self.extractors[type]
        id = mujoco.mj_name2id(self.model, obj_type, name)
        if id == -1:
            return None, id
        return func(id), id

    def get_id(self, type: str, name: str) -> int:
        if type not in self.extractors:
            return -1
        obj_type, _ = self.extractors[type]
        return mujoco.mj_name2id(self.model, obj_type, name)

    def get_joint_qpos_adr(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return self.model.jnt_qposadr[joint_id] if joint_id != -1 else -1

    def get_joint_qvel_adr(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return self.model.jnt_dofadr[joint_id] if joint_id != -1 else -1

    def get_value_by_id(self, type: str, id: int):
        if type == "time":
            return self.data.time

        if type not in self.extractors:
            return None

        _, func = self.extractors[type]
        return func(id)


class ModelBuilder:
    def __init__(self, path: str):
        self.spec: mujoco.MjSpec = mujoco.MjSpec.from_file(path)

    def attach(
        self,
        xml_path: str,
        mount_type: str,
        mount_name: str,
        prefix: str = "",
        pos=None,
        quat=None,
        count: int = 1
    ):
        spec_template = mujoco.MjSpec.from_file(xml_path)
        mount = self.spec.site(mount_name) if mount_type == "site" else \
            self.spec.frame(mount_name) if mount_type == "frame" else None
        if mount is None:
            raise ValueError(
                f'No valid mount or parent found with name: {mount_name}')

        for i in range(count):
            prefix = prefix.rstrip("/")
            instance_prefix = f"{prefix}/{i}/" if count > 1 else f"{prefix}/"
            child_spec = spec_template.copy()
            child_spec.copy_during_attach = True
            if isinstance(mount, mujoco.MjsSite):
                frame = self.spec.attach(
                    child_spec, prefix=instance_prefix, site=mount)
            elif isinstance(mount, mujoco.MjsFrame):
                frame = self.spec.attach(
                    child_spec, prefix=instance_prefix, frame=mount)
            if pos:
                frame.pos = pos
            if quat:
                frame.quat = quat
        return self.spec

    def compile(self) -> mujoco.MjModel:
        return self.spec.compile()
