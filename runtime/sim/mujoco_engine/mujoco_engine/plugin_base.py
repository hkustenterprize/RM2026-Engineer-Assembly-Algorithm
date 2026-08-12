import mujoco
from typing import Any, Dict


class PluginContext:
    def __init__(self,
                 model: mujoco.MjModel,
                 data: mujoco.MjData,
                 spec: mujoco.MjSpec,
                 node: Any,
                 ):
        self._model = model
        self._data = data
        self._node = node
        self._spec = spec

    @property
    def model(self) -> mujoco.MjModel: return self._model
    @property
    def data(self) -> mujoco.MjData: return self._data
    @property
    def node(self) -> Any: return self._node
    @property
    def spec(self) -> mujoco.MjSpec: return self._spec


class PluginSetupContext:
    def __init__(self, spec: mujoco.MjSpec, node: Any):
        self._spec = spec
        self._node = node

    @property
    def spec(self) -> mujoco.MjSpec: return self._spec
    @property
    def node(self) -> Any: return self._node


class BasePlugin:
    def __init__(self, **kwargs):
        """
        Base class for all simulation plugins.

        Args:
            **kwargs: Configuration overrides from YAML.
                      'ros_subscriptions': Dict[topic_name, msg_type]
                      'ros_events': Dict[alias, hz]
        """
        # Default ROS inputs and timers (can be defined by subclasses)
        self.ros_subscriptions: Dict[str, Any] = {}
        self.ros_events: Dict[str, float] = {}

        # Merge configuration overrides
        if 'ros_subscriptions' in kwargs:
            self.ros_subscriptions.update(kwargs['ros_subscriptions'])
        if 'ros_events' in kwargs:
            self.ros_events.update(kwargs['ros_events'])

    def setup(self, context: PluginSetupContext):
        """
        Called before MjModel compilation. Use this to modify the MjSpec (e.g., attach sub-models).
        """
        pass

    def on_compile_callback(self, context: PluginContext):
        """
        Called after model compilation and before physics starts.
        Use this to get IDs (mj_name2id) or pre-allocate resources.
        """
        pass

    def on_step_callback(self, context: PluginContext):
        """
        Called before every simulation step. Use this for controls that should
        affect the upcoming MuJoCo integration.
        (Note: This callback should be lightweight to avoid slowing down the simulation.)
        """
        pass

    def on_message_callback(self, context: PluginContext, topic: str, msg: Any):
        """
        Callback for ROS 2 subscriptions. Topic serves as the logical ID.
        """
        pass

    def on_timer_callback(self, context: PluginContext, alias: str):
        """
        Callback for timers. Alias serves as the logical ID.
        """
        pass
