# MuJoCo Engine

Reusable ROS 2 execution engine for configuration-driven MuJoCo simulations.
It owns model assembly, plugin construction and lifecycle callbacks, physics
stepping, camera rendering, and the `simulator` executable.

Task-specific plugins, scenes, assets, and configuration belong to application
packages such as `arm_exchange_sim`. Plugins implement `BasePlugin` and are
instantiated from the application YAML through `make_object_from_config`.
