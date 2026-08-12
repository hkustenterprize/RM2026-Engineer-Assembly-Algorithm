import threading
import time
import yaml
import numpy as np
import rclpy
import cv2
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import CameraInfo, Image
import mujoco
from readerwriterlock import rwlock
from mujoco_engine.factory import (
    ModelBuilder,
    import_and_get,
    make_object_from_config,
    resolve_path
)
from mujoco_engine.plugin_base import PluginContext, PluginSetupContext
from mujoco_engine.glfw_ import GLFW, Camera


class RosImagePublisher:
    """Publish rendered camera frames using the standard ROS image message."""

    def __init__(self, node, topic):
        self.node = node
        self.publisher = node.create_publisher(
            Image,
            topic,
            rclpy.qos.qos_profile_sensor_data,
        )
        self._buffer = None
        self._first_frame_sent = False

    def get_buffer(self, height, width):
        self._buffer = np.empty((height, width), dtype=np.uint8)
        return self._buffer

    def send(self, stamp, encoding, frame_id):
        if self._buffer is None:
            return
        height, width = self._buffer.shape
        message = Image()
        message.header.stamp = stamp
        message.header.frame_id = frame_id
        message.height = height
        message.width = width
        message.encoding = encoding
        message.is_bigendian = 0
        message.step = width
        message.data = self._buffer.tobytes()
        self.publisher.publish(message)
        if not self._first_frame_sent:
            self.node.get_logger().info(
                f"Published first camera frame topic={self.publisher.topic_name} "
                f"size={width}x{height} encoding={encoding}"
            )
            self._first_frame_sent = True


def threaded_high_precision_loop(func):
    """
    Decorator to run a function in a high-precision threaded loop.
    Usage: @threaded_high_precision_loop, then call func(hz=1000, verbose=True)
    """

    def wrapper(self, hz, verbose=False, *args, **kwargs):
        interval = 1.0 / hz

        def loop():
            next_time = time.perf_counter()
            count = 0
            last_stat_time = time.perf_counter()
            while rclpy.ok() and not self.stop_event.is_set():
                next_time += interval
                try:
                    func(self, *args, **kwargs)
                except Exception:
                    if not rclpy.ok() or self.stop_event.is_set():
                        break
                    raise
                if verbose:
                    count += 1
                    if count >= 1000:
                        now = time.perf_counter()
                        duration = now - last_stat_time
                        actual_hz = count / duration
                        self.get_logger().info(
                            f"Loop '{func.__name__}': {actual_hz:.2f} Hz"
                        )
                        if actual_hz < hz * 0.950:
                            self.get_logger().warn(
                                f"Loop '{func.__name__}' is running slower than expected! This might cause a mismatch in simulation timing.")
                        count = 0
                        last_stat_time = now
                while time.perf_counter() < next_time:
                    time.sleep(0)
        t = threading.Thread(target=loop, daemon=True)
        t.start()
        return t
    return wrapper


class MujocoCameraMessageManager(Camera):

    def __init__(self,
                 publisher,
                 frame_id="camera_optical_frame",
                 render_scale=1.0,
                 **kwargs
                 ):
        self.render_scale = render_scale
        self.target_width = kwargs.get('width', 1440)
        self.target_height = kwargs.get('height', 1080)

        if self.render_scale != 1.0:
            kwargs['width'] = int(self.target_width * self.render_scale)
            kwargs['height'] = int(self.target_height * self.render_scale)

        super().__init__(**kwargs)
        self.publisher = publisher
        self.frame_id = frame_id

        self.camera_info_msg = CameraInfo(
            width=self.target_width, height=self.target_height)
        self.camera_info_msg.header.frame_id = frame_id

        fovy = self.model.cam_fovy[self.cam.fixedcamid]
        f = (self.target_height / 2) / np.tan(np.deg2rad(fovy) / 2)
        cx, cy = self.target_width / 2.0, self.target_height / 2.0

        self.camera_info_msg.k = [f, 0.0, cx, 0.0, f, cy, 0.0, 0.0, 1.0]
        self.camera_info_msg.p = [f, 0.0, cx, 0.0,
                                  0.0, f, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.camera_info_msg.d = [0.0] * 5

        self.upscale_rgb = None
        if self.render_scale != 1.0:
            self.upscale_rgb = np.empty(
                (self.target_height, self.target_width, 3), dtype=np.uint8)

    def update_and_get_camera_info_msg(self, stamp):
        self.camera_info_msg.header.stamp = stamp
        return self.camera_info_msg

    def process(self, glfw, data, stamp):
        glfw.render_cameras([self], data)
        view = self.publisher.get_buffer(self.target_height, self.target_width)
        if view is not None:
            if self.render_scale != 1.0:
                cv2.resize(self.out, (self.target_width, self.target_height),
                           dst=self.upscale_rgb, interpolation=cv2.INTER_LINEAR)
                src = self.upscale_rgb
            else:
                src = self.out
            view[0::2, 0::2] = src[0::2, 0::2, 0]  # R
            view[0::2, 1::2] = src[0::2, 1::2, 1]  # G
            view[1::2, 0::2] = src[1::2, 0::2, 1]  # G
            view[1::2, 1::2] = src[1::2, 1::2, 2]  # B
            self.publisher.send(stamp, "bayer_rggb8", self.frame_id)


class MujocoSimulator(Node):
    def __init__(self):
        super().__init__('mujoco_simulator')

        c_path = self.declare_parameter('config_path', '').value
        if not c_path:
            self.get_logger().error("model_path and config_path must be provided!")
            return

        with open(c_path, 'r') as f:
            self.config = yaml.safe_load(f)

        build_kwargs = self.config.get("build", {})
        self.builder = ModelBuilder(
            path=resolve_path(build_kwargs.get("base_scene", {}).get("path", "")),
        )
        for asset_kwargs in build_kwargs.get("assets", []):
            if "xml_path" in asset_kwargs:
                asset_kwargs["xml_path"] = resolve_path(asset_kwargs["xml_path"])
            self.builder.attach(
                **asset_kwargs
            )

        plugin_setup_context = PluginSetupContext(self.builder.spec, self)
        self.plugins = []
        for p_cfg in self.config.get('plugins', []):
            plugin = make_object_from_config(p_cfg)
            plugin.setup(plugin_setup_context)
            self.plugins.append(plugin)

        self.model = self.builder.compile()
        self.get_logger().info(
            f"Model loaded with {self.model.nbody} bodies"
        )

        self.data = mujoco.MjData(self.model)
        self._apply_initial_state()

        self.data_render = mujoco.MjData(self.model)
        self.data_sensor = mujoco.MjData(self.model)
        self.lock = rwlock.RWLockFair()

        self.plugin_context = PluginContext(
            self.model,
            self.data,
            self.builder.spec,
            self
        )
        for plugin in self.plugins:
            plugin.on_compile_callback(self.plugin_context)

            # Setup Subscriptions (Topic as ID)
            for topic, msg_type in plugin.ros_subscriptions.items():
                if isinstance(msg_type, str):
                    msg_type = import_and_get(msg_type)
                self.create_subscription(
                    msg_type,
                    topic,
                    lambda msg, p=plugin,
                    t=topic: self._plugin_msg_cb(
                        p, t, msg
                    ),
                    rclpy.qos.qos_profile_sensor_data
                )

            for alias, hz in plugin.ros_events.items():
                self.create_timer(
                    1.0/float(hz),
                    lambda p=plugin, a=alias: self._plugin_timer_cb(p, a)
                )
        self.get_logger().info(
            f"Initialized {len(self.plugins)} plugins."
        )

        self._setup_camera()
        self.stop_event = threading.Event()
        physics_cfg = self.config.get('physics', {})
        self.physics_thread = self._physics_step(
            hz=1.0 / self.model.opt.timestep, **physics_cfg)

    def _plugin_msg_cb(self, plugin, topic, msg):
        with self.lock.gen_wlock():
            plugin.on_message_callback(self.plugin_context, topic, msg)

    def _plugin_timer_cb(self, plugin, alias):
        with self.lock.gen_wlock():
            plugin.on_timer_callback(self.plugin_context, alias)

    def _setup_camera(self):
        """
        Configure cameras and visualization based on YAML config.
        """
        # 1. Camera Initialization
        cam_list = self.config.get('cameras', [])

        self.camera_entries = []
        self.camera_manager = None
        if isinstance(cam_list, list):
            for cfg in cam_list:
                if not cfg or not cfg.get('enabled', True):
                    continue
                pub = RosImagePublisher(self, cfg['topic'])
                info_pub = self.create_publisher(
                    CameraInfo,
                    cfg["info_topic"],
                    rclpy.qos.qos_profile_sensor_data,
                )
                manager = MujocoCameraMessageManager(
                    model=self.model, publisher=pub, **cfg)
                self.camera_entries.append({
                    'manager': manager,
                    'info_pub': info_pub,
                    'int': 1.0 / cfg.get('fps', 60),
                    'next': time.perf_counter(),
                })
                self.get_logger().info(
                    f"Camera enabled name={cfg.get('name')} topic={cfg.get('topic')} frame_id={cfg.get('frame_id')}")

        if self.camera_entries:
            # Backward-compatible alias for older code paths and single-camera configs.
            self.camera_manager = self.camera_entries[0]['manager']
        else:
            self.get_logger().info("No camera configured or disabled.")

        # 2. Visualization Config
        self.vis_cfg = self.config.get('visualization', {})
        self.vis_cfg.setdefault('enable_visualization', False)
        self.vis_int, self.last_vis = 1.0 / self.vis_cfg.get('fps', 30), 0.0

    def _apply_initial_state(self):
        """
        Apply initial states (keyframe, ctrl, qpos, qvel) from config to mjData.
        Order: Keyframe -> YAML Overwrites (ctrl, qpos, qvel) -> mj_forward
        """
        init_cfg = self.config.get('initialization', {})
        if not init_cfg:
            return

        # 1. Apply keyframe if specified
        key_id = init_cfg.get('keyframe', -1)
        if key_id != -1:
            if key_id < self.model.nkey:
                mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
                self.get_logger().info(
                    f"Applied keyframe {key_id} at startup.")
            else:
                self.get_logger().warn(
                    f"Requested keyframe {key_id}, but model has only {self.model.nkey} keyframes.")

        # 2. Apply ctrl overwrites
        init_ctrl = init_cfg.get('ctrl', {})
        if init_ctrl:
            for name, value in init_ctrl.items():
                act_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                if act_id != -1:
                    self.data.ctrl[act_id] = float(value)
                    self.get_logger().info(
                        f"Initialized ctrl for '{name}' to {value}")
                else:
                    self.get_logger().warn(
                        f"Actuator '{name}' defined in initialization.ctrl not found.")

        # 3. Final sync state (Important for synchronizing xpos/xmat, etc.)
        mujoco.mj_forward(self.model, self.data)

    @threaded_high_precision_loop
    def _physics_step(self, **kwargs):
        with self.lock.gen_wlock():
            for plugin in self.plugins:
                plugin.on_step_callback(self.plugin_context)
            mujoco.mj_step(self.model, self.data)

    def render_loop(self):
        cameras = [entry['manager'] for entry in self.camera_entries]
        first_camera_cycle = True
        with GLFW(
            self.model, self.data_render,
            cameras=cameras, **self.vis_cfg
        ) as glfw:
            tasks = {
                'vis': {'int': self.vis_int, 'next': time.perf_counter()}
            }

            rate = self.create_rate(200)

            while rclpy.ok() and not self.stop_event.is_set():
                now = time.perf_counter()
                due_camera_entries = [
                    entry for entry in self.camera_entries
                    if now >= entry['next']
                ]
                do_cam = bool(due_camera_entries)
                do_vis = now >= tasks['vis']['next']

                if do_cam or do_vis:
                    if first_camera_cycle and do_cam:
                        self.get_logger().info("Starting first camera render cycle")
                    with self.lock.gen_rlock():
                        self.data_render = self.data.__copy__()
                        snapshot_stamp = self.get_clock().now().to_msg()
                    if first_camera_cycle and do_cam:
                        self.get_logger().info("Captured first camera state snapshot")

                    for entry in due_camera_entries:
                        manager = entry['manager']
                        manager.process(glfw, self.data_render, snapshot_stamp)
                        entry['info_pub'].publish(
                            manager.update_and_get_camera_info_msg(snapshot_stamp))
                        entry['next'] += entry['int']
                    if first_camera_cycle and do_cam:
                        first_camera_cycle = False

                    if do_vis:
                        if not glfw.render_window(self.data_render):
                            break
                        tasks['vis']['next'] += tasks['vis']['int']

                rate.sleep()

    def spin(self):
        while rclpy.ok() and not self.stop_event.is_set():
            try:
                rclpy.spin_once(self, timeout_sec=0.001)
            except ExternalShutdownException:
                break
            except Exception:
                if not rclpy.ok() or self.stop_event.is_set():
                    break
                raise


def main(args=None):
    rclpy.init(args=args)
    node = MujocoSimulator()
    spin_thread = threading.Thread(target=node.spin, daemon=True)
    spin_thread.start()
    try:
        node.render_loop()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stop_event.set()
        spin_thread.join(timeout=1.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
