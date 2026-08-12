# Arm Exchange Host

Host-side ROS 2 nodes for the public MuJoCo integration workflow.

Nodes:

- `perception_node`: camera image + camera info -> `/host/perception/exchange_station_pose`
- `planning_node`: `/host/planning/type3_request` -> `/host/planning/type3_result`
- `task_node`: MCU control frames + perception + planning result -> `/host/arm/host_output`

The package is intended to run with the public MuJoCo backend. Numerical
perception, kinematics, collision checking and planning are provided by the
workspace package `arm_exchange_core`.

Build and source the public workspace before launching:

```bash
cd /path/to/public_archive
runtime/scripts/build.sh
source runtime/scripts/setup_env.sh
ros2 launch arm_exchange_host sim_host.launch.py
```

Run the package and data-contract tests from the same workspace with
`runtime/scripts/test.sh`.

The same launch file can also start the Linux event-device operator input node
when a keyboard device is available:

```bash
ros2 launch arm_exchange_host sim_host.launch.py \
  enable_operator_input:=true \
  keyboard_device:=/dev/input/by-path/platform-i8042-serio-0-event-kbd
```

Operator input is disabled by default because the event-device paths are
machine-specific. Mouse input is optional; pass `mouse_device:=/dev/input/...`
only when a mouse event device is available.

The camera input uses the standard `sensor_msgs/Image` message. No
`shm-tools`, `shm_msgs`, or shared-memory bridge is required.

`hpp-fcl` / `cmeel-boost` currently require the NumPy 1.26 ABI, while newer
OpenCV wheels may require NumPy 2.x. Keep NumPy and OpenCV pinned together
unless `hpp-fcl` is rebuilt against NumPy 2.

The Host runtime uses exported OpenVINO models for YOLO detection and
LiteHRNet keypoint inference. The default configuration runs YOLO on the
OpenVINO `GPU` device and LiteHRNet on `CPU`; these are independent OpenVINO
device assignments and do not use CUDA device strings.

Runtime settings are machine-local. Initialize them before the first build:

```bash
cp runtime/core/arm_exchange_core/system_config.example.yaml \
  runtime/core/arm_exchange_core/system_config.yaml
```

The copied `system_config.yaml` is ignored by Git. Repository-relative model
paths and deployment-specific OpenVINO devices can be changed there without
modifying the tracked template.

`perception_node` consumes the MuJoCo camera image and `CameraInfo`, then can
publish an OpenCV overlay with YOLO boxes, keypoints, and perception status.
