# Arm Exchange Host

Host-side ROS 2 nodes for the semi-automatic assembly workflow. Workspace setup, configuration and complete launch
commands are maintained in the [runtime guide](../README.md).

## Nodes

| Node | Responsibility | Principal output |
| --- | --- | --- |
| `perception_node` | YOLO detection, LiteHRNet keypoints, PnP and TF conversion | `/host/perception/exchange_station_pose` |
| `planning_node` | Type II approach, Type III constrained path and recovery planning | Planning result topics |
| `task_node` | Human-machine state machine, planning requests and trajectory playback | `/host/arm/host_output` |

The package is intended to run with the public MuJoCo backend. Numerical
perception, kinematics, collision checking and planning are provided by the
workspace package `arm_exchange_core`.

## Data Flow

The perception node consumes the active second-camera `sensor_msgs/Image` and `CameraInfo`, then publishes the
estimated assembly-station pose in the arm-base frame. The task node combines that pose with operator commands and the
current joint state. It asks the planning node for approach, constrained assembly or recovery trajectories and sends
the selected trajectory to the simulated lower controller.

Insertion remains manual. Once the operator confirms insertion, the task node reconstructs the five-degree-of-freedom
assembly reference from current FK and continues with the constrained automatic stages.

## Perception

The Host uses the exported YOLO and LiteHRNet OpenVINO models configured in the machine-local
`system_config.yaml`. YOLO and LiteHRNet have independent OpenVINO device fields. The node can also display an OpenCV
overlay containing all YOLO detections, selected keypoints, visibility scores and the resulting perception status.

The camera transport uses standard `sensor_msgs/Image`; no `shm-tools`, `shm_msgs` or shared-memory bridge is required.
Pose estimation uses the camera calibration from `CameraInfo` and the configured 12-point object schema.
RANSAC/SQPnP selects the initial PnP solution before Levenberg-Marquardt refinement.

## Planning and Task Orchestration

The planning node owns the robot model, collision environment and Type II/III planners. The task node owns workflow
state and is the only component that turns operator events into planning or execution requests. This separation keeps
the geometric planners independent of confirmation, manual insertion and recovery semantics.

The task node also publishes the disabled-by-default `/host/arm/feedforward_wrench` research interface. Its intended
force projection, simulator-side torque mapping and current validation boundary are documented in the
[simulation guide](../sim/arm_exchange_sim/README.md#force-feedforward-research-interface).

The public launch file starts all Host nodes required by the selected options. See the [runtime guide](../README.md)
for launch arguments and the [simulation guide](../sim/arm_exchange_sim/README.md) for operator controls and ROS topic
inspection.
