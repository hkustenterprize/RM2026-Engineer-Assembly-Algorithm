# Arm Exchange Core

`arm_exchange_core` contains the numerical models and algorithms shared by the ROS 2 Host and the MuJoCo application.
It deliberately contains no ROS nodes or ROS message conversions: callers pass NumPy arrays and plain configuration
mappings, while the packages at the runtime boundary own timestamps, frame lookup, topic semantics and task state.

The package is installed as an `ament_python` package so the same implementation is available from the Host and
simulation processes after the runtime workspace is built.

## Package Layout

```text
runtime/core/
├── arm_exchange_core/
│   ├── __init__.py              # Package-owned YAML configuration loader
│   ├── arm_model.py             # Batched FK, analytic IK and rigid-body dynamics
│   ├── joint_space.py           # Joint limits, periodic differences and penalties
│   ├── trajectory.py            # Fixed-duration quintic time parameterization
│   ├── transform.py             # Batched transform and quaternion utilities
│   ├── perception/
│   │   ├── detector.py          # OpenVINO YOLO + LiteHRNet inference
│   │   └── pose.py              # Keypoint PnP estimation
│   ├── planning/
│   │   ├── collision.py         # Capsule-to-mesh collision and distance bounds
│   │   ├── type2.py             # OMPL BIT* point-to-point planning
│   │   ├── type3.py             # Assembly-constrained layered planning
│   │   └── viterbi.py           # Layered shortest-path dynamic programming
│   ├── assets/collision/        # Package-owned station collision meshes
│   └── system_config.example.yaml
└── test/test_core_contracts.py  # Numerical and configuration contract tests
```

## Robot Model

`ArmModel` is the common six-axis robot model. All public kinematics and dynamics methods use an explicit batch
dimension; a single query therefore has shape `(1, ...)` rather than a separate scalar API.

| Operation | Input | Output |
| --- | --- | --- |
| Forward kinematics | joints `(B, 6)` | seven link transforms `(B, 7, 4, 4)` and TCP transforms `(B, 4, 4)` |
| Analytic inverse kinematics | TCP transforms `(B, 4, 4)` | eight branches `(B, 8, 6)` and validity mask `(B, 8)` |
| Inverse dynamics | position, velocity and acceleration `(B, 6)` | joint torque `(B, 6)` |
| External TCP wrench | wrench `[force, torque]` `(B, 6)` | equivalent joint torque `(B, 6)` |

The configured tool offset distinguishes DH frame 6 from the true tool center point (TCP). `JointSpace` centralizes
joint-limit checks and the difference convention: unlimited continuous joints are compared modulo 2 pi, while
bounded joints use their direct coordinate difference. Planners and controllers therefore share the same continuity
and limit semantics.

The inverse-dynamics and external-wrench methods support the simulated lower controller. The latter propagates a TCP
wrench through the serial chain and produces the virtual-work-equivalent joint torque. Its use by the reserved force
feedforward path is described in the [simulation guide](../sim/arm_exchange_sim/README.md#force-feedforward-research-interface).

## Perception

`YoloHRNetBackend` implements the published top-down perception path. An OpenVINO YOLO model first detects candidate
boxes, each selected crop is normalized for an OpenVINO LiteHRNet model, and the resulting keypoints are mapped back to
the source image. The backend returns `KeypointObservation`, a numerical observation carrying ordered image points,
confidence values and detection metadata.

`PnPEstimator` matches those image points to the configured three-dimensional schema. It uses RANSAC with SQPnP to
select an initial solution, rejects solutions with invalid camera depth, and applies Levenberg-Marquardt refinement.
The result is the camera-from-object homogeneous transform and its mean reprojection error. Camera calibration and TF
composition remain responsibilities of the Host perception node.

## Planning

The planning modules correspond to the Type II and Type III definitions in the technical report:

- `plan_joint_path_bitstar()` wraps OMPL BIT* for a bounded six-dimensional joint-space query. The caller supplies the
  state-validity function, so collision ownership remains outside the OMPL adapter.
- `Type3Planner` samples the prescribed assembly path and roll domain, evaluates one configured analytic IK branch,
  applies collision and joint-limit costs, and solves the resulting layered graph with `solve_viterbi()`.
- When the strict Type III graph has no complete valid path, the optional best-effort pass assigns a high cost to
  invalid nodes and repairs only the selected invalid nodes with bounded Trust Region Reflective least squares. Every
  repaired path is checked for collision again before it can be returned.
- Terminal continuation checks allow a P-axis path to account for the subsequent Q-axis sweep when selecting its final
  roll state.

`CollisionChecker` represents arm links as configured capsules and the station as package-owned triangle meshes. It
uses hpp-fcl for collision queries and its conservative distance lower bound for soft clearance penalties. Collision
assets are resolved through Python package resources; callers do not supply repository-relative mesh paths.

`FixedDurationParameterizer` converts sparse waypoints into a sampled quintic joint trajectory. It supplies position,
velocity and acceleration arrays with a shared timestamp vector; execution and ROS message publication remain in the
Host layer.

## Configuration

Create the ignored machine-local configuration from the tracked template:

```bash
cp runtime/core/arm_exchange_core/system_config.example.yaml \
  runtime/core/arm_exchange_core/system_config.yaml
```

`load_config()` reads this package-owned file by default and accepts an explicit path for tests or external tooling.
The main sections are:

| Section | Ownership |
| --- | --- |
| `arm` | DH geometry, tool offset, joint limits and dynamic parameters |
| `collision` | Arm capsules and collision-asset selection |
| `perception` | OpenVINO models, devices, keypoint schema and PnP settings |
| `planning.exchange` | Approach, Type III graph, recovery and trajectory settings |
| `planning.type3` | Physical geometry of the assembly constraint |

Model paths and OpenVINO device choices are deployment-specific. Geometry, joint conventions and keypoint ordering
must remain consistent across perception, planning and simulation.

## Runtime Boundary

The core package does not subscribe or publish. The Host performs the narrow conversions required at its boundaries:

- `sensor_msgs/Image` and `CameraInfo` become image arrays and camera matrices for perception.
- `geometry_msgs/PoseStamped` becomes a homogeneous transform for planning.
- `sensor_msgs/JointState` becomes a `(1, 6)` joint batch for numerical operations.
- Planned NumPy trajectories become the project-specific Host/controller messages used by the task layer.

This keeps numerical code testable without a ROS graph and avoids maintaining a second message-like application data
model inside the algorithms.

## Validation

Build the workspace first, then run the complete runtime suite from the repository root:

```bash
runtime/scripts/build.sh
runtime/scripts/test.sh
```

The core contract tests cover PnP reconstruction, detector/configuration compatibility, batched FK/IK and dynamics,
collision assets and batch behavior, Type III frame composition, and the P-to-Q terminal continuation constraint.
