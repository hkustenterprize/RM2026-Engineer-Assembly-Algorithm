# RM2026 Engineer Assembly Algorithm

Open-source engineering assembly algorithms developed by HKUST ENTERPRIZE for the RoboMaster 2026 season.
The project supports a semi-automatic energy-unit assembly workflow: the upper computer provides visual estimation,
approach planning, constrained motion planning and recovery planning, while the operator performs insertion with a
custom controller.

## Overview

The public archive contains the following parts:

- Blender scene and synthetic-data generation.
- YOLO pose and object-detection training pipelines.
- MMPose LiteHRNet top-down keypoint training, inference and OpenVINO export.
- Dataset conversion and visualization utilities used by the vision pipelines.

The accompanying technical report also covers PnP pose estimation, motion planning, task orchestration and MuJoCo
simulation. A minimal MuJoCo and Host integration workspace is included for reproducible simulation-based testing.
The real backend is still being organized and is not part of the current release.

The technical report provides the algorithmic background and system design:

- [RM2026 Engineer Assembly Algorithm](doc/RM2026-Engineer-Assembly-Algorithm.pdf)

The public implementation is organized as follows:

```text
public_archive/
├── doc/                         # Technical report
├── src/
│   ├── blender/                 # Blender rendering and dataset composition
│   │   ├── exchange.blend       # Public Blender scene
│   │   ├── configs/             # Rendering configuration and field reference
│   │   ├── pipeline/            # Rendering, composition and visualization
│   │   └── setup_blender_env.sh
│   ├── nn/                      # Shared uv environment for vision models
│   │   ├── __main__.py          # Unified training, inference and export CLI
│   │   ├── pyproject.toml
│   │   ├── uv.lock
│   │   ├── setup_env.sh
│   │   ├── utils.py             # Configuration-driven object construction
│   │   ├── yolo/                # Ultralytics YOLO pose/detection pipeline
│   │   └── hrnet/               # MMPose LiteHRNet pipeline
├── runtime/
│   ├── interfaces/              # ROS 2 task and controller interfaces
│   ├── core/                    # Numerical perception and planning package
│   ├── sim/                     # Reusable MuJoCo engine and task simulation package
│   ├── host/                    # Perception, planning and task nodes
│   └── scripts/                 # Workspace build and environment setup
├── LICENSE
└── README.md
```

Detailed instructions are maintained next to each implementation:

- [Blender data generation](src/blender/README.md)
- [Shared neural-network environment](src/nn/README.md)
- [YOLO pose and detection](src/nn/yolo/README.md)
- [LiteHRNet](src/nn/hrnet/README.md)

## Environment

YOLO and LiteHRNet share the uv environment defined in `src/nn`:

```bash
cd public_archive
bash src/nn/setup_env.sh
source src/nn/.venv/bin/activate
rm26-nn check
```

Blender uses its bundled Python environment and is fixed to Blender 4.5.0. Install its rendering dependencies with:

```bash
cd public_archive/src/blender
bash setup_blender_env.sh
```

See the module READMEs for model-specific dependencies, dataset conversion and execution commands.

## MuJoCo and Host Integration

The public ROS 2 workspace uses standard `sensor_msgs/Image` messages for simulated camera frames and does not depend on
`shm-tools`, `shm_msgs`, or a shared-memory bridge. Create the machine-local configuration from the tracked template,
then build and source the workspace:

```bash
cd public_archive
cp runtime/core/arm_exchange_core/system_config.example.yaml \
  runtime/core/arm_exchange_core/system_config.yaml
# Adjust model paths and OpenVINO devices in system_config.yaml when needed.
runtime/scripts/build.sh
source runtime/scripts/setup_env.sh
ros2 launch arm_exchange_host sim_host.launch.py camera_view:=false
```

The default launch starts MuJoCo, the planning node and the task node. Model-based perception is optional because its
checkpoint files are released separately:

```bash
ros2 launch arm_exchange_host sim_host.launch.py enable_perception:=true
```

Install the Python dependencies listed in
[runtime/requirements-sim-host.txt](runtime/requirements-sim-host.txt) before enabling perception or running the
collision-aware planning path. The real hardware backend and real-device launch files are still being organized and
are not part of the current release.

Run the runtime contract and package tests with:

```bash
runtime/scripts/test.sh
```

## Dataset Release

The synthetic dataset is hosted on Hugging Face Datasets:
[hkustenterprize/RM26_engineer_exchange](https://huggingface.co/datasets/hkustenterprize/RM26_engineer_exchange).

| Split | Ratio | Samples |
| --- | ---: | ---: |
| Train | 90% | 40,860 |
| Val | 10% | 4,540 |
| Total | 100% | 45,400 |

The dataset provides YOLO pose labels for the `pillar` and `exchange` targets, including 12 keypoints. A detection
dataset can be derived from the same labels with the conversion command documented in the YOLO README.

## Checkpoint Release

Model checkpoints are hosted on Hugging Face:
[hkustenterprize/RM26_engineer_exchange_model](https://huggingface.co/hkustenterprize/RM26_engineer_exchange_model).

| Model | Task | Input size | Pipeline | Native checkpoint | OpenVINO export |
| --- | --- | --- | --- | --- | --- |
| LiteHRNet-30-v9.1 | 12-keypoint | 256x256 | top-down | [best.pth](https://huggingface.co/hkustenterprize/RM26_engineer_exchange_model/resolve/main/litehr/litehrnet30_v9.1/best.pth?download=true) | [openvino_best](https://huggingface.co/hkustenterprize/RM26_engineer_exchange_model/tree/main/litehr/litehrnet30_v9.1/openvino_best) |
| YOLO26-s-v10.12 | Box Detection | 640x640 | top-down | [best.pt](https://huggingface.co/hkustenterprize/RM26_engineer_exchange_model/resolve/main/yolo/v10.12/best.pt?download=true) | [openvino_best](https://huggingface.co/hkustenterprize/RM26_engineer_exchange_model/tree/main/yolo/v10.12/openvino_best) |
| YOLO26-s-pose-v11.02 | Box Detection + 12-keypoint | 640x640 | bottom-up | [best.pt](https://huggingface.co/hkustenterprize/RM26_engineer_exchange_model/resolve/main/yolopose/v11.02/best.pt?download=true) | [openvino_best](https://huggingface.co/hkustenterprize/RM26_engineer_exchange_model/tree/main/yolopose/v11.02/openvino_best) |

The native links are direct downloads. The OpenVINO links point to the exported model directories. All listed models
were trained with synthetic data from the dataset release; the top-down pipeline was used in the deployed workflow,
while the bottom-up checkpoint is provided as a reference and comparison implementation.

## Release and Repository Update Log

This repository is released incrementally. The log records public artifacts and major repository reorganizations.

### 2026-08-12

- Added the self-contained MuJoCo and Host ROS 2 workspace under `runtime/`.
- Packaged numerical perception and planning code as `arm_exchange_core`.
- Removed shared-memory camera transport and legacy runtime data abstractions.
- Added package-owned collision assets and simulation integration tests.

### 2026-08-11

- Published the Blender scene and aligned the fixed-camera rendering pipeline with Blender 4.5.0.
- Added an explicit `main()` entry point to `render_dataset.py` and a warning for exhausted framing retries.
- Consolidated the neural-network CLI and documented the shared uv environment.

### 2026-06-24

- Released the open-source technical report PDF.
- Added the public README and project overview.
- Added the Hugging Face dataset and checkpoint entries.

## License

This repository is released under the MIT License. See [LICENSE](LICENSE) for details.

The dataset and model checkpoints are hosted on Hugging Face. Please refer to their corresponding pages for
asset-specific license and usage notes.

## Acknowledgement

This project was developed by HKUST ENTERPRIZE for the RoboMaster 2026 season. The implementation builds on open-source
tools including MuJoCo, OMPL, hpp-fcl, Ultralytics, MMPose, SciPy and Albumentations.
