# RM2026 Engineer Assembly Algorithm

Open-source engineering assembly algorithms developed by HKUST ENTERPRIZE for the RoboMaster 2026 season.
The project supports a semi-automatic energy-unit assembly workflow: the upper computer provides visual estimation,
approach planning, constrained motion planning and recovery planning, while the operator performs insertion with a
custom controller.

## Project Contents

The repository publishes the synthetic-data pipeline, vision-model training code, numerical perception and planning
algorithms, and a self-contained ROS 2 + MuJoCo integration workspace. The current runtime release targets simulation;
the real-hardware backend is still being organized.

```text
./
├── doc/                  # Open-source technical report
├── src/blender/          # Rendering and synthetic-data generation
├── src/nn/               # YOLO and LiteHRNet training/export workspace
├── runtime/              # ROS 2 Host, planning core and MuJoCo simulation
├── LICENSE
└── README.md
```

All relative paths and commands in the documentation assume that the current working directory is the repository root.

## Environments

The runtime, neural-network training and Blender rendering workflows use separate Python environments. Their
dependencies should not be installed into one shared environment:

| Workflow | Environment | Setup guide |
| --- | --- | --- |
| ROS 2 Host and MuJoCo runtime | ROS 2 Humble system Python at `/usr/bin/python3` | [Runtime setup](runtime/README.md#environment-setup) |
| YOLO and LiteHRNet training, inference and export | Python 3.10 virtual environment managed by `uv` under `src/nn/.venv` | [Neural-network environment](src/nn/README.md#环境要求) |
| Blender scene rendering | Blender 4.5.0 bundled Python 3.11 | [Blender environment](src/blender/README.md#环境) |

Blender rendering runs inside Blender's bundled Python. Dataset composition and annotation visualization run after
rendering and reuse the `src/nn` `uv` environment; they do not run inside Blender. Likewise, the ROS 2 runtime uses
the system Python selected by the ROS installation and must not be launched from the training virtual environment.

## Documentation

- Start with the [runtime setup and launch guide](runtime/README.md) to run the complete ROS 2 + MuJoCo workflow.
- [Technical report](doc/RM2026-Engineer-Assembly-Algorithm.pdf)
- [Blender data generation](src/blender/README.md)
- [Shared neural-network environment](src/nn/README.md)
- [YOLO pose and detection](src/nn/yolo/README.md)
- [LiteHRNet](src/nn/hrnet/README.md)

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

### 2026-08-13

- Consolidated the YOLO and LiteHRNet tools behind the shared `rm26-nn` command-line interface.
- Expanded the ROS 2 runtime, MuJoCo engine and simulation architecture documentation.
- Simplified operator input to the keyboard controls consumed by the simulation and removed an unused custom event
  message.
- Aligned ROS package metadata with the repository maintainer and MIT license.

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
