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
simulation. Those runtime components are outside this public archive.

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
