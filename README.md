# RM2026 Engineer Assembly Algorithm

香港科技大学 ENTERPRIZE 战队 RoboMaster 2026 赛季工程装配算法开源仓库。

## Overview

RoboMaster 2026 赛季工程机器人新增的能量单元装配任务，对操作手在接近、插入、翻转和退出等阶段提出了较高要求。该项目尝试在不替代操作手主导控制的前提下，引入上位机辅助算法，完成装配站视觉定位、接近规划、受约束动作段规划、恢复动作和仿真验证。

更完整的工程说明见技术报告：

- [Open Source Technical Report](doc/open_source_technical_report.pdf)

## Release and Repository Update Log

This repository is being released incrementally. The current public archive focuses on the technical report, dataset entry and model checkpoints. Source code and runnable examples will be added after cleanup.

### 2026-06-24 Public Preview

- Released the open-source technical report PDF.
- Added the public README and project overview.
- Added the Hugging Face dataset entry.
- Added model checkpoint links for LiteHRNet, YOLO detection and YOLOPose baseline.


## Data Generation, Training and Inference Pipeline

*The Blender-based data generation scripts, training configurations and inference examples are being cleaned for public release. TBA*

## Project and Source Code Release

*The project engineering files and algorithm implementation are being reorganized before release. TBA*

## Dataset Release

The synthetic dataset is hosted on Hugging Face Datasets: [hkustenterprize/RM26_engineer_exchange](https://huggingface.co/datasets/hkustenterprize/RM26_engineer_exchange)

| Split | Ratio | Samples |
| --- | ---: | ---: |
| Train | 90% | 40,860 |
| Val | 10% | 4,540 |
| Total | 100% | 45,400 |

## Checkpoint Release

All checkpoints are hosted on Hugging Face: [hkustenterprize/RM26_engineer_exchange_model](https://huggingface.co/hkustenterprize/RM26_engineer_exchange_model)

| Model | Task | Input size | Pipeline Catagories | Native checkpoint | OpenVINO export |
| --- | --- | --- | --- | --- | --- |
| LiteHRNet-30-v9.1 | 12-keypoint | 256x256 | top-down | [best.pth](https://huggingface.co/hkustenterprize/RM26_engineer_exchange_model/resolve/main/litehr/litehrnet30_v9.1/best.pth?download=true) | [openvino_best](https://huggingface.co/hkustenterprize/RM26_engineer_exchange_model/tree/main/litehr/litehrnet30_v9.1/openvino_best) |
| YOLO26-s-v10.12 | Box Detection | 640x640 | top-down | [best.pt](https://huggingface.co/hkustenterprize/RM26_engineer_exchange_model/resolve/main/yolo/v10.12/best.pt?download=true) | [openvino_best](https://huggingface.co/hkustenterprize/RM26_engineer_exchange_model/tree/main/yolo/v10.12/openvino_best) |
| YOLO26-s-pose-v11.02 | Box Detection + 12-keypoint | 640x640 | bottom-up | [best.pt](https://huggingface.co/hkustenterprize/RM26_engineer_exchange_model/resolve/main/yolopose/v11.02/best.pt?download=true) | [openvino_best](https://huggingface.co/hkustenterprize/RM26_engineer_exchange_model/tree/main/yolopose/v11.02/openvino_best) |

The `resolve/main/...?...download=true` links point to direct checkpoint downloads. The `tree/main/.../openvino_best` links point to exported OpenVINO model directories. All of these checkpoints are trained with 100% synthetic data released above. Also we adopted the top-down pipeline in the real application, we also release the bottom-up checkpoint (yolo26-s-pose-v11.02) trained with the same data for reference.


## License

License information will be added before the first full public release. Until then, please treat this repository as a preview release for technical reference.

## Acknowledgement

This project was developed by HKUST ENTERPRIZE for the RoboMaster 2026 season. The implementation builds on open-source robotics and computer vision tools including MuJoCo, OMPL, hpp-fcl, Ultralytics, MMPose, SciPy and Albumentations.
