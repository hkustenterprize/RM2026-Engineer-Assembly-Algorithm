"""Debug script: batch image inference with hardcoded model and eval samples.

For CLI video inference use infer_video.py instead.
"""

from pathlib import Path

from ultralytics import YOLO

YOLOPOSE_ROOT = Path(__file__).resolve().parent.parent
CV_NN_ROOT = YOLOPOSE_ROOT.parent
EVAL_SAMPLES = CV_NN_ROOT / "data/eval/eval_samples"
EVAL_RESULTS = YOLOPOSE_ROOT / "eval_results"

# Load a model
model = YOLO(
    str(YOLOPOSE_ROOT / "runs/detect/yolo26s_det_v10.12/weights/best.pt")
)

# Run batched inference on a list of images
image_names = [
    "image0.png",
    "image1.png",
    "image2.png",
    "image3.png",
    "image4.png",
    "image5.png",
    "image6.png",
    "image7.png",
    "image8.png",
    "image9.png",
    "image10.png",
    "image11.png",
    "image12.png",
    "image13.png",
    "image14.png",
    "image15.png",
    "image16.png",
    "image17.png",
    "image18.png",
    "image19.png",
    "image21.png",
    "image22.png",
    "image23.png",
    "image24.png",
]
results = model(
    [str(EVAL_SAMPLES / name) for name in image_names],
    conf=0.3,
)

# Process results list
for index, result in enumerate(results):
    boxes = result.boxes
    masks = result.masks
    keypoints = result.keypoints
    probs = result.probs
    obb = result.obb
    result.save(filename=str(EVAL_RESULTS / f"result_v10.12_best_{index}.jpg"))
