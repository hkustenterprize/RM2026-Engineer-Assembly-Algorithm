import argparse
from pathlib import Path
from ultralytics import YOLO

YOLOPOSE_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOPose video inference")
    parser.add_argument(
        "--model",
        type=str,
        default=str(YOLOPOSE_ROOT / "runs/pose/yolo_v9.03/weights/best.pt"),
        help="Path to model weights (.pt)",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Input video file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(YOLOPOSE_ROOT / "eval_results/video"),
        help="Output directory to save annotated video",
    )
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on, e.g. '0', '0,1', 'cpu'",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    predict_kwargs = dict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        save=True,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
        stream=True,   # memory-efficient frame-by-frame processing
    )
    if args.device is not None:
        predict_kwargs["device"] = args.device

    print(f"[infer_video] model  : {args.model}")
    print(f"[infer_video] source : {args.source}")
    print(f"[infer_video] output : {output_dir}")

    for result in model.predict(**predict_kwargs):
        pass  # ultralytics saves annotated frames automatically

    print("[infer_video] Done. Results saved to:", output_dir)


if __name__ == "__main__":
    main()
