#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from inference import PillarHrnetPipeline, draw_pose_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO detect + LiteHRNet video inference")
    parser.add_argument("--yolo-weights", required=True)
    parser.add_argument("--hrnet-config", required=True)
    parser.add_argument("--hrnet-checkpoint", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bbox-class-id", type=int, default=1)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--crop-margin", type=float, default=0.05)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=-1)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--output-fps", type=float, default=0.0)
    parser.add_argument("--no-draw-scores", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    pipe = PillarHrnetPipeline(
        yolo_weights=args.yolo_weights,
        hrnet_config=args.hrnet_config,
        hrnet_checkpoint=args.hrnet_checkpoint,
        device=args.device,
        bbox_class_id=args.bbox_class_id,
        yolo_conf=args.yolo_conf,
        crop_margin=args.crop_margin,
    )

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {source}")

    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, args.frame_stride)
    out_fps = args.output_fps if args.output_fps > 0 else max(in_fps / stride, 1.0)

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        out_fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {output}")

    frame_idx = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx < args.start_frame:
            frame_idx += 1
            continue
        if args.end_frame >= 0 and frame_idx > args.end_frame:
            break
        if (frame_idx - args.start_frame) % stride != 0:
            frame_idx += 1
            continue
        if args.max_frames >= 0 and written >= args.max_frames:
            break

        result = pipe.predict_numpy(frame, image_id=f"frame_{frame_idx:06d}")
        writer.write(draw_pose_result(frame, result, draw_scores=not args.no_draw_scores))
        written += 1
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"saved video: {output}")


if __name__ == "__main__":
    main()
