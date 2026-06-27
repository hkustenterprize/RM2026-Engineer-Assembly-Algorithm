#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2

from inference import PillarHrnetPipeline, RtmoPipeline, draw_pose_result


def parse_args():
    p = argparse.ArgumentParser(
        description="Video keypoint inference: YOLO+HRNet topdown or RTMO bottomup"
    )
    p.add_argument(
        "--mode",
        choices=("hrnet", "rtmo"),
        default="hrnet",
        help="Inference backend. hrnet keeps the old YOLO+HRNet pipeline.",
    )
    p.add_argument("--yolo-weights", help="Ultralytics YOLO Detect .pt weights")
    p.add_argument("--hrnet-config", help="MMPose HRNet config")
    p.add_argument("--hrnet-checkpoint", help="HRNet .pth checkpoint")
    p.add_argument("--rtmo-config", help="MMPose RTMO config")
    p.add_argument("--rtmo-checkpoint", help="RTMO .pth checkpoint")
    p.add_argument(
        "--rtmo-score-thr",
        type=float,
        default=None,
        help="Optional extra filter on RTMO instance score.",
    )
    p.add_argument("--source", required=True, help="Input video path")
    p.add_argument("--output", required=True, help="Output annotated video path")
    p.add_argument("--device", default="cuda:0")

    p.add_argument("--yolo-conf", type=float, default=0.25)
    p.add_argument("--pillar-class-id", type=int, default=0)
    p.add_argument(
        "--bbox-class-id",
        type=int,
        default=None,
        help="YOLO class used as crop bbox. Default auto: 0 for 5-kpt, 1 for 12-kpt.",
    )
    p.add_argument(
        "--crop-margin",
        type=float,
        default=0.5,
        help="Expand bbox by this fraction on each side. 0.5 makes width/height about 2x.",
    )

    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=-1, help="-1 means until video end")
    p.add_argument("--frame-stride", type=int, default=1, help="Process every N frames")
    p.add_argument("--max-frames", type=int, default=-1)
    p.add_argument("--output-fps", type=float, default=0.0, help="0 = input_fps / stride")

    p.add_argument("--no-corner-refine", action="store_true")
    p.add_argument("--no-draw-scores", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {source}")

    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, args.frame_stride)
    out_fps = args.output_fps if args.output_fps > 0 else max(in_fps / stride, 1.0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, out_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create output video: {output}")

    if args.mode == "rtmo":
        missing = [
            name
            for name in ("rtmo_config", "rtmo_checkpoint")
            if getattr(args, name) is None
        ]
        if missing:
            raise ValueError(
                "--mode rtmo requires: "
                + ", ".join("--" + name.replace("_", "-") for name in missing)
            )
        pipe = RtmoPipeline(
            rtmo_config=args.rtmo_config,
            rtmo_checkpoint=args.rtmo_checkpoint,
            device=args.device,
            score_thr=args.rtmo_score_thr,
        )
    else:
        missing = [
            name
            for name in ("yolo_weights", "hrnet_config", "hrnet_checkpoint")
            if getattr(args, name) is None
        ]
        if missing:
            raise ValueError(
                "--mode hrnet requires: "
                + ", ".join("--" + name.replace("_", "-") for name in missing)
            )
        pipe = PillarHrnetPipeline(
            yolo_weights=args.yolo_weights,
            hrnet_config=args.hrnet_config,
            hrnet_checkpoint=args.hrnet_checkpoint,
            device=args.device,
            pillar_class_id=args.pillar_class_id,
            bbox_class_id=args.bbox_class_id,
            yolo_conf=args.yolo_conf,
            crop_margin=args.crop_margin,
            corner_refine=not args.no_corner_refine,
        )

    log_prefix = f"[infer_video_{args.mode}]"
    print(log_prefix, "source:", source)
    print(log_prefix, "output:", output)
    print(log_prefix, "input_fps:", in_fps, "output_fps:", out_fps)
    if args.mode == "hrnet":
        print(log_prefix, "bbox_class_id:", pipe.bbox_class_id)

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
        annotated = draw_pose_result(
            frame,
            result,
            draw_scores=not args.no_draw_scores,
        )
        writer.write(annotated)
        written += 1

        if written % 20 == 0:
            print(f"{log_prefix} processed frame={frame_idx}, written={written}")

        frame_idx += 1

    cap.release()
    writer.release()

    print(log_prefix, "Done. frames written:", written)


if __name__ == "__main__":
    main()
