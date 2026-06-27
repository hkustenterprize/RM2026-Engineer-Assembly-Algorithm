#!/usr/bin/env python3
"""Export an Ultralytics YOLO *.pt checkpoint to OpenVINO IR (*.xml + *.bin).

Uses Ultralytics' exporter: YOLO(weights).export(format="openvino", ...).

Example:

  python export_openvino.py \\
    --weights /path/to/best.pt \\
    --imgsz 640 \\
    --half

Requires: ultralytics (+ OpenVINO toolchain as expected by your ultralytics version).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO .pt -> OpenVINO (Ultralytics export)")
    p.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to YOLO weights (.pt)",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Square inference size for export (default: 640)",
    )
    p.add_argument(
        "--half",
        dest="half",
        action="store_true",
        default=True,
        help="Export FP16 IR (default: on)",
    )
    p.add_argument(
        "--no-half",
        dest="half",
        action="store_false",
        help="Export FP32 IR",
    )
    p.add_argument(
        "--dynamic",
        action="store_true",
        help="Export dynamic shapes (if supported by exporter/model)",
    )
    p.add_argument(
        "--nms",
        dest="nms",
        action="store_true",
        default=True,
        help="Embed NMS in export when supported (default: on)",
    )
    p.add_argument(
        "--no-nms",
        dest="nms",
        action="store_false",
        help="Disable embedded NMS export",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Exporter device hint, e.g. cpu, 0, cuda:0 (passed through if supported)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    weights = Path(args.weights).expanduser().resolve()
    if not weights.is_file():
        print(f"[export_openvino] Missing weights file: {weights}", file=sys.stderr)
        return 2

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print(
            "[export_openvino] ultralytics is not installed in this environment.\n"
            "Install with: pip install ultralytics\n"
            f"Original error: {exc}",
            file=sys.stderr,
        )
        return 3

    print(f"[export_openvino] Loading: {weights}")
    model = YOLO(str(weights))

    export_kw: dict = {
        "format": "openvino",
        "imgsz": args.imgsz,
        "half": bool(args.half),
        "dynamic": bool(args.dynamic),
        "nms": bool(args.nms),
    }
    if args.device:
        export_kw["device"] = args.device

    print(f"[export_openvino] Export kwargs: {export_kw}")
    out = model.export(**export_kw)
    print(f"[export_openvino] Done: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
