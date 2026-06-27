#!/usr/bin/env python3
"""Export an MMPose HRNet / LiteHRNet checkpoint (.pth) to OpenVINO IR.

HRNet inference in this repo is PyTorch + MMPose. OpenVINO does not load .pth
directly; this script exports the **heatmap computation subgraph**:

    normalized_RGB_patch (N,3,H,W) -> heatmaps (N,K,h',w')

where (H,W) defaults to ``codec.input_size`` from your config (often 256x256).

Pipeline pieces NOT included in the IR (must stay in CPU/host code):
  - YOLO bbox, TopdownAffine crop/warp, RGB/BGR handling
  - Heatmap -> keypoint decode (MSRAHeatmap / dark refine), mapping back to image coords

Steps:
  1) ``torch.onnx.export`` of a small wrapper module (backbone + optional neck + head.forward)
  2) ``openvino.convert_model(onnx_path)`` -> IR (.xml + .bin)

Requires: mmpose, mmengine, torch, onnx, openvino (same env you train/infer HRNet with).

Example::

  PYTHONPATH=/data/datasets/zguobd/mmpose:$PYTHONPATH \\
  python /data/datasets/zguobd/RM2026-Engineer-Host/cv/nn/hrnet/export_hrnet_openvino.py \\
    --config /path/to/td-hm_litehrnet30_exchange12_v9.1.py \\
    --checkpoint /path/to/best_coco_AP_epoch_xx.pth \\
    --output-dir /path/to/out_ov \\
    --opset 17
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MMPose HRNet/LiteHRNet .pth -> ONNX -> OpenVINO IR")
    p.add_argument("--config", required=True, type=str, help="MMPose config .py")
    p.add_argument("--checkpoint", required=True, type=str, help="HRNet checkpoint .pth")
    p.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory to write hrnet.onnx and OpenVINO IR",
    )
    p.add_argument(
        "--onnx-name",
        type=str,
        default="hrnet_heatmap.onnx",
        help="ONNX filename inside output-dir (default: hrnet_heatmap.onnx)",
    )
    p.add_argument(
        "--xml-name",
        type=str,
        default="hrnet_heatmap.xml",
        help="OpenVINO IR xml filename inside output-dir (default: hrnet_heatmap.xml)",
    )
    p.add_argument("--opset", type=int, default=17, help="ONNX opset (default: 17)")
    p.add_argument(
        "--skip-openvino",
        action="store_true",
        help="Only export ONNX, skip OpenVINO conversion",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for export trace (default: cpu). Use cuda:0 if needed.",
    )
    return p.parse_args()


class HeatmapForwardOnnx(nn.Module):
    """ONNX-friendly wrapper matching HeatmapHead.forward(feat_tuple)[-1]."""

    def __init__(self, pose_estimator: nn.Module) -> None:
        super().__init__()
        self.backbone = pose_estimator.backbone
        self.neck = getattr(pose_estimator, "neck", None)
        self.head = pose_estimator.head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if self.neck is not None:
            feats = self.neck(feats)
        if isinstance(feats, torch.Tensor):
            feat_tuple = (feats,)
        else:
            feat_tuple = feats  # type: ignore[assignment]
        return self.head(feat_tuple)


def _patch_litehrnet_for_onnx() -> None:
    """Replace LiteHRNet dynamic pooling with static avg pooling for export.

    PyTorch's ONNX exporter cannot handle
    ``adaptive_avg_pool2d(x, output_size=tensor.shape[-2:])``. LiteHRNet uses
    that pattern in CrossResolutionWeighting to pool all high-resolution
    branches to the lowest-resolution branch. For a fixed export input size the
    branch ratios are static, so avg_pool2d with kernel=stride is equivalent.
    """
    try:
        from mmpose.models.backbones.litehrnet import CrossResolutionWeighting
    except ImportError:
        return

    if getattr(CrossResolutionWeighting, "_onnx_export_patched", False):
        return

    def forward_onnx(self, x):
        mini_h = int(x[-1].shape[-2])
        mini_w = int(x[-1].shape[-1])

        pooled = []
        for s in x[:-1]:
            h = int(s.shape[-2])
            w = int(s.shape[-1])
            if h % mini_h != 0 or w % mini_w != 0:
                raise RuntimeError(
                    "LiteHRNet ONNX export expects branch sizes to be exact "
                    f"multiples of the lowest-resolution branch, got {(h, w)} "
                    f"and {(mini_h, mini_w)}."
                )
            kernel = (h // mini_h, w // mini_w)
            pooled.append(F.avg_pool2d(s, kernel_size=kernel, stride=kernel))

        out = pooled + [x[-1]]
        out = torch.cat(out, dim=1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = torch.split(out, self.channels, dim=1)
        out = [
            s * F.interpolate(a, size=s.shape[-2:], mode="nearest")
            for s, a in zip(x, out)
        ]
        return out

    CrossResolutionWeighting.forward = forward_onnx
    CrossResolutionWeighting._onnx_export_patched = True


def _resolve_input_hw(cfg) -> tuple[int, int]:
    """Return (H, W) for model input from codec.input_size."""
    head_cfg = cfg.model["head"]
    decoder = head_cfg.get("decoder")
    if decoder is None:
        # Many configs store MSRAHeatmap settings under top-level ``codec = dict(...)``
        decoder = cfg.get("codec")
    if decoder is None:
        raise ValueError(
            "Cannot resolve MSRAHeatmap codec settings: expected "
            "``head.decoder`` or top-level ``codec`` in config."
        )

    input_size = decoder.get("input_size")
    if input_size is None:
        raise ValueError("decoder/input_size missing in config")

    if isinstance(input_size, (list, tuple)):
        if len(input_size) != 2:
            raise ValueError(f"Unexpected input_size: {input_size}")
        h, w = int(input_size[0]), int(input_size[1])
    else:
        s = int(input_size)
        h, w = s, s
    return h, w


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / args.onnx_name
    xml_path = out_dir / args.xml_name

    cfg_path = Path(args.config).expanduser().resolve()
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not cfg_path.is_file():
        print(f"[export_hrnet_openvino] Missing config: {cfg_path}", file=sys.stderr)
        return 2
    if not ckpt_path.is_file():
        print(f"[export_hrnet_openvino] Missing checkpoint: {ckpt_path}", file=sys.stderr)
        return 2

    try:
        from mmengine.config import Config
        from mmpose.apis import init_model
    except ImportError as exc:
        print(
            "[export_hrnet_openvino] Need mmengine + mmpose in this python env.\n"
            f"{exc}",
            file=sys.stderr,
        )
        return 3

    cfg = Config.fromfile(str(cfg_path))
    device = torch.device(args.device)

    _patch_litehrnet_for_onnx()

    print(f"[export_hrnet_openvino] Loading model:\n  config={cfg_path}\n  ckpt={ckpt_path}")
    model = init_model(str(cfg_path), str(ckpt_path), device=str(device))
    model.eval()

    in_h, in_w = _resolve_input_hw(cfg)
    dummy = torch.randn(1, 3, in_h, in_w, device=device)

    wrapped = HeatmapForwardOnnx(model).to(device)
    wrapped.eval()

    print(
        f"[export_hrnet_openvino] Export ONNX: {onnx_path}\n"
        f"  input: (1,3,{in_h},{in_w}) RGB normalized like training patch\n"
        f"  output: heatmaps tensor (see HeatmapHead)"
    )

    torch.onnx.export(
        wrapped,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["heatmaps"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch"}, "heatmaps": {0: "batch"}},
    )

    if args.skip_openvino:
        print("[export_hrnet_openvino] Skip OpenVINO (--skip-openvino). Done.")
        return 0

    try:
        from openvino import convert_model, save_model
    except ImportError as exc:
        print(
            "[export_hrnet_openvino] OpenVINO Python package missing.\n"
            "Install openvino in this env, or pass --skip-openvino.\n"
            f"{exc}",
            file=sys.stderr,
        )
        return 4

    print(f"[export_hrnet_openvino] Converting ONNX -> OpenVINO IR: {xml_path}")
    ov_model = convert_model(str(onnx_path))
    save_model(ov_model, str(xml_path))

    print("[export_hrnet_openvino] Done.")
    print(f"  onnx: {onnx_path}")
    print(f"  openvino: {xml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
