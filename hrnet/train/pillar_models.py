"""
pillar_models.py — Custom mmpose components for 5-keypoint pillar detection.

Provides:
  - GeometricConsistencyLoss     : HomographyConsistency + Chirality
  - PillarHeatmapHead            : HeatmapHead subclass with geometric loss
  - PillarHeatmapHeadWithVis     : PillarHeatmapHead + optional visibility prediction
  - PillarRTMCCHead              : RTMCCHead subclass with geometric loss

Keypoint ordering (fixed throughout this file):
    0 = TL  (top-left  outer corner)
    1 = TR  (top-right outer corner)
    2 = BL  (bottom-left  outer corner)
    3 = BR  (bottom-right outer corner)
    4 = ring (assembly pillar center)

Background
----------
The 4 corner keypoints live on the same plane in 3D (the exchange station face).
Under perspective projection they are related to a known 2D template by a
**homography H**.  This gives us one "free" consistency check: given TL/TR/BL/BR,
where should `ring` land?  Any deviation is a sign that ≥ 1 keypoint is wrong.

Additionally, the correct corner ordering TL→TR→BR→BL is *clockwise* in image
coordinates (y-axis pointing down).  Flipping TL↔TR changes the winding to
counter-clockwise, which we detect via the diagonal cross-product.

Both losses are differentiable end-to-end through **soft-argmax** decoding
(expected value of the predicted distribution) — no argmax involved.

Usage (config snippet)
----------------------
In td-hm_litehrnet30_pillar.py (HeatmapHead):

    custom_imports = dict(
        imports=['cv.nn.hrnet.pillar_models'], allow_failed_imports=False)

    model = dict(
        ...
        head=dict(
            type='PillarHeatmapHead',
            in_channels=40,
            out_channels=5,
            deconv_out_channels=(256,),
            deconv_kernel_sizes=(4,),
            loss=dict(type='KeypointMSELoss', use_target_weight=True),
            decoder=codec,
            geo_loss=dict(
                type='GeometricConsistencyLoss',
                aspect_ratio=1.0,
                input_size=256,
                heatmap_size=128,
                hom_weight=0.05,
                chiral_weight=0.02,
            ),
        ),
        ...
    )

In td-hm_litehrnet30_pillar_vis.py (with visibility head):

    head=dict(
        type='PillarHeatmapHeadWithVis',
        use_vis=True,                           # ← enable visibility prediction
        vis_loss=dict(type='BCELoss', use_target_weight=False, use_sigmoid=True),
        vis_target_mode='binary',               # 'binary' or 'continuous'
        ...same as PillarHeatmapHead...
    )

When ``use_vis=False`` (default), PillarHeatmapHeadWithVis behaves identically
to PillarHeatmapHead — making it a drop-in replacement with zero overhead.

In rtmpose_litehrnet30_pillar.py (RTMCCHead):

    head=dict(
        type='PillarRTMCCHead',
        ...same as RTMCCHead...
        geo_loss=dict(
            type='GeometricConsistencyLoss',
            aspect_ratio=1.0,
            input_size=256,
            simcc_split_ratio=2.0,
            hom_weight=0.05,
            chiral_weight=0.02,
        ),
    )
"""

from __future__ import annotations

import math
import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from torch.nn.modules.batchnorm import _BatchNorm
from torch import Tensor

# ── mmpose imports ──────────────────────────────────────────────────────────
# Support both "installed" mmpose and path-based mmpose
try:
    from mmengine.registry import OPTIMIZERS, OPTIM_WRAPPER_CONSTRUCTORS
    from mmpose.registry import MODELS
    from mmpose.models.heads import HeatmapHead
    from mmpose.models.necks import CSPNeXtPAFPN
    from mmpose.models.heads.coord_cls_heads import RTMCCHead
except ImportError:
    # Fallback: MMPOSE_ROOT env, then common install locations
    _MMPOSE_ROOT = os.environ.get("MMPOSE_ROOT")
    if not _MMPOSE_ROOT:
        for _candidate in (
            "/data/datasets/zguobd/mmpose",
            os.path.expanduser("~/mmpose"),
        ):
            if os.path.isdir(_candidate):
                _MMPOSE_ROOT = _candidate
                break
    if _MMPOSE_ROOT and _MMPOSE_ROOT not in sys.path:
        sys.path.insert(0, _MMPOSE_ROOT)
    from mmengine.registry import OPTIMIZERS, OPTIM_WRAPPER_CONSTRUCTORS
    from mmpose.registry import MODELS
    from mmpose.models.heads import HeatmapHead
    from mmpose.models.necks import CSPNeXtPAFPN
    from mmpose.models.heads.coord_cls_heads import RTMCCHead

try:
    from ultralytics.optim import MuSGD
    OPTIMIZERS.register_module(name="MuSGD", module=MuSGD, force=True)
except ImportError:
    MuSGD = None


@OPTIM_WRAPPER_CONSTRUCTORS.register_module(force=True)
class MuonSGDOptimWrapperConstructor:
    """Build Ultralytics MuSGD parameter groups for MMEngine.

    2D+ non-norm parameters use Muon+SGD. Bias, norm, and 1D parameters use
    ordinary SGD, matching Ultralytics' optimizer grouping semantics.
    """

    def __init__(self, optim_wrapper_cfg: dict, paramwise_cfg: dict | None = None):
        self.optim_wrapper_cfg = optim_wrapper_cfg.copy()
        self.optimizer_cfg = self.optim_wrapper_cfg.pop("optimizer").copy()
        self.paramwise_cfg = paramwise_cfg or {}

    def __call__(self, model: nn.Module):
        if MuSGD is None:
            raise ImportError("ultralytics.optim.MuSGD is not available in this environment")

        from mmengine.registry import OPTIM_WRAPPERS

        wrapper_cfg = self.optim_wrapper_cfg.copy()
        optimizer_cfg = self.optimizer_cfg.copy()
        wrapper_type = wrapper_cfg.pop("type", "OptimWrapper")
        optimizer_cfg.pop("type", None)

        base_lr = optimizer_cfg.get("lr", 1e-3)
        momentum = optimizer_cfg.get("momentum", 0.9)
        weight_decay = optimizer_cfg.get("weight_decay", 0.0)
        nesterov = optimizer_cfg.get("nesterov", True)
        muon = optimizer_cfg.pop("muon", 0.2)
        sgd = optimizer_cfg.pop("sgd", 1.0)
        bypass_duplicate = self.paramwise_cfg.get("bypass_duplicate", True)

        seen: set[nn.Parameter] = set()
        groups = dict(muon=[], decay=[], norm=[], bias=[])
        norm_types = tuple(
            cls for name, cls in nn.__dict__.items()
            if "Norm" in name and isinstance(cls, type)
        ) + (_BatchNorm,)

        for module_name, module in model.named_modules():
            is_norm = isinstance(module, norm_types)
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                if bypass_duplicate and param in seen:
                    continue
                seen.add(param)
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                if param.ndim >= 2 and not is_norm:
                    groups["muon"].append(param)
                elif "bias" in full_name:
                    groups["bias"].append(param)
                elif is_norm:
                    groups["norm"].append(param)
                else:
                    groups["decay"].append(param)

        param_groups = [
            dict(
                params=groups["muon"],
                lr=base_lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=nesterov,
                use_muon=True,
                param_group="muon",
            ),
            dict(
                params=groups["decay"],
                lr=base_lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=nesterov,
                use_muon=False,
                param_group="weight",
            ),
            dict(
                params=groups["norm"],
                lr=base_lr,
                momentum=momentum,
                weight_decay=0.0,
                nesterov=nesterov,
                use_muon=False,
                param_group="norm",
            ),
            dict(
                params=groups["bias"],
                lr=base_lr,
                momentum=momentum,
                weight_decay=0.0,
                nesterov=nesterov,
                use_muon=False,
                param_group="bias",
            ),
        ]
        param_groups = [group for group in param_groups if group["params"]]
        optimizer = MuSGD(param_groups, muon=muon, sgd=sgd)
        return OPTIM_WRAPPERS.build(dict(type=wrapper_type, optimizer=optimizer, **wrapper_cfg))


@MODELS.register_module(force=True)
class TopDownCSPNeXtPAFPN(CSPNeXtPAFPN):
    """Official CSPNeXtPAFPN top-down path only.

    The upstream PAFPN always builds and executes the bottom-up path, but this
    heatmap config only consumes the highest-resolution output. Keeping unused
    bottom-up parameters breaks DDP reduction and wastes inference kernels.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if tuple(self.out_indices) != (0,):
            raise ValueError("TopDownCSPNeXtPAFPN only supports out_indices=(0,)")
        if self.out_channels is not None:
            raise ValueError("TopDownCSPNeXtPAFPN expects out_channels=None")

        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()

    def forward(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        assert len(inputs) == len(self.in_channels)

        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_high = self.reduce_layers[len(self.in_channels) - 1 - idx](
                feat_high)
            inner_outs[0] = feat_high

            upsample_feat = self.upsample(feat_high)
            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.cat([upsample_feat, feat_low], 1))
            inner_outs.insert(0, inner_out)

        return (inner_outs[0],)


@MODELS.register_module(force=True)
class FullCSPNeXtPAFPNFuse(CSPNeXtPAFPN):
    """Use all CSPNeXtPAFPN branches and fuse them for a heatmap head.

    Official ``HeatmapHead`` consumes only ``feats[-1]``. This neck keeps the
    complete PAFPN path, resizes all selected outputs to the highest-resolution
    branch, concatenates them, and projects the result to one feature map.
    """

    def __init__(
        self,
        fuse_out_channels: int = 64,
        fuse_mode: str = "bilinear",
        align_corners: bool = False,
        conv_cfg: dict | None = None,
        norm_cfg: dict | None = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: dict | None = dict(type="SiLU"),
        **kwargs,
    ) -> None:
        kwargs["out_channels"] = None
        super().__init__(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg, **kwargs)
        self.fuse_mode = fuse_mode
        self.align_corners = align_corners
        self.fuse_conv = ConvModule(
            sum(self.in_channels[i] for i in self.out_indices),
            fuse_out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, inputs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        outs = super().forward(inputs)
        target_size = outs[0].shape[-2:]
        resized = [
            x if x.shape[-2:] == target_size else F.interpolate(
                x,
                size=target_size,
                mode=self.fuse_mode,
                align_corners=self.align_corners
                if self.fuse_mode in ("linear", "bilinear", "bicubic", "trilinear")
                else None,
            )
            for x in outs
        ]
        return (self.fuse_conv(torch.cat(resized, dim=1)),)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: soft-argmax from heatmap
# ═══════════════════════════════════════════════════════════════════════════════


def soft_argmax_heatmap(heatmaps: Tensor, stride: float) -> Tensor:
    """
    Extract keypoint coordinates from 2-D heatmaps via soft-argmax (expected value).

    Args:
        heatmaps : (B, K, Hm, Wm)  raw logits (unnormalised)
        stride   : heatmap-to-input scaling factor (e.g. input_size/heatmap_size)

    Returns:
        coords   : (B, K, 2)  x, y coordinates in *input* pixel space [0, input_size)
    """
    B, K, Hm, Wm = heatmaps.shape
    dev, dt = heatmaps.device, heatmaps.dtype

    # Pixel-centre positions in input space: (i + 0.5) * stride
    gy = (torch.arange(Hm, device=dev, dtype=dt) + 0.5) * stride  # (Hm,)
    gx = (torch.arange(Wm, device=dev, dtype=dt) + 0.5) * stride  # (Wm,)
    grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")  # (Hm, Wm)

    flat_hm = heatmaps.reshape(B, K, -1)  # (B, K, Hm*Wm)
    probs = torch.softmax(flat_hm, dim=-1)  # (B, K, Hm*Wm)

    cx = (probs * grid_x.reshape(-1)).sum(dim=-1)  # (B, K)
    cy = (probs * grid_y.reshape(-1)).sum(dim=-1)  # (B, K)
    return torch.stack([cx, cy], dim=-1)  # (B, K, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: soft-argmax from SimCC 1-D distributions
# ═══════════════════════════════════════════════════════════════════════════════


def soft_argmax_simcc(
    pred_x: Tensor, pred_y: Tensor, simcc_split_ratio: float
) -> Tensor:
    """
    Extract keypoint coordinates from SimCC 1-D distributions via soft-argmax.

    Args:
        pred_x           : (B, K, W_bins)  logits along x-axis
        pred_y           : (B, K, H_bins)  logits along y-axis
        simcc_split_ratio: bins = input_size × ratio  (e.g. 2.0 → 512 bins per 256 px)

    Returns:
        coords : (B, K, 2) x, y in input pixel space
    """
    B, K, Wb = pred_x.shape
    Hb = pred_y.shape[2]
    dev, dt = pred_x.device, pred_x.dtype

    bins_x = torch.arange(Wb, device=dev, dtype=dt)  # (Wb,)
    bins_y = torch.arange(Hb, device=dev, dtype=dt)  # (Hb,)

    px = torch.softmax(pred_x, dim=-1)  # (B, K, Wb)
    py = torch.softmax(pred_y, dim=-1)  # (B, K, Hb)

    cx = (px * bins_x).sum(dim=-1) / simcc_split_ratio  # (B, K)
    cy = (py * bins_y).sum(dim=-1) / simcc_split_ratio  # (B, K)
    return torch.stack([cx, cy], dim=-1)  # (B, K, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: differentiable 4-point DLT homography
# ═══════════════════════════════════════════════════════════════════════════════


def dlt_homography_4pt(src: Tensor, dst: Tensor) -> Tensor:
    """
    Compute homography H (batch) so that H @ src[i] ≈ dst[i] in homogeneous coords.

    Uses the Direct Linear Transform (DLT) on 4 point pairs — exactly determined.
    Differentiable via torch.linalg.svd.

    Args:
        src : (B, 4, 2)  source points  (e.g. normalised template corners)
        dst : (B, 4, 2)  destination points  (e.g. predicted corner coords)

    Returns:
        H   : (B, 3, 3)  with H[:, 2, 2] = 1
    """
    B = src.shape[0]
    x, y = src[:, :, 0], src[:, :, 1]  # (B, 4) each
    u, v = dst[:, :, 0], dst[:, :, 1]

    z = torch.zeros_like(x)
    o = torch.ones_like(x)

    # 2 rows per point correspondence
    row1 = torch.stack([-x, -y, -o, z, z, z, u * x, u * y, u], dim=2)  # (B,4,9)
    row2 = torch.stack([z, z, z, -x, -y, -o, v * x, v * y, v], dim=2)  # (B,4,9)
    # Interleave → (B, 8, 9)
    A = torch.stack([row1, row2], dim=2).reshape(B, 8, 9)

    # Null-space of A via SVD; last right-singular vector → h
    _, _, Vh = torch.linalg.svd(A, full_matrices=True)  # Vh: (B, 9, 9)
    h = Vh[:, -1, :]  # (B, 9)
    H = h.reshape(B, 3, 3)

    # Normalise so H[2,2] = 1
    denom = H[:, 2:3, 2:3].abs().clamp(min=1e-8)
    return H / denom


def project_pts(H: Tensor, pts: Tensor) -> Tensor:
    """
    Apply homography H to a set of 2-D points.

    Args:
        H   : (B, 3, 3)
        pts : (B, N, 2)

    Returns:
        out : (B, N, 2)  projected coordinates
    """
    B, N = pts.shape[:2]
    ones = torch.ones(B, N, 1, device=pts.device, dtype=pts.dtype)
    ph = torch.cat([pts, ones], dim=2)  # (B, N, 3)
    out = torch.bmm(H, ph.transpose(1, 2)).transpose(1, 2)  # (B, N, 3)
    w = out[:, :, 2:3].abs().clamp(min=1e-8)
    return out[:, :, :2] / w  # (B, N, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# GeometricConsistencyLoss
# ═══════════════════════════════════════════════════════════════════════════════


@MODELS.register_module(force=True)
class GeometricConsistencyLoss(nn.Module):
    """
    Geometric consistency loss for 5-keypoint pillar detection.

    Combines two complementary constraints:

    1. **DLT Reprojection Consistency**:
       The 5 keypoints have fixed 3-D physical coordinates (from CAD).
       Given the predicted 2-D positions, a 3×4 projection matrix P is
       estimated via Direct Linear Transform (5 point-pairs → 10 equations,
       SVD least-squares).  P absorbs K·[R|t] and any crop/resize transforms,
       so no camera intrinsics are needed.

       The same 3-D points are then reprojected through P; the reprojection
       residual w.r.t. the predicted 2-D points is the loss:

           L_reproj = mean_k ||P @ X_k / (P[2] @ X_k)  − x_k||²

       This is exact for all camera poses and dataset augmentations.

    2. **Chirality**: The quadrilateral TL→TR→BR→BL should be *clockwise*
       in image coordinates.  This is measured via the cross-product of the
       two diagonals (TL→BR) × (TR→BL).  A flip like TL↔TR reverses the
       sign of this cross product, so the loss fires immediately.

    Both losses use soft-argmax to decode differentiable coordinates from the
    model's distribution output.

    Physical keypoints (exchange station, metres):
        TL   : [-0.040,  0.040, 0.012]
        TR   : [ 0.040,  0.040, 0.012]
        BL   : [-0.040, -0.040, 0.012]
        BR   : [ 0.040, -0.040, 0.012]
        ring : [ 0.000,  0.000, 0.112]

    Args:
        aspect_ratio     : Unused, kept for API compatibility.
        hom_weight       : Weight for DLT reprojection loss.
        chiral_weight    : Weight for chirality loss.
        chiral_margin    : Safety margin in sin(θ) space for chirality;
                           the loss fires when sin_theta < chiral_margin.
        input_size       : Input image edge length (pixels), e.g. 256.
        heatmap_size     : (HeatmapHead mode) heatmap edge length AFTER any
                           deconv, e.g. 64 or 128.  Used to compute stride.
                           Ignored when simcc_split_ratio is set.
        simcc_split_ratio: (RTMCCHead / SimCC mode) bins = input_size * ratio.
                           Set this instead of heatmap_size for SimCC models.
    """

    def __init__(
        self,
        aspect_ratio: float = 1.0,
        hom_weight: float = 0.05,
        chiral_weight: float = 0.02,
        chiral_margin: float = 0.0,
        input_size: int = 256,
        heatmap_size: int = 64,  # used when simcc_split_ratio is None
        simcc_split_ratio: float | None = None,
    ) -> None:
        super().__init__()
        self.hom_weight = hom_weight
        self.chiral_weight = chiral_weight
        self.chiral_margin = chiral_margin
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.simcc_split_ratio = simcc_split_ratio

        if simcc_split_ratio is None:
            self.stride = float(input_size) / float(heatmap_size)
        else:
            self.stride = None  # unused for SimCC

        # aspect_ratio: unused, kept for API compat

        # 3-D physical coordinates of the 5 keypoints (metres):
        #   order: TL, TR, BL, BR, ring
        pts3d = torch.tensor(
            [
                [-0.040, 0.040, 0.012],
                [0.040, 0.040, 0.012],
                [-0.040, -0.040, 0.012],
                [0.040, -0.040, 0.012],
                [0.000, 0.000, 0.112],
            ],
            dtype=torch.float32,
        )  # (5, 3)
        # Homogeneous: (5, 4)
        ones = torch.ones(5, 1, dtype=torch.float32)
        self.register_buffer("pts3d_h", torch.cat([pts3d, ones], dim=1))  # (5, 4)

    # ── coordinate extraction ───────────────────────────────────────────────

    def _decode_heatmap(self, pred: Tensor) -> Tensor:
        """pred: (B, K, Hm, Wm) → coords (B, K, 2)"""
        return soft_argmax_heatmap(pred, self.stride)

    def _decode_simcc(self, pred_x: Tensor, pred_y: Tensor) -> Tensor:
        """pred_x/y: (B, K, bins) → coords (B, K, 2)"""
        return soft_argmax_simcc(pred_x, pred_y, self.simcc_split_ratio)

    # ── loss terms ─────────────────────────────────────────────────────────

    def _dlt_reproj_loss(self, coords: Tensor, weights: Tensor | None) -> Tensor:
        """
        3-D → 2-D DLT reprojection consistency.

        For each sample in the batch:
          1. Build a 10×12 DLT matrix A from the 5 predicted 2-D points
             and the fixed 3-D physical coordinates.
          2. Solve for the 3×4 projection matrix P via SVD (null-space of A).
          3. Reproject the 3-D points through P → predicted 2-D coords.
          4. Loss = mean squared reprojection residual (normalised to [0,1]²).

        P absorbs K·[R|t] and any crop/resize augmentation, so no camera
        intrinsics or data-pipeline knowledge is required.

        coords : (B, 5, 2)  predicted keypoint positions in input-image pixels
        weights: (B, 5) or None
        """
        B = coords.shape[0]
        # Guard against degenerate inputs (NaN/Inf coords from bad predictions)
        if not torch.isfinite(coords).all():
            return coords.new_zeros(1).squeeze()
        # Normalise 2-D coords to [0, 1] for numerical stability
        uv = coords / float(self.input_size)  # (B, 5, 2)
        u = uv[:, :, 0]  # (B, 5)
        v = uv[:, :, 1]

        # 3-D homogeneous coords, broadcast to batch: (B, 5, 4)
        X = self.pts3d_h.unsqueeze(0).expand(B, -1, -1)  # (B, 5, 4)

        z = torch.zeros_like(u)  # (B, 5)

        # DLT: for each point (X, u, v) two equations:
        #   [-X^T,  0^T, u*X^T] · p = 0
        #   [ 0^T, -X^T, v*X^T] · p = 0
        # p = vec(P^T), shape (12,)
        row1 = torch.stack(
            [
                -X[..., 0],
                -X[..., 1],
                -X[..., 2],
                -X[..., 3],
                z,
                z,
                z,
                z,
                u * X[..., 0],
                u * X[..., 1],
                u * X[..., 2],
                u * X[..., 3],
            ],
            dim=-1,
        )  # (B, 5, 12)
        row2 = torch.stack(
            [
                z,
                z,
                z,
                z,
                -X[..., 0],
                -X[..., 1],
                -X[..., 2],
                -X[..., 3],
                v * X[..., 0],
                v * X[..., 1],
                v * X[..., 2],
                v * X[..., 3],
            ],
            dim=-1,
        )  # (B, 5, 12)

        # Interleave rows → (B, 10, 12)
        A = torch.stack([row1, row2], dim=2).reshape(B, 10, 12)

        # Null-space via SVD → last right singular vector
        # detach A before SVD: gradient flows only through the reprojection step,
        # not through the SVD itself. This avoids NaN/Inf gradients when singular
        # values are near-degenerate (common at training start).
        with torch.no_grad():
            _, _, Vh = torch.linalg.svd(
                A.detach(), full_matrices=True
            )  # Vh: (B, 12, 12)
            p = Vh[:, -1, :]  # (B, 12)
            p = p / (p.norm(dim=-1, keepdim=True).clamp(min=1e-8))
        P = p.reshape(B, 3, 4)  # (B, 3, 4)  — treated as a constant w.r.t. autograd

        # Reproject: P @ X^T → (B, 3, 5), then homogeneous divide
        Xt = X.transpose(1, 2)  # (B, 4, 5)
        proj = torch.bmm(P, Xt)  # (B, 3, 5)

        # SVD null-vector has arbitrary sign; fix by checking mean depth.
        # detach() keeps sign decision non-differentiable but preserves full
        # gradient flow through proj.
        mean_depth = proj[:, 2, :].mean(dim=-1)  # (B,)
        sign = mean_depth.detach().sign()
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        proj = proj * sign.view(B, 1, 1)

        w_denom = proj[:, 2:3, :].abs().clamp(min=1e-6)
        uv_proj = (proj[:, :2, :] / w_denom).transpose(1, 2)  # (B, 5, 2)

        # Clamp per-point error to avoid single-sample explosion (e.g. degenerate geometry)
        err = (uv_proj - uv).pow(2).sum(dim=-1).clamp(max=100.0)  # (B, 5)

        if weights is not None:
            vis = (weights > 0).float()  # (B, 5)
            err = (err * vis).sum(dim=-1)  # (B,)
            denom = vis.sum(dim=-1).clamp(min=1.0)  # (B,)
            return (err / denom).mean()
        return err.mean()

    def _chirality_loss(self, coords: Tensor, weights: Tensor | None) -> Tensor:
        """
        Diagonal cross-product of TL→BR  and  TR→BL, normalised to
        sin(θ) ∈ [-1, 1]  where θ is the angle between the two diagonals.

        Correct winding (CW in image space, y-axis down):
            sin(θ) > 0

        TL↔TR flip reverses winding → sin(θ) < 0 → loss fires.

        Normalisation by diagonal lengths removes scale sensitivity, fixing
        the false-zero problem for degenerate symmetric inputs.

        Loss = ReLU(margin - sin_theta)  (default margin=0)
               → penalises only when diagonals cross in wrong direction

        Note: `chiral_margin` is now in [-1, 1] space.  A value of 0.1
        means the penalty fires unless the diagonals cross with a positive
        angle of >~5.7°.

        coords : (B, 5, 2)
        """
        tl = coords[:, 0]  # (B, 2)
        tr = coords[:, 1]
        bl = coords[:, 2]
        br = coords[:, 3]

        d1 = br - tl  # diagonal TL→BR  (B, 2)
        d2 = bl - tr  # diagonal TR→BL  (B, 2)

        # 2-D cross product (z-component) = |d1| |d2| sin(θ)
        cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]  # (B,)

        # Normalise to sin(θ) so the loss is scale-independent
        len1 = d1.norm(dim=-1).clamp(min=1e-6)  # (B,)
        len2 = d2.norm(dim=-1).clamp(min=1e-6)  # (B,)
        sin_theta = cross / (len1 * len2)  # (B,)  ∈ [-1, 1]

        # Penalise wrong chirality: fire when sin_theta < chiral_margin
        loss_raw = torch.relu(self.chiral_margin - sin_theta)  # (B,)

        if weights is not None:
            vis = (weights[:, :4] > 0).all(dim=-1).float()  # (B,)
            loss_raw = loss_raw * vis
            denom = vis.sum().clamp(min=1.0)
            return loss_raw.sum() / denom
        return loss_raw.mean()

    # ── public forward ──────────────────────────────────────────────────────

    def forward_heatmap(
        self,
        pred_heatmaps: Tensor,
        keypoint_weights: Tensor | None = None,
    ) -> Tensor:
        """
        Entry point for HeatmapHead.

        Args:
            pred_heatmaps   : (B, 5, Hm, Wm)  raw predicted heatmaps
            keypoint_weights: (B, 5)           GT visibility weights (optional)

        Returns:
            scalar loss tensor
        """
        coords = self._decode_heatmap(pred_heatmaps)  # (B, 5, 2)
        return self._combined(coords, keypoint_weights)

    def forward_simcc(
        self,
        pred_x: Tensor,
        pred_y: Tensor,
        keypoint_weights: Tensor | None = None,
    ) -> Tensor:
        """
        Entry point for RTMCCHead.

        Args:
            pred_x          : (B, 5, W_bins)
            pred_y          : (B, 5, H_bins)
            keypoint_weights: (B, 5)

        Returns:
            scalar loss tensor
        """
        coords = self._decode_simcc(pred_x, pred_y)  # (B, 5, 2)
        return self._combined(coords, keypoint_weights)

    def forward_coords(
        self,
        coords: Tensor,
        keypoint_weights: Tensor | None = None,
    ) -> Tensor:
        """
        Entry point when keypoint coordinates are already computed
        (e.g. after GAU attention refinement).

        Args:
            coords          : (B, 5, 2)  positions in input pixel space
            keypoint_weights: (B, 5)     GT visibility weights (optional)

        Returns:
            scalar loss tensor
        """
        return self._combined(coords, keypoint_weights)

    def _combined(self, coords: Tensor, weights: Tensor | None) -> Tensor:
        loss = coords.new_zeros(1).squeeze()
        if self.hom_weight > 0:
            loss = loss + self.hom_weight * self._dlt_reproj_loss(coords, weights)
        if self.chiral_weight > 0:
            loss = loss + self.chiral_weight * self._chirality_loss(coords, weights)
        return loss

    # `forward` is intentionally not defined here; call forward_heatmap or
    # forward_simcc from the head subclass so the interface is explicit.


# ═══════════════════════════════════════════════════════════════════════════════
# PillarHeatmapHead — HeatmapHead + GeometricConsistencyLoss
# ═══════════════════════════════════════════════════════════════════════════════


@MODELS.register_module(force=True)
class PillarHeatmapHead(HeatmapHead):
    """
    Drop-in replacement for HeatmapHead that optionally adds
    GeometricConsistencyLoss alongside the standard KeypointMSELoss.

    All constructor arguments are passed through to HeatmapHead.
    Additional:

    Args:
        geo_loss (dict | None):  Config for GeometricConsistencyLoss.
                                 If None, behaves identically to HeatmapHead.
    """

    def __init__(
        self,
        geo_loss: dict | None = None,
        geo_kpt_indices: tuple[int, ...] | list[int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if geo_loss is not None:
            self.geo_loss: GeometricConsistencyLoss = MODELS.build(geo_loss)
        else:
            self.geo_loss = None
        self.geo_kpt_indices = (
            tuple(geo_kpt_indices) if geo_kpt_indices is not None else None
        )

    def loss(self, feats, batch_data_samples, train_cfg={}):
        # Standard MSE loss from parent
        losses = super().loss(feats, batch_data_samples, train_cfg)

        if self.geo_loss is None:
            return losses

        # Re-run forward to get predicted heatmaps (lightweight, ~same as parent)
        pred_heatmaps = self.forward(feats)  # (B, K, Hm, Wm)

        # Gather keypoint visibility weights
        import torch

        keypoint_weights = torch.cat(
            [d.gt_instance_labels.keypoint_weights for d in batch_data_samples]
        )  # (B, K)

        if self.geo_kpt_indices is not None:
            pred_heatmaps = pred_heatmaps[:, self.geo_kpt_indices, ...]
            keypoint_weights = keypoint_weights[:, self.geo_kpt_indices]

        losses["loss_geo"] = self.geo_loss.forward_heatmap(
            pred_heatmaps, keypoint_weights
        )
        return losses


# ═══════════════════════════════════════════════════════════════════════════════
# PillarRTMCCHead — RTMCCHead + GeometricConsistencyLoss
# ═══════════════════════════════════════════════════════════════════════════════


@MODELS.register_module(force=True)
class PillarRTMCCHead(RTMCCHead):
    """
    Drop-in replacement for RTMCCHead that optionally adds:
      1. A spatial pooling layer before the head to match the official
         RTMPose feature-map size budget.
      2. GeometricConsistencyLoss alongside the standard KLDiscretLoss.

    Motivation
    ----------
    Official RTMPose uses CSPNeXt (stride=32) → in_featuremap_size=(6,8) →
    flatten_dims=48 → MLP = Linear(48, 256) ≈ 12K params.

    LiteHRNet (stride=4) gives 64×64 features → flatten_dims=4096 →
    MLP = Linear(4096, 256) ≈ 1M params — makes the loss hard to converge.
    Setting ``spatial_pool_size=(8,8)`` adds an AdaptiveAvgPool2d(8,8)
    *before* the head, reducing flatten_dims to 64 (≈ official 48).

    Args:
        spatial_pool_size (tuple[int,int] | None):
            If given, an AdaptiveAvgPool2d is inserted before processing.
            ``in_featuremap_size`` must be set to match (e.g. (8,8)).
        geo_loss (dict | None):
            Config for GeometricConsistencyLoss (use simcc_split_ratio).
    """

    def __init__(
        self,
        spatial_pool_size: tuple | None = None,
        geo_loss: dict | None = None,
        geo_kpt_indices: tuple[int, ...] | list[int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if spatial_pool_size is not None:
            self.spatial_pool = nn.AdaptiveAvgPool2d(spatial_pool_size)
        else:
            self.spatial_pool = None
        if geo_loss is not None:
            self.geo_loss: GeometricConsistencyLoss = MODELS.build(geo_loss)
        else:
            self.geo_loss = None
        self.geo_kpt_indices = (
            tuple(geo_kpt_indices) if geo_kpt_indices is not None else None
        )

    def forward(self, feats):
        """Optionally pool feature map spatially before the RTMCCHead pipeline."""
        if self.spatial_pool is not None:
            feats = list(feats)
            feats[-1] = self.spatial_pool(feats[-1])
        return super().forward(feats)

    def loss(self, feats, batch_data_samples, train_cfg={}):
        losses = super().loss(feats, batch_data_samples, train_cfg)

        if self.geo_loss is None:
            return losses

        import torch

        # Re-run forward to get SimCC distributions
        pred_x, pred_y = self.forward(feats)  # (B, K, W), (B, K, H)

        keypoint_weights = torch.cat(
            [d.gt_instance_labels.keypoint_weights for d in batch_data_samples]
        )  # (B, K)

        if self.geo_kpt_indices is not None:
            pred_x = pred_x[:, self.geo_kpt_indices, ...]
            pred_y = pred_y[:, self.geo_kpt_indices, ...]
            keypoint_weights = keypoint_weights[:, self.geo_kpt_indices]

        losses["loss_geo"] = self.geo_loss.forward_simcc(
            pred_x, pred_y, keypoint_weights
        )
        return losses


# ═══════════════════════════════════════════════════════════════════════════════
# PillarHeatmapHeadWithVis — PillarHeatmapHead + optional visibility head
# ═══════════════════════════════════════════════════════════════════════════════


@MODELS.register_module(force=True)
class PillarHeatmapHeadWithVis(PillarHeatmapHead):
    """
    PillarHeatmapHead with optional per-keypoint classification head.

    When ``use_vis=True``, adds a lightweight vis_head that takes the backbone
    features (before deconv) and predicts one score per keypoint:

        AdaptiveAvgPool2d(1) → Flatten → Linear(in_channels, K) → [Sigmoid]

    The supervision target is controlled by ``vis_label_mode``:

      - ``'in_frame'``: positive for any keypoint still inside the image
        (COCO v=1 or v=2). This matches the user's "occluded still counts
        as present; only fully out-of-frame is negative" requirement.
      - ``'visible'``: positive only for COCO v=2. This mode requires
        ``raw_ann_info['keypoints']`` to preserve the original 0/1/2 labels.

    During training, an additional BCE loss is added on top of MSE + geo_loss.
    During inference, ``keypoints_visible`` is always populated for
    compatibility. When ``vis_label_mode='in_frame'``, an extra
    ``keypoints_in_frame`` field is also populated.

    When ``use_vis=False`` (default), behaves **identically** to PillarHeatmapHead
    — no extra params, no extra loss.

    Args:
        use_vis (bool): Enable visibility prediction. Default: ``False``
        vis_loss (dict): Config for visibility BCE loss.
        vis_label_mode (str):
            ``'in_frame'`` (default) or ``'visible'``.
        vis_target_mode (str):
            ``'binary'`` (default): predict a 0/1 target.
            ``'continuous'`` is only meaningful with ``vis_label_mode='visible'``
            and maps raw COCO visibility 0/1/2 → 0/0.5/1.
        **kwargs: Passed to PillarHeatmapHead → HeatmapHead.
    """

    def __init__(self,
                 use_vis: bool = False,
                 vis_loss: dict | None = None,
                 vis_label_mode: str = 'in_frame',
                 vis_target_mode: str = 'binary',
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.use_vis = use_vis
        self.vis_label_mode = vis_label_mode
        self.vis_target_mode = vis_target_mode

        if not use_vis:
            self.vis_head = None
            self.vis_loss_module = None
            return

        # Build vis_head on raw backbone feature channels (before deconv)
        in_channels = self.in_channels  # e.g. 40 for LiteHRNet stage 3
        out_channels = self.out_channels  # K = 5

        if vis_loss is None:
            vis_loss = dict(
                type='BCELoss', use_target_weight=False, use_sigmoid=True)
        self.vis_loss_module = MODELS.build(vis_loss)

        self.use_sigmoid = vis_loss.get('use_sigmoid', True)

        modules = [
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, out_channels),
        ]
        if self.use_sigmoid:
            modules.append(nn.Sigmoid())
        self.vis_head = nn.Sequential(*modules)

    def _get_in_frame_target(self, data_sample, device: torch.device) -> Tensor:
        """Return 1 for keypoints that remain valid after crop/affine.

        Prefer ``gt_instance_labels.keypoint_weights`` because it is produced
        after ``TopdownAffine`` + ``GenerateTarget`` and therefore reflects
        whether a keypoint still has a valid supervision target in the cropped
        training sample. This keeps the visibility branch aligned with the
        heatmap branch under random bbox shift/scale/rotation.
        """
        keypoint_weights = getattr(data_sample.gt_instance_labels,
                                   'keypoint_weights', None)
        if keypoint_weights is not None:
            keypoint_weights = torch.as_tensor(
                keypoint_weights, device=device, dtype=torch.float32)
            if keypoint_weights.ndim == 1:
                keypoint_weights = keypoint_weights.unsqueeze(0)
            elif keypoint_weights.ndim > 2:
                keypoint_weights = keypoint_weights.reshape(
                    keypoint_weights.shape[0], keypoint_weights.shape[1], -1
                ).amax(dim=2)
            return (keypoint_weights > 0).float()

        keypoints_visible = getattr(data_sample.gt_instances, 'keypoints_visible',
                                    None)
        if keypoints_visible is None:
            raise AttributeError('Neither keypoint_weights nor keypoints_visible '
                                 'is available in data_sample.')

        keypoints_visible = torch.as_tensor(
            keypoints_visible, device=device, dtype=torch.float32)
        if keypoints_visible.ndim == 1:
            keypoints_visible = keypoints_visible.unsqueeze(0)
        elif keypoints_visible.ndim == 3:
            keypoints_visible = keypoints_visible[..., 0]
        return (keypoints_visible > 0.5).float()

    def _get_raw_visibility_target(self, data_sample,
                                   device: torch.device) -> Tensor | None:
        """Recover raw COCO 0/1/2 visibility from raw_ann_info when available."""
        raw_ann_info = data_sample.metainfo.get('raw_ann_info')
        if not raw_ann_info or 'keypoints' not in raw_ann_info:
            return None

        raw_keypoints = torch.as_tensor(
            raw_ann_info['keypoints'], device=device, dtype=torch.float32)
        return raw_keypoints.view(1, -1, 3)[..., 2]

    def _build_vis_target(self, data_sample,
                          device: torch.device) -> Tensor:
        if self.vis_label_mode == 'in_frame':
            return self._get_in_frame_target(data_sample, device)

        if self.vis_label_mode != 'visible':
            raise ValueError(f'Unknown vis_label_mode: {self.vis_label_mode}')

        raw_visibility = self._get_raw_visibility_target(data_sample, device)
        if raw_visibility is None:
            raw_visibility = self._get_in_frame_target(data_sample, device)

        if self.vis_target_mode == 'binary':
            return (raw_visibility >= 2).float()
        if self.vis_target_mode == 'continuous':
            return raw_visibility / 2.0
        raise ValueError(f'Unknown vis_target_mode: {self.vis_target_mode}')

    # ── forward helpers ─────────────────────────────────────────────────────

    def vis_forward(self, feats):
        """Forward vis_head on backbone features.

        Args:
            feats (Tuple[Tensor]): Multi-scale feature maps from backbone.

        Returns:
            Tensor | None: (B, K) visibility scores in [0, 1], or None
                if ``use_vis=False``.
        """
        if not self.use_vis or self.vis_head is None:
            return None
        x = feats[-1]
        return self.vis_head(x).reshape(-1, self.out_channels)

    # ── loss ────────────────────────────────────────────────────────────────

    def loss(self,
             feats,
             batch_data_samples,
             train_cfg = None):
        """Calculate losses: standard MSE + geo + optional in-frame/vis BCE.

        Args:
            feats: Multi-scale features.
            batch_data_samples: Batch data with GT labels.
            train_cfg: Training config.

        Returns:
            dict: Loss dict with keys ``loss_kpt``, optionally ``loss_geo``,
                ``loss_vis``, ``acc_vis``.
        """
        if train_cfg is None:
            train_cfg = {}
        losses = super().loss(feats, batch_data_samples, train_cfg)

        if not self.use_vis or self.vis_head is None:
            return losses

        import torch

        # Forward vis_head
        vis_pred = self.vis_forward(feats)  # (B, K)

        # Build per-keypoint targets from GT visibility semantics.
        vis_target = torch.cat([
            self._build_vis_target(d, vis_pred.device)
            for d in batch_data_samples
        ])

        vis_weights = None
        if self.vis_loss_module.use_target_weight:
            vis_weights = torch.cat([
                torch.as_tensor(
                    getattr(d.gt_instance_labels, 'keypoints_visible_weights',
                            None),
                    device=vis_pred.device,
                    dtype=torch.float32,
                ).reshape(1, -1)
                if getattr(d.gt_instance_labels, 'keypoints_visible_weights',
                           None) is not None else torch.ones_like(
                               self._build_vis_target(d, vis_pred.device))
                for d in batch_data_samples
            ])

        loss_vis = self.vis_loss_module(vis_pred, vis_target, vis_weights)
        losses['loss_vis'] = loss_vis

        # Accuracy for monitoring
        with torch.no_grad():
            pred_score = vis_pred if self.use_sigmoid else torch.sigmoid(vis_pred)
            pred_binary = (pred_score > 0.5).float()
            target_binary = (vis_target >= 0.5).float()
            acc = (pred_binary == target_binary).float().mean()
            losses['acc_vis'] = acc

        return losses

    # ── predict ─────────────────────────────────────────────────────────────

    def predict(self,
                feats,
                batch_data_samples,
                test_cfg = None):
        """Predict keypoints + optional visibility.

        Args:
            feats: Features (or [feats, feats_flip] for TTA).
            batch_data_samples: Batch data samples.
            test_cfg: Test config.

        Returns:
            Same as HeatmapHead.predict(), but when ``use_vis=True`` each
            ``InstanceData`` also contains ``keypoints_visible`` (np.ndarray,
            shape (1, K)) with values in [0, 1].
        """
        if test_cfg is None:
            test_cfg = {}
        preds = super().predict(feats, batch_data_samples, test_cfg)

        if not self.use_vis or self.vis_head is None:
            return preds

        from mmpose.models.utils.tta import flip_visibility

        # Get visibility predictions with optional TTA
        if test_cfg.get('flip_test', False):
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _vis = self.vis_forward(feats[0])
            _vis_flip = flip_visibility(
                self.vis_forward(feats[1]), flip_indices=flip_indices)
            batch_vis = (_vis + _vis_flip) * 0.5
        else:
            batch_vis = self.vis_forward(feats)  # (B, K)

        if batch_vis is None:
            return preds

        from mmpose.utils.tensor_utils import to_numpy

        batch_vis_np = to_numpy(batch_vis, unzip=True)

        if isinstance(preds, tuple):
            pred_instances, pred_fields = preds
        else:
            pred_instances = preds
            pred_fields = None

        for idx, pred_instance in enumerate(pred_instances):
            if len(pred_instance) == 0:
                continue

            vis = batch_vis_np[idx]
            if getattr(vis, 'ndim', 1) == 1:
                vis = vis[None, :]

            pred_instance.keypoints_visible = vis
            if self.vis_label_mode == 'in_frame':
                pred_instance.keypoints_in_frame = vis

        if pred_fields is not None:
            return pred_instances, pred_fields
        return pred_instances
