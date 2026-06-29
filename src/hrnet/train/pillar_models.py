from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from mmpose.models.heads import HeatmapHead
from mmpose.registry import MODELS


@MODELS.register_module(force=True)
class PillarHeatmapHead(HeatmapHead):
    """Minimal HeatmapHead wrapper kept for the public top-down pipeline."""

    def __init__(self, **kwargs) -> None:
        kwargs.pop("geo_loss", None)
        kwargs.pop("geo_kpt_indices", None)
        super().__init__(**kwargs)


@MODELS.register_module(force=True)
class PillarHeatmapHeadWithVis(PillarHeatmapHead):
    """Heatmap head with an optional per-keypoint visibility branch.

    The public release keeps only the features needed by the two exchange12
    LiteHRNet training lines:

    - standard heatmap regression;
    - optional per-keypoint BCE supervision;
    - ``in_frame`` or ``visible`` targets.
    """

    def __init__(
        self,
        use_vis: bool = False,
        vis_loss: dict | None = None,
        vis_label_mode: str = "in_frame",
        vis_target_mode: str = "binary",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.use_vis = bool(use_vis)
        self.vis_label_mode = str(vis_label_mode)
        self.vis_target_mode = str(vis_target_mode)

        if not self.use_vis:
            self.vis_head = None
            self.vis_loss_module = None
            self.use_sigmoid = True
            return

        if vis_loss is None:
            vis_loss = dict(
                type="BCELoss",
                use_target_weight=False,
                use_sigmoid=True,
            )

        self.vis_loss_module = MODELS.build(vis_loss)
        self.use_sigmoid = bool(vis_loss.get("use_sigmoid", True))

        layers: list[nn.Module] = [
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.in_channels, self.out_channels),
        ]
        if self.use_sigmoid:
            layers.append(nn.Sigmoid())
        self.vis_head = nn.Sequential(*layers)

    def _get_in_frame_target(self, data_sample, device: torch.device) -> Tensor:
        keypoint_weights = getattr(data_sample.gt_instance_labels, "keypoint_weights", None)
        if keypoint_weights is not None:
            keypoint_weights = torch.as_tensor(
                keypoint_weights,
                device=device,
                dtype=torch.float32,
            )
            if keypoint_weights.ndim == 1:
                keypoint_weights = keypoint_weights.unsqueeze(0)
            elif keypoint_weights.ndim > 2:
                keypoint_weights = keypoint_weights.reshape(
                    keypoint_weights.shape[0],
                    keypoint_weights.shape[1],
                    -1,
                ).amax(dim=2)
            return (keypoint_weights > 0).float()

        keypoints_visible = getattr(data_sample.gt_instances, "keypoints_visible", None)
        if keypoints_visible is None:
            raise AttributeError(
                "Neither keypoint_weights nor keypoints_visible is available in data_sample."
            )

        keypoints_visible = torch.as_tensor(
            keypoints_visible,
            device=device,
            dtype=torch.float32,
        )
        if keypoints_visible.ndim == 1:
            keypoints_visible = keypoints_visible.unsqueeze(0)
        elif keypoints_visible.ndim == 3:
            keypoints_visible = keypoints_visible[..., 0]
        return (keypoints_visible > 0.5).float()

    def _get_raw_visibility_target(self, data_sample, device: torch.device) -> Tensor | None:
        raw_ann_info = data_sample.metainfo.get("raw_ann_info")
        if not raw_ann_info or "keypoints" not in raw_ann_info:
            return None

        raw_keypoints = torch.as_tensor(
            raw_ann_info["keypoints"],
            device=device,
            dtype=torch.float32,
        )
        return raw_keypoints.view(1, -1, 3)[..., 2]

    def _build_vis_target(self, data_sample, device: torch.device) -> Tensor:
        if self.vis_label_mode == "in_frame":
            return self._get_in_frame_target(data_sample, device)

        if self.vis_label_mode != "visible":
            raise ValueError(f"Unknown vis_label_mode: {self.vis_label_mode}")

        raw_visibility = self._get_raw_visibility_target(data_sample, device)
        if raw_visibility is None:
            raw_visibility = self._get_in_frame_target(data_sample, device)

        if self.vis_target_mode == "binary":
            return (raw_visibility >= 2).float()
        if self.vis_target_mode == "continuous":
            return raw_visibility / 2.0
        raise ValueError(f"Unknown vis_target_mode: {self.vis_target_mode}")

    def vis_forward(self, feats) -> Tensor | None:
        if not self.use_vis or self.vis_head is None:
            return None
        return self.vis_head(feats[-1]).reshape(-1, self.out_channels)

    def loss(self, feats, batch_data_samples, train_cfg=None):
        if train_cfg is None:
            train_cfg = {}
        losses = super().loss(feats, batch_data_samples, train_cfg)

        if not self.use_vis or self.vis_head is None:
            return losses

        vis_pred = self.vis_forward(feats)
        vis_target = torch.cat(
            [self._build_vis_target(sample, vis_pred.device) for sample in batch_data_samples]
        )

        vis_weights = None
        if self.vis_loss_module.use_target_weight:
            vis_weights = torch.cat(
                [
                    torch.as_tensor(
                        getattr(sample.gt_instance_labels, "keypoints_visible_weights", None),
                        device=vis_pred.device,
                        dtype=torch.float32,
                    ).reshape(1, -1)
                    if getattr(sample.gt_instance_labels, "keypoints_visible_weights", None) is not None
                    else torch.ones_like(self._build_vis_target(sample, vis_pred.device))
                    for sample in batch_data_samples
                ]
            )

        losses["loss_vis"] = self.vis_loss_module(vis_pred, vis_target, vis_weights)

        with torch.no_grad():
            pred_score = vis_pred if self.use_sigmoid else torch.sigmoid(vis_pred)
            pred_binary = (pred_score > 0.5).float()
            target_binary = (vis_target >= 0.5).float()
            losses["acc_vis"] = (pred_binary == target_binary).float().mean()

        return losses

    def predict(self, feats, batch_data_samples, test_cfg=None):
        if test_cfg is None:
            test_cfg = {}
        preds = super().predict(feats, batch_data_samples, test_cfg)

        if not self.use_vis or self.vis_head is None:
            return preds

        batch_vis = self.vis_forward(feats)
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
            if getattr(vis, "ndim", 1) == 1:
                vis = vis[None, :]
            pred_instance.keypoints_visible = vis
            if self.vis_label_mode == "in_frame":
                pred_instance.keypoints_in_frame = vis

        if pred_fields is not None:
            return pred_instances, pred_fields
        return pred_instances
