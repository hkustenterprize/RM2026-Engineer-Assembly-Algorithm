"""pipeline_new — 固定相机 + 物体变换 的合成数据管线."""
from .context import RenderContext
from .ops import build_ops

__all__ = ["RenderContext", "build_ops"]
