"""固定相机与物体变换的 Blender 合成数据管线。"""
from .context import RenderContext
from .ops import build_ops

__all__ = ["RenderContext", "build_ops"]
