"""utils.py — 复用旧 pipeline 工具函数 + 新增物体变换辅助."""
import numpy as np

from pipeline.utils import (
    get_camera_K,
    blender_cam_to_opencv,
    project_keypoints,
    all_kps_in_frame,
    get_out_of_view_kps,
    forward_distort_kps,
    crop_offset_kps,
    compute_bbox_from_alpha,
    compute_bbox_3d_to_2d,
    check_occlusion_raycasts,
)

__all__ = [
    "get_camera_K", "blender_cam_to_opencv",
    "project_keypoints", "all_kps_in_frame", "get_out_of_view_kps",
    "forward_distort_kps", "crop_offset_kps",
    "compute_bbox_from_alpha", "compute_bbox_3d_to_2d",
    "check_occlusion_raycasts",
    "transform_keypoints_3d",
]


def transform_keypoints_3d(kp_3d_initial, matrix_world_4x4):
    """用 4x4 变换矩阵将初始 3D 关键点变换到当前帧世界坐标.

    Parameters
    ----------
    kp_3d_initial : list of np.ndarray (3,)
        初始关键点世界坐标 (从 YAML 读取的静态值).
    matrix_world_4x4 : Matrix or np.ndarray (4,4)
        Blender Empty 的 matrix_world (包含旋转 + 平移).

    Returns
    -------
    list of np.ndarray (3,)  变换后的关键点世界坐标.
    """
    M = np.array(matrix_world_4x4, dtype=np.float64)
    result = []
    for pt in kp_3d_initial:
        p_homo = np.append(pt, 1.0)
        p_new = (M @ p_homo)[:3]
        result.append(p_new)
    return result
