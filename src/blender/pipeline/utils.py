"""相机、关键点投影和遮挡检查工具。"""

import numpy as np

import bpy
from mathutils import Vector


def get_camera_K(cam_data, scene):
    """根据 Blender 相机和渲染分辨率计算 OpenCV 内参矩阵。"""
    width = scene.render.resolution_x
    height = scene.render.resolution_y
    sensor_width = cam_data.sensor_width
    sensor_height = sensor_width * height / width
    focal_length = cam_data.lens
    focal_x = focal_length / sensor_width * width
    focal_y = focal_length / sensor_height * height
    return np.array(
        [[focal_x, 0.0, width / 2.0],
         [0.0, focal_y, height / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def blender_cam_to_opencv(cam_obj):
    """将 Blender 相机位姿转换为 OpenCV 的世界到相机变换。"""
    world_to_blender = np.array(cam_obj.matrix_world.inverted())
    blender_to_opencv = np.diag([1.0, -1.0, -1.0, 1.0])
    world_to_opencv = blender_to_opencv @ world_to_blender
    return world_to_opencv[:3, :3], world_to_opencv[:3, 3]


def project_keypoints(points_world, camera_matrix, rotation, translation, width, height):
    """投影三维关键点，返回 ``[u, v, visibility]`` 列表。"""
    projected = []
    for point in points_world:
        point_camera = rotation @ point + translation
        if point_camera[2] <= 0:
            projected.append([0.0, 0.0, 0])
            continue

        u = (camera_matrix[0, 0] * point_camera[0] / point_camera[2]
             + camera_matrix[0, 2])
        v = (camera_matrix[1, 1] * point_camera[1] / point_camera[2]
             + camera_matrix[1, 2])
        visibility = int(0 <= u < width and 0 <= v < height)
        projected.append([round(float(u), 2), round(float(v), 2), visibility])
    return projected


def all_kps_in_frame(points_world, camera_matrix, rotation, translation,
                     width, height, margin=4):
    """检查关键点是否位于图像内，并排除位于相机背后的点。"""
    for point in points_world:
        point_camera = rotation @ point + translation
        if point_camera[2] <= 0:
            return False
        u = (camera_matrix[0, 0] * point_camera[0] / point_camera[2]
             + camera_matrix[0, 2])
        v = (camera_matrix[1, 1] * point_camera[1] / point_camera[2]
             + camera_matrix[1, 2])
        if not margin <= u <= width - margin or not margin <= v <= height - margin:
            return False
    return True


def get_out_of_view_kps(keypoint_names, keypoints_2d):
    """返回投影结果中位于视野外的关键点名称。"""
    return [
        keypoint_names[index]
        for index, (_, _, visibility) in enumerate(keypoints_2d)
        if visibility == 0
    ]


def check_occlusion_raycasts(keypoints_2d, points_world, camera_position,
                             scene=None, offset_m=0.003):
    """沿关键点到相机的方向做射线检查，并更新可见性标记。

    ``visibility=0`` 表示视野外，``1`` 表示遮挡，``2`` 表示可见。
    只有投影在图像内且尚未完成遮挡检查的关键点会被更新。
    """
    scene = scene or bpy.context.scene
    camera = Vector(np.asarray(camera_position, dtype=np.float64).tolist())
    depsgraph = bpy.context.evaluated_depsgraph_get()

    for keypoint, point_world in zip(keypoints_2d, points_world):
        if keypoint[2] != 1:
            continue

        point = Vector(np.asarray(point_world, dtype=np.float64).tolist())
        direction = camera - point
        distance = direction.length
        if distance < 1e-6:
            keypoint[2] = 2
            continue

        direction.normalize()
        origin = point + direction * offset_m
        hit, *_ = scene.ray_cast(depsgraph, origin, direction, distance=distance)
        keypoint[2] = 1 if hit else 2

    return keypoints_2d


__all__ = [
    "all_kps_in_frame",
    "blender_cam_to_opencv",
    "check_occlusion_raycasts",
    "get_camera_K",
    "get_out_of_view_kps",
    "project_keypoints",
]
