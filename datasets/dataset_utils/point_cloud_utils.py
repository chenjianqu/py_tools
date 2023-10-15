from typing import List

import open3d as o3d
import numpy as np


def vis_points_cloud_with_o3d(xyz: List[np.ndarray], colors: List[np.ndarray] = None):
    """
    Args:
        xyz: 多帧点云组成List,每个元素都是一个点云的 ndarray，which size is [N,3]
        colors: 多帧点云对应的颜色List，每个元素 [N,3]
    Returns:

    """
    assert xyz[0].shape[1] == 3

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='depth pcd')
    # vis.toggle_full_screen() #全屏
    # 设置
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])  # 背景
    opt.point_size = 1  # 点云大小

    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
    vis.add_geometry(axis_pcd)

    for i, points in enumerate(xyz):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            assert points.shape[0] == colors[i].shape[0]
            point_cloud.colors = o3d.utility.Vector3dVector(colors[i])

        vis.add_geometry(point_cloud)

    vis.run()
    vis.destroy_window()


def write_ply(write_path: str, xyz: np.ndarray, colors: np.ndarray = None):
    """
    Args:
        write_path: 写入路径
        xyz: 点云ndarray，which size is [N,3]
        colors: 点云颜色[N,3]
    Returns:

    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz.tolist())
    if colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(write_path, point_cloud)
