import os.path
import os.path as osp

from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion

from datasets.dataset_utils.point_cloud_utils import vis_points_cloud_with_o3d


def get_all_sample_data(nusc, chanenl):
    # 获取该scene所有的lidar帧
    sample_data_list = []
    for sample_data in nusc.sample_data:
        if sample_data["channel"] == chanenl:
            sample_data_list.append(sample_data)

    # 根据时间戳进行排序
    sample_data_list.sort(key=lambda x: (x["timestamp"], x["timestamp"]))
    return sample_data_list


def get_all_sample_data_of_scene(nusc, scene, chanenl="LIDAR_TOP"):
    """
    获得 属于scene的，传感器通道为chanenl的所有数据，并进行排序
    :param nusc:
    :param scene:
    :param chanenl:
    :return:
    """
    first_sample = nusc.get("sample", scene["first_sample_token"])
    samples_data_list = []

    curr_lidar_data = nusc.get("sample_data", first_sample["data"][chanenl])
    samples_data_list.append(curr_lidar_data)

    next_token = curr_lidar_data["next"]
    while True:
        if next_token == '':
            break
        sample_data = nusc.get("sample_data", next_token)
        samples_data_list.append(sample_data)
        next_token = sample_data["next"]

    # print("next len:{}".format(len(samples_data_list)))
    prev_token = curr_lidar_data["prev"]
    while True:
        if prev_token == '':
            break
        sample_data = nusc.get("sample_data", prev_token)
        samples_data_list.append(sample_data)
        prev_token = sample_data["prev"]

    # print("prev len:{}".format(len(samples_data_list)))

    # 根据时间戳进行排序
    samples_data_list.sort(key=lambda x: (x["timestamp"], x["timestamp"]))
    return samples_data_list


def generate_one_scene_pointcloud(nusc, scene):
    # 获取该scene所有的lidar帧
    sample_data_list = []
    for sample_data in nusc.sample_data:
        if sample_data["channel"] == "CAM_FRONT":
            sample_data_list.append(sample_data)

    # 根据时间戳进行排序
    sample_data_list.sort(key=lambda x: (x["timestamp"], x["timestamp"]))


def vis_all_lidar_points():
    data_root = '/media/cjq/新加卷2/BaiduNetdiskDownload/mini/v1.0-mini'
    nusc = NuScenes(version='v1.0-mini', dataroot=data_root, verbose=True)
    scene = nusc.scene[0]
    data_list = get_all_sample_data_of_scene(nusc,scene,chanenl="LIDAR_TOP")
    pc_list = []

    # 将所有的点云叠起来
    for top_lidar in data_list:
        pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, top_lidar["filename"]))
        ex_T_lidar_to_car = nusc.get('calibrated_sensor', top_lidar['calibrated_sensor_token'])
        pc.rotate(Quaternion(ex_T_lidar_to_car['rotation']).rotation_matrix)
        pc.translate(np.array(ex_T_lidar_to_car['translation']))

        # Second step: transform from ego to the global frame.
        T_car0_to_lidar = nusc.get('ego_pose', top_lidar['ego_pose_token'])
        pc.rotate(Quaternion(T_car0_to_lidar['rotation']).rotation_matrix)
        pc.translate(np.array(T_car0_to_lidar['translation']))
        pc_list.append(pc.points) # [4,N] np.ndarray

        # 根据3D框的点云，去除动态物体

    # 合并点云
    points_all = np.concatenate(pc_list, axis=1) # [4,NxM]
    points_all = points_all[:3,:]

    vis_points_cloud_with_o3d([points_all.transpose()])
    # write_ply(write_path: str, xyz: np.ndarray)

    # nusc.render_pointcloud_in_image()


def main():
    vis_all_lidar_points()


if __name__ == "__main__":
    main()


