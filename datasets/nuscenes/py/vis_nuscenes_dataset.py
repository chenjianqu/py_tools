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


def transform_points_to_image(nusc, top_lidar, cam_front):
    pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, top_lidar["filename"]))

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    ex_T_lidar_to_car = nusc.get('calibrated_sensor', top_lidar['calibrated_sensor_token'])
    pc.rotate(Quaternion(ex_T_lidar_to_car['rotation']).rotation_matrix)
    pc.translate(np.array(ex_T_lidar_to_car['translation']))

    # Second step: transform from ego to the global frame.
    T_car0_to_lidar = nusc.get('ego_pose', top_lidar['ego_pose_token'])
    pc.rotate(Quaternion(T_car0_to_lidar['rotation']).rotation_matrix)
    pc.translate(np.array(T_car0_to_lidar['translation']))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    T_car1_to_lidar = nusc.get('ego_pose', cam_front['ego_pose_token'])
    pc.translate(-np.array(T_car1_to_lidar['translation']))
    pc.rotate(Quaternion(T_car1_to_lidar['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    ex_T_cam_to_car = nusc.get('calibrated_sensor', cam_front['calibrated_sensor_token'])
    pc.translate(-np.array(ex_T_cam_to_car['translation']))
    pc.rotate(Quaternion(ex_T_cam_to_car['rotation']).rotation_matrix.T)

    return pc


def map_pointcloud_to_image_depth(
        nusc,
        pointsensor_token: str,
        camera_token: str,
        min_dist: float = 1.0) -> Tuple:
    """
    Given a point sensor (lidar/radar) token and camera sample_data token, load pointcloud and map it to the image
    plane.
    :param pointsensor_token: Lidar/radar sample_data token.
    :param camera_token: Camera sample_data token.
    :param min_dist: Distance from the camera below which points are discarded.
    :param render_intensity: Whether to render lidar intensity instead of point depth.
    :param show_lidarseg: Whether to render lidar intensity instead of point depth.
    :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
        or the list is empty, all classes will be displayed.
    :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                    predictions for the sample.
    :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
        to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        If show_lidarseg is True, show_panoptic will be set to False.
    :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
    """

    cam = nusc.get('sample_data', camera_token)
    pointsensor = nusc.get('sample_data', pointsensor_token)

    im = Image.open(osp.join(nusc.dataroot, cam['filename']))
    pc = transform_points_to_image(nusc, pointsensor, cam)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]
    coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    calib_para = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    points = view_points(pc.points[:3, :], np.array(calib_para['camera_intrinsic']), normalize=True)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    return points, coloring, im


def get_first_sample_test():
    data_root = '/media/cjq/新加卷2/BaiduNetdiskDownload/mini/v1.0-mini'
    nusc = NuScenes(version='v1.0-mini', dataroot=data_root, verbose=True)

    first_scene = nusc.scene[0]
    # 获得该scene第一个样本的信息
    first_sample_token = first_scene["first_sample_token"]
    my_sample = nusc.get('sample', first_sample_token)

    my_sample_data = my_sample["data"]  # dict{name:token}
    print(my_sample_data)
    my_sample_anns = my_sample["anns"]  # list[token]

    # print(my_sample_data["CAM_FRONT"])
    # nusc.render_sample_data(my_sample_data["CAM_FRONT"])

    my_sample_data_cam_front = nusc.get('sample_data', my_sample_data["CAM_FRONT"])
    print(my_sample_data_cam_front)
    print(os.path.join(data_root, my_sample_data_cam_front["filename"]))
    cam_front = cv2.imread(os.path.join(data_root, my_sample_data_cam_front["filename"]))
    cv2.imshow("cam_front", cam_front)
    cv2.waitKey(0)


def get_all_samples_of_scene():
    data_root = '/media/cjq/新加卷2/BaiduNetdiskDownload/mini/v1.0-mini'
    nusc = NuScenes(version='v1.0-mini', dataroot=data_root, verbose=True)

    first_scene = nusc.scene[0]
    print(first_scene)
    nbr_samples = first_scene["nbr_samples"]
    samples_tokens = []
    next_sample_token = first_scene["first_sample_token"]
    while True:
        if next_sample_token == '':
            break
        sample = nusc.get("sample", next_sample_token)
        samples_tokens.append(next_sample_token)
        print(sample)
        next_sample_token = sample["next"]
    print(len(samples_tokens))


def vis_all_image():
    data_root = '/media/cjq/新加卷2/BaiduNetdiskDownload/mini/v1.0-mini'
    nusc = NuScenes(version='v1.0-mini', dataroot=data_root, verbose=True)

    sample_data_list = []
    for sample_data in nusc.sample_data:
        if sample_data["channel"] == "CAM_FRONT":
            sample_data_list.append(sample_data)

    # 根据时间戳进行排序
    sample_data_list.sort(key=lambda x: (x["timestamp"], x["timestamp"]))

    for sample_data in sample_data_list:
        timestamp = sample_data["timestamp"]
        timestamp_s = timestamp / 1000000
        is_key_frame = sample_data["is_key_frame"]
        channel = sample_data["channel"]

        print(f"{timestamp_s} {is_key_frame} {channel}")

        cam_front = cv2.imread(os.path.join(data_root, sample_data["filename"]))
        cv2.imshow("cam_front", cam_front)
        cv2.waitKey(100)

    nusc.render_pointcloud_in_image()
    nusc.explorer.map_pointcloud_to_image()


def vis_sample_test():
    data_root = '/media/cjq/新加卷2/BaiduNetdiskDownload/mini/v1.0-mini'
    nusc = NuScenes(version='v1.0-mini', dataroot=data_root, verbose=True)

    first_scene = nusc.scene[0]
    first_sample_token = first_scene["first_sample_token"] # 获得该scene第一个样本的信息
    my_sample = nusc.get('sample', first_sample_token)

    cam_front = nusc.get('sample_data', my_sample["data"]["CAM_FRONT"])
    cam_front_image = cv2.imread(os.path.join(data_root, cam_front["filename"]))

    top_lidar = nusc.get('sample_data', my_sample["data"]["LIDAR_TOP"])

    pc = transform_points_to_image(nusc, top_lidar, cam_front)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]
    coloring = depths  # [N,]

    # [3,N],投影到像素点
    ex_T_cam_to_car = nusc.get('calibrated_sensor', cam_front['calibrated_sensor_token'])
    points = view_points(pc.points[:3, :], np.array(ex_T_cam_to_car['camera_intrinsic']), normalize=True)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    H, W = cam_front_image.shape[:2]
    min_dist = 0
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < W - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < H - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    points_int = points.astype(np.int16)
    for i in range(points.shape[1]):
        cv2.circle(cam_front_image, (points_int[0, i], points_int[1, i]), 1, (255, 255, 255), -1)

    cv2.imshow("cam_front", cam_front_image)
    cv2.waitKey(0)

    # 上面的过程等价于
    # out_path = "my_sample_data.png"
    # nusc.render_pointcloud_in_image(my_sample["token"], pointsensor_channel='LIDAR_TOP',
    #                                camera_channel='CAM_FRONT', verbose=True, out_path=out_path)


def vis_sample_of_render_pointcloud_in_image():
    data_root = '/media/cjq/新加卷2/BaiduNetdiskDownload/mini/v1.0-mini'
    nusc = NuScenes(version='v1.0-mini', dataroot=data_root, verbose=True)

    first_scene = nusc.scene[0]
    # 获得该scene第一个样本的信息
    first_sample_token = first_scene["first_sample_token"]
    my_sample = nusc.get('sample', first_sample_token)

    my_sample_data = my_sample["data"]  # dict{name:token}
    print(my_sample_data)

    # Here we just grab the front camera and the point sensor.
    pointsensor_token = my_sample['data']["LIDAR_TOP"]
    camera_token = my_sample['data']["CAM_FRONT"]
    # points, coloring, im = nusc.explorer.map_pointcloud_to_image(pointsensor_token,camera_token)
    points, coloring, im = map_pointcloud_to_image_depth(nusc, pointsensor_token, camera_token)
    img = np.asarray(im)

    points_int = points.astype(np.int16)
    for i in range(points.shape[1]):
        cv2.circle(img, (points_int[0, i], points_int[1, i]), 1, (255, 255, 255), -1)

    cv2.imshow("im", img)
    cv2.waitKey(0)


def main():
    # data_root = '/media/cjq/新加卷2/BaiduNetdiskDownload/mini/v1.0-mini'
    # nusc = NuScenes(version='v1.0-mini', dataroot=data_root, verbose=True)

    # nusc.list_attributes()
    # nusc.list_categories()
    # nusc.list_scenes()

    # print(f"num_samples:{len(nusc.sample)}")
    # nusc.list_sample(nusc.sample[0]["token"])

    # print(f"sample_data:{len(nusc.sample_data)}")
    # print(nusc.sample_data[0])

    # vis_all_image()

    vis_sample_test()

    # vis_sample_of_render_pointcloud_in_image()


if __name__ == "__main__":
    main()
