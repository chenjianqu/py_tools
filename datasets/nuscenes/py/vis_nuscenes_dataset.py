import os.path

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion


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
    # 获得该scene第一个样本的信息
    first_sample_token = first_scene["first_sample_token"]
    my_sample = nusc.get('sample', first_sample_token)

    my_sample_data = my_sample["data"]  # dict{name:token}
    print(my_sample_data)
    my_sample_anns = my_sample["anns"]  # list[token]

    # print(my_sample_data["CAM_FRONT"])
    # nusc.render_sample_data(my_sample_data["CAM_FRONT"])

    my_sample_data_cam_front = nusc.get('sample_data', my_sample_data["CAM_FRONT"])
    cam_front = cv2.imread(os.path.join(data_root, my_sample_data_cam_front["filename"]))

    my_sample_data_lidar = nusc.get('sample_data', my_sample_data["LIDAR_TOP"])
    pc = LidarPointCloud.from_file(os.path.join(data_root, my_sample_data_lidar["filename"]))

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor', my_sample_data_lidar['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', my_sample_data_lidar['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', my_sample_data_cam_front['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    cs_record = nusc.get('calibrated_sensor', my_sample_data_cam_front['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]
    coloring = depths

    points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    H,W = cam_front.shape[:2]
    min_dist = 0
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < H - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < W - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    cv2.imshow("cam_front", cam_front)
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

    vis_all_image()



if __name__ == "__main__":
    main()
