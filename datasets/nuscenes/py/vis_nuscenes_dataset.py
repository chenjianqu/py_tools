import os.path

import cv2
from nuscenes.nuscenes import NuScenes


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
