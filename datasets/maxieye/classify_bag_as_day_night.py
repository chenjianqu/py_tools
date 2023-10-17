import json
import os
import shutil
import time
from pathlib import Path

"""
在服务器上有很多个数据包。本脚本用于统计每个数据包的信息，包括时间戳、图像数量等。
"""

def traverse(sub_dir):
    datasets = []
    for date_dir in os.listdir(sub_dir):
        date_path = os.path.join(sub_dir, date_dir)
        for dataset in os.listdir(date_path):
            datasets.append(os.path.join(date_dir, dataset))
    return datasets


def traverse_sub_dir(root_dir):
    sub_dir_list = ["bev_gnd", "bev_gnd_fucai"]

    datasets_all = []
    for sub_dir in sub_dir_list:
        sub_path = os.path.join(root_dir, sub_dir)
        datasets = traverse(sub_path)
        datasets_path = [os.path.join(sub_dir, dataset) for dataset in datasets]
        datasets_all = datasets_all + datasets_path
    return datasets_all


def get_bag_time(dataset_path):
    """
    统计每个bag录制的时间，并返回bag中图像的数量
    :param dataset_path:
    :return:
    """
    images_path = os.path.join(dataset_path, "org", "img_ori")
    if not os.path.exists(images_path):
        return "","", 0
    img_names = os.listdir(images_path)
    img_names.sort()
    if len(img_names) < 10:  # 放弃
        return "","", len(img_names)
    # 获取开始时间戳,Unix时间戳是一种表示时间的方式，它是从1970年1月1日午夜（UTC）开始的秒数。
    img_start_time = float(Path(img_names[0]).stem)

    # 使用gmtime函数将Unix时间戳转换为时间结构体
    time_struct = time.gmtime(img_start_time)
    # 使用strftime函数将时间结构体格式化为可读日期字符串（YYYY-MM-DD HH:MM:SS）
    date_string = time.strftime("%Y-%m-%d %H:%M:%S", time_struct)
    hour = time.strftime("%H", time_struct)  # 小时
    return date_string, hour, len(img_names)


def get_bag_one_image(dataset_path, save_path):
    images_path = os.path.join(dataset_path, "org", "img_ori")
    if not os.path.exists(images_path):
        return "", 0
    img_names = os.listdir(images_path)
    img_names.sort()
    if len(img_names) < 10:  # 放弃
        return "", len(img_names)
    src_path = os.path.join(images_path, img_names[0])
    dst_path = os.path.join(save_path, os.path.basename(dataset_path) + ".jpg")
    shutil.copyfile(src_path, dst_path)


def main():
    data_root = "/defaultShare/aishare/share"
    lidar_data_root = "/defaultShare/aishare/share/occ_data/occ_data_ori"

    datasets_org = traverse_sub_dir(data_root)
    datasets_occ = traverse_sub_dir(lidar_data_root)

    datasets_final = list(set(datasets_org) & set(datasets_occ))  # 交集

    day_list = []
    night_list = []
    bag_info = {}
    image_sample = "image_sample"
    os.makedirs(image_sample,exist_ok=True)

    day_num_image = 0
    night_num_image = 0

    for dataset in datasets_final:
        date_string, hour, num_image = get_bag_time(os.path.join(data_root, dataset))
        if date_string != "":
            hour = int(hour)
            if 6 <= hour <= 18:
                day_night = "day"
                day_list.append(dataset)
                day_num_image += num_image
            else:
                day_night = "night"
                night_list.append(dataset)
                night_num_image += num_image

            bag_info[dataset] = {
                "day_night": day_night,
                "date":date_string,
                "image_num": num_image
            }

            # 复制一个图像
            get_bag_one_image(os.path.join(data_root, dataset), image_sample)
            print(dataset + " " + day_night)

    print(f"day_num_image:{day_num_image} night_num_image:{night_num_image}")

    # 写入txt
    day_list.sort()
    with open("day_list.txt", "w") as f:
        for dataset in day_list:
            f.write(dataset + "\n")

    night_list.sort()
    with open("night_list.txt", "w") as f:
        for dataset in night_list:
            f.write(dataset + "\n")

    with open('./bag_info.json', 'w', encoding='utf-8') as fp:
        json.dump(bag_info, fp, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
