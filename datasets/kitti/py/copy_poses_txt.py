import os
import shutil

"""
复制pose.txt到新的目录
"""

src_root_dir = "/media/cjq/新加卷2/datasets/kitti/raw/raw_data"
dst_root_dir = "/media/cjq/新加卷2/datasets/kitti/raw/pose_data"


def main():
    dataset_list = ['_'.join(name.split('_')[:-1]) for name in os.listdir(src_root_dir) if name.endswith("_sync")]
    dataset_list = list(set(dataset_list))
    for dataset in dataset_list:
        # 获取日期
        date = '_'.join(dataset.split('_')[:3])

        dataset_path = dataset + "_sync"
        dataset_path = os.path.join(dataset_path, date, dataset + "_sync")
        # print(dataset_path)
        pose_path = os.path.join(dataset_path, "oxts", "poses.txt")

        src_dst = os.path.join(src_root_dir, pose_path)
        dst_path = os.path.join(dst_root_dir, pose_path)
        print(dst_path)

        if not os.path.exists(os.path.dirname(dst_path)):
            os.makedirs(os.path.dirname(dst_path))
        shutil.copy(src_dst, dst_path)


if __name__ == "__main__":
    main()
