import os

dataset_root_dir = "/media/cjq/新加卷1/datasets/kitti/raw/raw_data"
ros_tools_dir = "/home/cjq/CLionProjects/RosTools/ROS_Tools"


def convert_gnss_pose(dataset):
    read_path = os.path.join(dataset, "oxts/data")
    save_path = os.path.join(dataset, "oxts/poses.txt")

    cmd_str = ros_tools_dir + "/devel/setup.bash && " \
                              "rosrun gps_convert convert_kitti_gps " + read_path + " " + save_path
    os.system(cmd_str)


'''
source /home/cjq/CLionProjects/RosTools/ROS_Tools/devel/setup.bash
'''


def main():
    dataset_list = ['_'.join(name.split('_')[:-1]) for name in os.listdir(dataset_root_dir) if name.endswith("_sync")]
    dataset_list = list(set(dataset_list))

    for dataset in dataset_list:
        # 获取日期
        date = '_'.join(dataset.split('_')[:3])

        dataset_path = os.path.join(dataset_root_dir, dataset + "_sync")
        dataset_path = os.path.join(dataset_path, date, dataset + "_sync")
        # print(dataset_path)

        convert_gnss_pose(dataset_path)


if __name__ == "__main__":
    main()
