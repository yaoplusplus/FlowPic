#!/usr/bin/env python
"""
datasets_generator.py creates final class_vs_all mydataset ready to be inserted to machine.
The input for this module are pre-created numpy array containing all classes session 2d_histograms created in traffic_csv_conveter.py
"""
import glob
import os
import numpy as np
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

CLASS = "browsing"
TEST_SIZE = 0.1
DATASET_DIR = "../dataset/" # TODO change

VPN_TYPES = {
    "reg": glob.glob("../raw_csvs/classes/**/reg/*.npy"), # means regular
    "vpn": glob.glob("../raw_csvs/classes/**/vpn/*.npy"),
    "tor": glob.glob("../raw_csvs/classes/**/tor/*.npy")
}


def import_array(input_array):
    print("Import mydataset " + input_array)
    dataset = np.load(input_array, allow_pickle=True)
    # print(mydataset.shape)
    return dataset


def export_dataset(dataset_dict, file_path):
    # with open(file_path + ".pkl", 'wb') as outfile:
    #     pickle.dump(dataset_list, outfile, pickle.HIGHEST_PROTOCOL)
    for name, array in dataset_dict.items():
        np.save(file_path + "_" + name, array)


def create_class_vs_all_specific_vpn_type_dataset(class_name, vpn_type="reg", validation=False, ratio=1.2):
    print(f'working on class: {class_name}, vpn_type: {vpn_type}, validation: {validation}, radio: {ratio}')
    class_array_file = [fn for fn in VPN_TYPES[vpn_type] if class_name in fn and "overlap" not in fn][0]
    print(f'class_array_file: {class_array_file:>10}')
    all_files = [fn for fn in VPN_TYPES[vpn_type] if class_name not in fn and "overlap" not in fn]
    print(f'all_files: {all_files}')

    class_array = import_array(class_array_file)
    count = len(class_array)
    print(f'count of class image: {count}')

    all_count = len(all_files)
    # 用指定类的所有样本数量/其余类别的文件的数目，意思是其余类.npy文件中应该提取的样本的数量,这个比例为什么是1.2呢？
    count_per_class = ratio * count / all_count
    print(f'count of other per class: {count_per_class}')
    # 从各其余类的.npy文件中提取image
    for fn in all_files:
        print(fn)
        fn_array = import_array(fn)
        p = count_per_class * 1.0 / len(fn_array)
        print(p)
        if p < 1:
            mask = np.random.choice([True, False], len(fn_array), p=[p, 1 - p])
            fn_array = fn_array[mask]

        print(len(fn_array))
        class_array = np.append(class_array, fn_array, axis=0)
        print(len(class_array))
        del fn_array

    labels = np.append(np.zeros(count), np.ones(len(class_array) - count))
    print(len(class_array), len(labels), labels[0], labels[count - 1], labels[count], labels[-1])
    dataset_dict = dict()

    if validation:
        x_train, x_val, y_train, y_val = train_test_split(class_array, labels, test_size=TEST_SIZE)
        print(len(y_train), sum(y_train), 1.0 * sum(y_train) / len(y_train))
        print(len(y_val), sum(y_val), 1.0 * sum(y_val) / len(y_val))

        dataset_dict["x_train"] = x_train
        dataset_dict["x_val"] = x_val
        dataset_dict["y_train"] = y_train
        dataset_dict["y_val"] = y_val
    else:
        dataset_dict["x_test"] = class_array
        dataset_dict["y_test"] = labels

    export_dataset(dataset_dict, DATASET_DIR + class_name + "_vs_all_" + vpn_type)


if __name__ == '__main__':
    # create_class_vs_all_specific_vpn_type_dataset(CLASS, validation=True)
    # create_class_vs_all_specific_vpn_type_dataset(CLASS, vpn_type="vpn", validation=False)
    # create_class_vs_all_specific_vpn_type_dataset(CLASS, vpn_type="tor", validation=False)
    create_class_vs_all_specific_vpn_type_dataset(CLASS, vpn_type="reg", validation=True)

    # print(os.getcwd())
    # os.chdir('D:\code\internet_flow\FlowPic\TrafficParser')
    # print(os.getcwd())
