#!/usr/bin/env python
"""
Read traffic_csv
"""

import os
import argparse
import csv
import pickle
import time

import numpy as np
import pandas as pd

from sessions_plotter import *
import glob
import re

from scipy import sparse
from typing import List

os.chdir(os.path.dirname(__file__))  # just for debug

FLAGS = None
INPUT = "../raw_csvs/classes/browsing/reg/CICNTTor_browsing.raw.csv"  # "../mydataset/iscxNTVPN2016/CompletePCAPs" # ""
INPUT_DIR = "../raw_csvs/classes/chat/vpn/"
CLASSES_DIR = "../raw_csvs/classes/**/**/"  # 对应15个类

CLASSES = ['browsing', 'chat', 'file_transfer', 'video', 'voip']
CLASSES_BASE_DIR = "../raw_csvs/classes/"

# LABEL_IND = 1
TPS = 60  # TimePerSession in secs
DELTA_T = 60  # Delta T between splitted sessions
MIN_TPS = 50


# def insert_dataset(mydataset, labels, session, label_ind=LABEL_IND):
#     mydataset.append(session)
#     labels.append(label_ind)

# def export_dataset(mydataset, labels):
#     print("Start export mydataset")
#     np.savez(INPUT.split(".")[0] + ".npz", X=mydataset, Y=labels)
#     print(mydataset.shape, labels.shape)

#
# def import_dataset():
#     print "Import mydataset"
#     mydataset = np.load(INPUT.split(".")[0] + ".npz")
#     print mydataset["X"].shape, mydataset["Y"].shape


def export_dataset_sparse(dataset: List[sparse.csr.csr_matrix]) -> None:
    print("Start export mydataset")
    dirpath = os.path.dirname(INPUT)
    file = dirpath.split('/')[3] + '_' + dirpath.split('/')[4]
    file = os.path.join(dirpath, file)
    print(f'save file: {file}')
    np.save(file, dataset)
    print(f'nums of FlowPic: {len(dataset)}')


def export_dataset(dataset: np.ndarray, npy_filepath):
    print("Start export mydataset")
    print(f'save file: {npy_filepath}')
    np.save(npy_filepath, dataset)
    print(f'nums of FlowPic: {len(dataset)}')


def export_class_dataset(dataset, class_dir):
    print("Start export mydataset")
    npy_filepath = class_dir.split('\\')[-2] + '_' + class_dir.split('\\')[-1] + '.npy'
    np.save(npy_filepath, dataset)
    print(npy_filepath)
    print(dataset.shape)


def import_dataset():
    print("Import mydataset")
    dataset = np.load(os.path.splitext(INPUT)[0] + ".npy")
    print(dataset.shape)
    return dataset


def time_costing(func):
    def core(file_path, test=False):
        start = time.time()
        res = func(file_path, test=test)
        print('time costing:', time.time() - start)
        return res

    return core


def extract_npy_filepath(file):
    """
    从.csv文件路径提取.npy文件名,并返回其路径
    Args:
        file:

    Returns:

    """
    file = os.path.normpath(file)
    dirpath = (os.path.dirname(file))
    npy_filepath = dirpath.split('\\')[-2] + '_' + dirpath.split('\\')[-1] + '.npy'
    return os.path.normpath(os.path.join(dirpath, npy_filepath))


def traffic_csv_converter(file, test=False, save=True):
    if test:
        file = "../raw_csvs/classes/browsing/reg/test.csv"
    # npy_filepath = extract_npy_filepath(file)
    npy_filepath = '/temp'
    if os.path.exists(npy_filepath):
        print(f'find .npy files in {os.path.dirname(file)}')
        dataset = np.load(npy_filepath, allow_pickle=True)
        print(f'load mydataset length: {len(dataset)}')
        return dataset
    print("Running on " + file)
    dataset = []
    # labels = []
    counter = 0
    with open(file, 'r') as csv_file:
        # reader = csv.reader(csv_file)
        reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(reader):
            # print row[0], row[7]
            session_tuple_key = tuple(row[:8])
            length = int(row[7])
            time_stamps = np.array(row[8:8 + length], dtype=float)
            sizes = np.array(row[9 + length:], dtype=int)
            assert len(time_stamps) == len(sizes)
            # if (sizes > MTU).any():
            #     a = [(sizes[i], i) for i in range(len(sizes)) if (np.array(sizes) > MTU)[i]]
            #     print len(a), session_tuple_key

            if length > 10:  # 只考虑数据包个数大于10的会话
            # if length >= 3:  # 针对nprint-applications-case中pcap文件的平均6.5个双向packge而修改
                # print time_stamps[0], time_stamps[-1]
                # h = session_2d_histogram(time_stamps, sizes)
                # session_spectogram(time_stamps, sizes, session_tuple_key[0])
                # mydataset.append([h])
                # counter += 1
                # if counter % 100 == 0:
                #     print counter

                for t in range(
                        int(time_stamps[-1] / DELTA_T - TPS / DELTA_T) + 1):
                    mask = (
                            (time_stamps >= t * DELTA_T) & (
                                time_stamps <= (t * DELTA_T + TPS)))  # 使用TPS将一张图片的数据时间跨度限制在60s,每个t对应一个时长60的片段
                    # print(t * DELTA_T, t * DELTA_T + TPS, time_stamps[-1])
                    ts_mask = time_stamps[mask]
                    sizes_mask = sizes[mask]
                    if len(ts_mask) > 10 and ts_mask[-1] - ts_mask[0] > MIN_TPS:
                    # if len(ts_mask) >= 3 and ts_mask[-1] - ts_mask[0] > MIN_TPS:
                        # if "facebook" in session_tuple_key[0]:
                        #     session_spectogram(time_stamps[mask], sizes[mask], session_tuple_key[0])
                        #     # session_2d_histogram(time_stamps[mask], sizes[mask], True)
                        #     session_histogram(sizes[mask], True)
                        #     exit()
                        # else:
                        #     continue

                        h = ndarray2sparse(
                            session_2d_histogram(ts_mask, sizes_mask, plot=True))
                        # session_spectogram(ts_mask, sizes_mask, session_tuple_key[0])
                        # dataset是sparse_matrix的列表
                        dataset.append([h])
                        counter += 1
                        if counter % 400 == 0:
                            print(counter)
    if save:
        export_dataset(dataset, npy_filepath)
        # return np.asarray(mydataset)  # , np.asarray(labels) # cost too many memory
        return dataset


def traffic_class_converter(dir_path):
    # dataset_tuple = ()
    dataset_tuple = []
    files = [
        os.path.join(dir_path, fn) for fn in next(os.walk(dir_path))[2]
        if (".csv" in os.path.splitext(fn)[-1])
    ]

    for file_path in [
        os.path.join(dir_path, fn) for fn in next(os.walk(dir_path))[2]
        if (".csv" in os.path.splitext(fn)[-1])
    ]:
        dataset_tuple += (traffic_csv_converter(os.path.normpath(file_path)))

    res = np.concatenate(dataset_tuple, axis=0)

    return res


def iterate_all_classes():
    for class_dir in glob.glob(CLASSES_DIR):  # 遍历十五个种类
        if "other" not in class_dir:  # "browsing" not in class_dir and
            print("working on " + class_dir)
            dataset = traffic_class_converter(class_dir)
            print(dataset.shape)
            export_class_dataset(dataset, class_dir)


def random_sampling_dataset(input_array, size=2000):
    print("Import mydataset " + input_array)
    filename = os.path.join(os.path.dirname(input_array), extract_filename(input_array)['filename_split'][0] +
                            '_samp.npy')
    if os.path.exists(filename):
        print(f'find *_samp.npy file')
        return 0

    dataset = np.load(input_array, allow_pickle=True)
    print(len(dataset))
    percentage = size * 1.0 / len(dataset)
    print(f'sample percentage: {percentage}')
    if percentage >= 1:
        raise Exception
    # # this 2 lines below is design for mydataset with type of ndarray[n,1,1500,1500]
    mask = np.random.choice([True, False], len(dataset), p=[percentage, 1 - percentage])
    dataset_samp = dataset[mask]
    # dataset_samp = np.random.choice(a=mydataset,
    #                                 size=size,
    #                                 replace=False,
    #                                 p=[percentage, 1 - percentage])
    print("Start export mydataset")
    np.save(filename, dataset_samp)


def random_sampling_dataset_split(input_array_path, dataset_len, size=2000):
    """
    random_sampling_dataset but the mydataset is stored in 5 splited .npy files
    Args:
        input_array_path: the path to the .npy files mydataset stored
        size:

    Returns:

    """
    p = (size * 1.0 / dataset_len)
    print('sample percentage: ', p)
    assert p < 1, "wrong sample percentage"
    dataset_samp = []
    print("Import mydataset " + input_array_path)
    for file in glob.glob(f'{input_array_path}/*.npy'):
        dataset_split = np.load(file)
        mask = np.random.choice([True, False],
                                len(dataset_split),
                                p=[p, 1 - p])
        dataset_samp.append(dataset_split[mask])
    dataset_samp = np.concatenate(dataset_samp)
    print("Start export mydataset")

    save_path = os.path.join(input_array_path, "browsing_reg_samp.npy")
    np.save(save_path, dataset_samp)


def ndarray2sparse(allmatrix: np.ndarray):
    """
    single to single trans or batch to batch
    """
    allmatrix_sp = sparse.csr_matrix(allmatrix)
    return allmatrix_sp


def sparse2ndarray(sparse_matrix: sparse.csr.csr_matrix):
    return sparse_matrix.toarray()


def test():
    traffic_csv_converter(file='./test1.csv', save=False)
    pass


def extract_filename(input):
    filename = os.path.basename(input)
    filename_split = filename.split('.')
    res = {'filename': filename, 'filename_split': filename_split}
    return res


def main():
    iterate_all_classes()

    # mydataset = traffic_class_converter(INPUT_DIR)

    # test func traffic_csv_converter
    # mydataset = traffic_csv_converter(file_path=INPUT)
    # print(len(mydataset))
    # result: ok

    # input_array_paths = []

    # test
    # input_array_path = "../raw_csvs/classes/browsing/reg/browsing_reg.npy"
    # random_sampling_dataset(input_array_path)
    # result: ok

    # test
    # export_class_dataset(mydataset)
    # result: ok

    # import_dataset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        type=str,
                        default=INPUT,
                        help='Path to csv file')
    FLAGS = parser.parse_args()

    # main()
    test()
