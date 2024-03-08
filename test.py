import os
import ast
import csv
import time
from typing import Dict
import PIL.Image as Image
from pprint import pprint

import numpy
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchmetrics
from PIL import ImageOps
from torch.nn.functional import one_hot
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from TrafficParser.datasets import FlowPicData
from classifier import FlowPicNet, LeNet
import yaml
# from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import mydataset
from mydataset import ISCX, SimpleDataset
from trainer import Trainer
from utils import load_config_from_yaml, save_config_to_yaml, get_time, device, print_hist, show_hist, int2one_hot
from trainer import get_dataloader_datasetname_numclasses

import torchvision.transforms as trans
from data_augmentations import ChangeRTT, TimeShift, PacketLoss

# global settings
torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_transform():
    # trans.RandomRotation(degrees=[-10, 10])
    transform = trans.Compose([ChangeRTT(), trans.ToTensor()])
    file = '/home/cape/data/trace/new_processed/VoIP_Video_Application_NonVPN/FlowPicOverlapped_train/Buster_VoIP/flowpic-1427915571818-131.202.240.87-32551-131.202.240.86-53753-17-src2dst.npz'
    # file = '/home/cape/data/trace/new_processed/VoIP_Video_Application_NonVPN/FlowPicOverlapped_test/Hangouts_video/flowpic-1427723340243-131.202.240.87-2571-173.194.123.111-443-6-src2dst.npz'
    obj = np.load(file, allow_pickle=True)

    hist = obj['flowpic']
    # 修改包个数，映射到[0, 255],修改数值类型为np.uint8
    hist = map_hist(hist)
    image = Image.fromarray(hist, mode='L')

    # 像素值是正确的，就是Image.Image.show()的颜色显示是反的，无关紧要
    # image.show()
    print(image.getcolors())
    # 数据增强
    hist_ = transform(image)  # [0-255] - [0-1]
    print(hist_.shape)
    # print_hist(hist_)


def test_mytransform():
    # trans.RandomRotation(degrees=[-10, 10])
    # transform = trans.Compose([ChangeRTT(), trans.ToTensor()])
    transform = ChangeRTT(debug=False)
    # file = '/home/cape/data/trace/new_processed/VoIP_Video_Application_NonVPN/FlowPicOverlapped_32_train/Hangouts_VoIP/flowpic-0-10.0.2.15-49421-173.194.123.98-443-6-src2dst.npz'
    file = '/home/cape/data/trace/new_processed/VoIP_Video_Application_NonVPN/FlowPicOverlapped_32_train/Buster_VoIP/flowpic-0-131.202.240.87-17208-131.202.240.242-62838-17-src2dst.npz'
    obj = np.load(file, allow_pickle=True)
    show_hist(obj['flowpic'])
    print(obj['flowpic'].sum())
    transformed_pic = transform(obj)
    show_hist(transformed_pic)
    print(transformed_pic.sum())
    # hist = obj['flowpic']
    # print(hist.sum())
    # print_hist(hist)
    # hist_ = transform(obj)
    # print_hist(hist_)
    # print(hist_.shape)
    # print(hist_.sum())
    pass


def show_npzs(file):
    obj = np.load(file, allow_pickle=True)
    show_hist(obj['flowpic'])
    if 'ChangeRTT' not in file:
        info = (obj['info'].tolist())  # 这样会得到字典-神奇诶
        # print(info)
        ts = info['ts']
        sizes = info['sizes']
        print(sizes)
        # print(list(ts / ts[-1] * 1500))
        # print(type(info))
    else:
        info = obj['info'].tolist()
        # print(info['time_stamps'])
        sizes = info['sizes']
        print(sizes)
        # print(type(info))
    print('*' * 20)


def test_npz_file(mode='info'):
    file = '/home/cape/data/trace/new_processed/VoIP_Video_Application_NonVPN/FlowPicOverlapped_32_train/Hangouts_VoIP/flowpic-0-10.0.2.15-49420-74.125.226.164-443-6-src2dst.npz'
    if mode == 'info':
        a = np.load(file, allow_pickle=True)['info']
        # 转为字典
        a = ast.literal_eval(str(a))
        print(len(a['sizes']))
        print(a.keys())
        print(a['image_dims'])
        print('image_dims' in a.keys())


class test_property:
    def __init__(self):
        pass

    @property
    def name(self):
        return 'test_property'


def test_Img_show_plt_show_flowpic():
    # 这个是测试
    test_array = np.zeros(shape=[1500, 1500])
    for i in range(2690):
        x = np.random.randint(high=1499, low=0)
        y = np.random.randint(high=1499, low=0)
        test_array[x, y] = np.random.randint(high=255, low=0)
        test_array[x, y] = 3
    show_hist(test_array)
    img_test = Image.fromarray(test_array, mode='L')
    # 反色
    im = ImageOps.invert(img_test)
    im.show()


def test_to_tensor():
    a = np.zeros([32, 32])
    return trans.ToTensor()(a)


def test_reshape():
    a = numpy.ones([1500, 1500], int)
    a.reshape(32, 1500 // 32, 32, 1500 // 32).sum(axis=(1, 3))
    print(a)


def test_LeNet_MiniFlowPicNet():
    model = LeNet(num_classes=10)
    img = torch.Tensor(np.random.rand(32, 32)).unsqueeze(0).unsqueeze(0)
    print(model(img))


if __name__ == '__main__':
    print(int2one_hot(4, 5))
