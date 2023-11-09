import os
import time
from typing import Dict
from pprint import pprint

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchmetrics
from torch.nn.functional import one_hot
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from TrafficParser.datasets import FlowPicData
from classifier import FlowPicNet
import yaml
# from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import mydataset
from mydataset import ISCX2016Tor, ISCX, SimpleDataset
from trainer import Trainer
from utils import get_time, get_dataloader_datasetname_numclasses

# global settings
torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_dataset():
    roots = {'ISCXVPN2016': [r'/home/cape/data/trace/ISCXVPN2016/FlowPic',
                             r'/home/cape/data/trace/ISCXVPN2016/MyFlowPic'],
             'ISCXTor2016': [r'/home/cape/data/trace/ISCXTor2016/FlowPic',
                             r'/home/cape/data/trace/ISCXTor2016/MyFlowPic'],
             }
    d = JointISCX(roots=roots['ISCXVPN2016'], train=True, flag=True)

    # count = 0
    print(d.get_num_classes())  # 1208
    print(len(d))  # 1208
    print(d.name())

    train_loader = DataLoader(dataset=d, batch_size=1, shuffle=False)
    flowpic, label = next(iter(train_loader))
    print(flowpic.shape)
    print(label)


if __name__ == '__main__':

    ##### test model
    # model_pt = '/home/cape/code/FlowPic/dataset/processed/ISCXTor2016_tor/JointFeature_trained_tor_model/video/flowpic-1436962234252-10.0.2.15-41071-198.52.200.39-443-6-src2dst.npz'
    # model = torch.load(model_pt)
    # print(model)
    ##### test model

    ##### test .npz file
    file = '/home/cape/code/FlowPic/dataset/processed/ISCXTor2016_tor/JointFeature_trained_tor_model/audio/flowpic-1437494738355-10.0.2.15-57188-82.161.239.177-110-6-src2dst.npz'
    a = np.load(file)['feature']
    print(a.shape)
    ##### test .npz file
    pass
