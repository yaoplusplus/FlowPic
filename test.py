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
from classifier import FlowPicNet, FlowPicNet_256, FlowPicNet_256_Reduce
import yaml
# from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

import mydataset
from mydataset import ISCX2016Tor, ISCX, JointISCX, SimpleDataset
from trainer import Trainer
from utils import get_time

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


def test_JointDataset():
    d = JointISCX(dataset='ISCXVPN2016_VPN', feature_method='FlowPic', train=True, root='./dataset/processed')
    loader = DataLoader(d, batch_size=1, shuffle=False)
    for f, l in loader:
        print(f.shape, l)
        exit(0)


if __name__ == '__main__':
    # 
    #     Trainer(r'checkpoints/FlowPicNet(num_classes=9)-ISCXTor2016_EIMTC-Adam-lrscheduler_0.05_20_0.5.yaml',
    #             logdir=f'./log/{get_time()}').train()
    # test_loss_batch('flowpic_data_config.yaml', logdir='')
    # compare_model_params(FlowPicNet().state_dict(), FlowPicNet().state_dict())

    # test_dataset()
    # d = SimpleDataset(dataset='ISCXVPN2016_VPN',feature_method='FlowPic', train=True,root='./dataset/processed')
    test_JointDataset()

    # print(len(d))
    # print(d.name())
    t = Trainer(r'./config.yaml')
    # print(len(t.train_dl))
    t.train()
    pass
