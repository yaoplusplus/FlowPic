import os
import time
from typing import Dict
from pprint import pprint

import numpy as np

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
from mydataset import ISCX2016Tor, ISCX
from trainer import Trainer
from utils import get_time

# global settings
torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

PATH_PREFIX = "./datasets/"

if __name__ == '__main__':
#     Trainer(r'checkpoints/FlowPicNet(num_classes=9)-ISCXTor2016_EIMTC-Adam-lrscheduler_0.05_20_0.5.yaml',
#             logdir=f'./log/{get_time()}').train()
    # test_loss_batch('flowpic_data_config.yaml', logdir='')
    # compare_model_params(FlowPicNet().state_dict(), FlowPicNet().state_dict())
    d = ISCX(root='/home/cape/data/trace/ISCXVPN2016/MyFlowPic', train=True, flag=True, part=None)
    # train_loader = DataLoader(dataset=d, batch_size=1, shuffle=True)
    # count = 0
    print(d.get_num_classes())  # 1208
    print(len(d.data))  # 1208
    print(d.name())
    pass
