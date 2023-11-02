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
from mydataset import ISCX2016Tor, ISCX, JointISCX
from trainer import Trainer
from utils import get_time

# global settings
torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

PATH_PREFIX = "./datasets/"

def csv_cmp(file1,file2):
    f1 = pd.read_csv(file1)
    f2 = pd.read_csv(file2)
    if len(f1) == len(f2):
        cmp_res = f1==f2
    else:
        print('different line numbers')
        return 0
    cmp_res = cmp_res['label'].to_list()
    print(sum(cmp_res))
    print(len(cmp_res))


def test_dataset():
    roots = {'ISCXVPN2016':[r'/home/cape/data/trace/ISCXVPN2016/FlowPic',
             r'/home/cape/data/trace/ISCXVPN2016/MyFlowPic'],
             'ISCXTor2016':[r'/home/cape/data/trace/ISCXTor2016/FlowPic',r'/home/cape/data/trace/ISCXTor2016/MyFlowPic'],
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

def FlowPic2MyFlowPic():
    file = './dataset/ISCXTor2016/FlowPic/nonTor/Browsing/flowpic-1438105370199-10.152.152.11-60375-216.58.210.35-443-6-dst2src.npz'
    flowpic = np.load(file)['flowpic']
    print(np.nonzero(flowpic))
    for loc in zip(np.nonzero(flowpic)):
        print(loc)
        x = loc[0]
        y= loc [1]
        print(flowpic[x][y])
        flowpic[x][y] = flowpic[x][y]*y
        print(flowpic[x][y])
    np.savez_compressed(save_path='./test/test.npz',flowpic=flowpic)
    pass

if __name__ == '__main__':
    # 
    #     Trainer(r'checkpoints/FlowPicNet(num_classes=9)-ISCXTor2016_EIMTC-Adam-lrscheduler_0.05_20_0.5.yaml',
    #             logdir=f'./log/{get_time()}').train()
    # test_loss_batch('flowpic_data_config.yaml', logdir='')
    # compare_model_params(FlowPicNet().state_dict(), FlowPicNet().state_dict())
    # d = ISCX(root='/home/cape/data/trace/ISCXVPN2016/MyFlowPic', train=True, flag=True, part=None)
    # tor:      resplit(same metadata.csv but file_path)
    # nonTor:   their metadata.csv has different lines
    # VPN:      all fine
    # csv_cmp(os.path.join(base,'ISCXVPN2016','FlowPic','VPN','train.csv'),os.path.join(base,'ISCXVPN2016','MyFlowPic','VPN','train.csv'))

    # test_dataset()
    # FlowPic2MyFlowPic()
    print(len(os.listdir('./dataset/ISCXTor2016/FlowPic/tor/audio')))
    pass
