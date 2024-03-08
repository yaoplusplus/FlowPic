"""
在纯净数据集上训练完备的流量分类模型的基础上，通过优化方法对图像添加扰动
1.希望达到混淆图片类别的效果
2.同时获得数据增强函数的参数
"""
import os.path
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Union
import torchsummary
from numpy._typing import NDArray
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from pathlib import Path
from data_augmentations import BasicTransform, ChangeRTT
from utils import device, get_flowpic, int2one_hot
from rich import print as rprint
from augmentation import AugmentationChangeRTT, numpy_to_tensor, tensor_to_numpy


class SeriesDisturber:
    def __init__(
            self,
            model: str,
            ts: NDArray,
            sizes: NDArray,
            **kwargs
    ):
        assert os.path.exists(model)
        assert len(ts) == len(sizes)
        assert len(ts) != 0

        self.ts = ts
        self.sizes = sizes
        self.model_path = Path(model)
        # 函数中设置
        self.factors: NDArray = None
        self.train_factors: Tensor = None
        self.model: nn.Module = None
        self.optim: optim = None
        self.loss: nn.Module = None

    def pre_for_train(self):
        # 模型
        self.model = torch.load(self.model_path, map_location=device)
        self.model.eval()
        # 优化器
        lr = 0.001
        self.factors = np.random.uniform(low=0.5, high=1.5, size=(len(self.sizes)))
        self.train_factors = torch.from_numpy(self.factors)
        self.optim = optim.Adam([self.train_factors], lr=lr)
        # 损失函数
        self.loss = nn.CrossEntropyLoss()

    @staticmethod
    def aug(ts: NDArray, sizes: NDArray, factors: NDArray):
        return ts * factors, sizes

    def __call__(self, *args, **kwargs):
        self.pre_for_train()
        # 根据序列生成flowpic
        flowpic = numpy_to_tensor(get_flowpic(self.ts, self.sizes))
        self.flowpic = flowpic.to(device)
        self.train(target_label=4)

    def train(self, target_label: int):
        epoch = 100
        target_out = int2one_hot(target_label, 10)
        for i in range(1, epoch + 1):
            rprint(f'epoch: {i}')
            ts_, sizes_ = self.aug(self.ts, self.sizes, self.factors)
            # flowpic_ = numpy_to_tensor(get_flowpic(ts_, sizes_))
            # flowpic_ = flowpic_.to(device).unsqueeze(0)
            # out = self.model(self.flowpic)
            # 损
            flowpic_ = torch.rand(1, 1, 32, 32).to(device).type(torch.uint8)
            out_ = self.model(flowpic_)
            loss_classification = self.loss(target_out, out_)
            self.optim.zero_grad()
            self.loss.backward()
            self.optim.step()
            rprint(f'loss: {loss_classification}')
            # 更新参数
            self.factors = self.train_factors.cpu().detach().numpy()


def aug(ts: NDArray, sizes: NDArray, factors: Union[NDArray, torch.Tensor]):
    if type(factors) != NDArray:
        factors = factors.cpu().detach().numpy()
    return ts * factors, sizes


if __name__ == '__main__':
    model_path = '/home/cape/code/FlowPic/checkpoints/MiniFlowPicNet_32-QUIC_DELTA_T=15-IMG_DIM=32-HISTOGRAM=set_bins-Adam---2024-01-10_15-53-49/0.9990-epoch_18.pt'
    file = '/home/cape/data/trace/new_processed/QUIC/pretraining/DELTA_T=15-IMG_DIM=32-HISTOGRAM=set_bins/Google Doc/GoogleDoc-617_0-0.npz'
    flowpic = np.load(file, allow_pickle=True)['info'].tolist()  # get a dict
    ts = np.asanyarray(flowpic['ts'])
    sizes = np.asanyarray(flowpic['sizes'])
    # 加载模型
    model = torch.load(model_path, map_location=device)
    model.eval()
    # 设置优化器
    lr = 0.001
    # 设置需要梯度方便更新
    # train_factors = torch.tensor(np.random.uniform(low=0.5, high=1.5, size=(len(sizes)))).to(
    #     torch.float32).requires_grad_(True).to(device)
    train_factors = torch.tensor(np.random.uniform(low=0.5, high=1.5, size=(len(sizes))), requires_grad=True).to(device)
    # train_factors = torch.clone(train_factors)
    print(f'train_factors.is_leaf: {train_factors.is_leaf}')
    # train_factors = torch.zeros([len(sizes)],requires_grad=True).to(device)
    # train_factors = (torch.rand((len(sizes)), dtype=torch.float32)+0.5).requires_grad_(True)
    optimizer = optim.Adam([train_factors], lr=lr)
    # 损失函数
    loss = nn.CrossEntropyLoss()
    # 目标标签
    target_label = 4
    # 训练轮数
    epoch = 3
    for i in range(1, epoch + 1):
        rprint(f'epoch: {i}')
        ts_, sizes_ = aug(ts, sizes, train_factors)
        # flowpic_ = numpy_to_tensor(get_flowpic(ts_, sizes_))
        # flowpic_ = flowpic_.to(device).unsqueeze(0)
        # out = model(flowpic)
        img = torch.from_numpy(get_flowpic(ts_, sizes_)).unsqueeze(0).unsqueeze(0).to(device)
        target_out = int2one_hot(target_label, 10).repeat(img.shape[0], 1)
        target_out = target_out.to(device)
        out = model(img)
        loss_classification = loss(target_out, out)

        optimizer.zero_grad()
        loss_classification.backward()
        optimizer.step()

        rprint(f'loss: {loss_classification}')
        # 更新参数
        rprint(train_factors.grad)
        print(train_factors)
    pass
