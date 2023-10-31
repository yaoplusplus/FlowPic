import ast
import glob
import os.path
import time
from os import path
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms as trans
import pickle
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot


class FlowPicData(Dataset):
    # train_dataset_count： {0: 4761, 1: 631, 2: 0, 3: 0, 4: 0}
    # test_dataset_count： {0: 49, 1: 5310, 2: 0, 3: 0, 4: 0}
    # val_dataset_count： {0: 536, 1: 64, 2: 0, 3: 0, 4: 0}
    # 我超，数据集有问题啊
    def __init__(self, target: str, flag: str, root='../datasets', num_classes=5, transform=None,
                 target_transform=None):
        """

        Args:
            target: 形如  browsing_vs_all_reg
            flag: 'train','test','val'
            root:path
            transform:
            target_transform:
        """
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.flag = flag
        self.x = np.load(os.path.join(root, target + '_x_' + self.flag + '.npy'), allow_pickle=True)
        self.y = np.load(os.path.join(root, target + '_y_' + self.flag + '.npy'), allow_pickle=True)

        self.num_classes = num_classes
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        feature = self.x[index][0].toarray()
        assert feature.sum()
        feature = torch.FloatTensor(feature.astype(float))  # dtype: torch.float32
        # [self.y[index]] 是float64的一个list
        label = torch.LongTensor([self.y[index]])
        # label = one_hot(torch.LongTensor(label), num_classes=self.num_classes).float()  # label的one_hot应该在loss计算环节
        # feature: torch.float32 torch.Size([1500, 1500])
        # label:   torch.int64   torch.Size([1])
        return feature, label

    def get_flag(self):
        return self.flag


def dataset_count(datasetr):
    count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    dataloader = DataLoader(dataset=d, batch_size=1, shuffle=True)
    for _, label in dataloader:
        count[label[0].item()] += 1
    return count


if __name__ == "__main__":
    # test FlowPicData
    # d = FlowPicData(target='browsing_vs_all_reg', flag='train')
    d = FlowPicData(target='browsing_vs_all_reg', flag='test')
    print(dataset_count(d))
    # result: ok

    pass
