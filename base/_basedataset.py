import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class BaseEIMTCFlowPicDataset(Dataset):
    """
    使用这个类的数据集的基础是有一个train.csv和test.csv文件，文件中有file_path和label_id两列
    """
    def __init__(self, root: str, train: bool = True, transform=None, target_transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if not (len(self.data)):
            self.load_data()
        file_path, label = self.data.iloc[index]
        # load .npz file
        feature = np.load(file_path)['flowpic'].astype(float)  # uint16 -> float64
        feature = torch.FloatTensor(feature)  # dtype: torch.float32
        label = torch.LongTensor([label])  # dtype: torch.int64
        return feature, label

    def load_data(self):
        self.data = pd.read_csv(os.path.join(self.root, 'train.csv')) if self.train else pd.read_csv(
            os.path.join(self.root, 'test.csv'))
