import glob
import os
import torch
import numpy as np
import pandas as pd
from typing import List

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from base._basedataset import BaseEIMTCFlowPicDataset


def construct_matrix(data: pd.DataFrame, block_size=5, resolution=256):
    # normalize packet time
    data.arrival_time = data.arrival_time - data.arrival_time.min()
    time_scale = 3000
    packet_range = [-1500, 1500]
    data.arrival_time = data.arrival_time / (block_size * 1000) * time_scale
    # construct histogram
    hist_bins = resolution
    bin_range = [[0, time_scale], packet_range]
    hist = np.histogram2d(data.arrival_time, data.packet_size, bins=hist_bins, range=bin_range)
    flow_matrix = hist[0]
    assert flow_matrix.max() > 0.0, 'Zero Matrix!'
    flow_matrix = flow_matrix / flow_matrix.max()
    flow_matrix = flow_matrix.astype(np.float32)
    return flow_matrix


class ISCX2016Tor(Dataset):
    """
    train_set: {0: 528, 1: 1844, 2: 287, 3: 1308, 4: 383, 5: 1694, 6: 1184, 7: 3572}
    test_set:  {0: 133, 1: 461, 2: 72, 3: 328, 4: 96, 5: 424, 6: 297, 7: 893}
    """

    def __init__(self, root=r'D:\data\trace\ISCXTor2016\tor\csv', flag: str = 'train', num_classes: int = 8,
                 tor: bool = False, part: List = None, transform=None,
                 target_transform=None):
        """
        Args:
            flag: 'train','test','val'
            root: path 2 dataset dir
            transform:
            target_transform:
        """
        self.root = root
        self.flag = flag
        self.num_classes = num_classes
        self.transform = transform
        self.target_transform = target_transform
        self.transform = transform
        self.target_transform = target_transform
        self.data = pd.read_csv(os.path.join(self.root, f'{self.flag}.csv'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file_path, label = self.data.iloc[index]
        feature = pd.read_csv(file_path)
        feature = construct_matrix(feature)
        feature = torch.FloatTensor(feature)  # dtype: torch.float32
        label = torch.LongTensor([label])  # dtype: torch.int64
        return feature, label


class PartISCX2016Tor(Dataset):
    """
   选择了self.data中的指定label的数据、设置了label重映射
    """

    def __init__(self, root=r'D:\data\trace\ISCXTor2016\tor\csv', flag: str = 'train', transform=None,
                 target_transform=None):
        """

        Args:
            flag: 'train','test','val'
            root:path
            transform:
            target_transform:
        """
        self.saved_classes = [1, 3, 5, 6]
        self.drop_classes = [0, 2, 4, 7]
        self.map = {1: 0, 3: 1, 5: 2, 6: 3}
        self.root = root
        self.flag = flag
        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = 4
        self.transform = transform
        self.target_transform = target_transform
        self.data = pd.read_csv(os.path.join(self.root, f'{self.flag}.csv'))
        # drop class
        for drop_class in self.drop_classes:
            self.data = self.data.drop(self.data[self.data['label'] == drop_class].index)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file_path, label = self.data.iloc[index]
        feature = pd.read_csv(file_path)
        feature = construct_matrix(feature)
        feature = torch.FloatTensor(feature)  # dtype: torch.float32
        # remaping [1, 3, 5, 6] -> [0,1,2,3]
        label = self.map[label]
        label = torch.LongTensor([label])  # dtype: torch.int64
        return feature, label


class ISCXTor2016EIMTC(BaseEIMTCFlowPicDataset):
    """
    tor: 使用ISCXTor2016/nonTor或者ISCXTor2016/tor
    part：形如[0,4,6]，由于数据集不均匀，是否使 用较均匀的类别来训练模型，0、4、6代表label_id
    """

    def __init__(self, root, train, tor: bool = False, part: List = None):
        super(ISCXTor2016EIMTC, self).__init__(root, train)
        self.tor = tor
        self.classes = [0, 1, 2, 3, 4, 5, 6]
        self.data = self.load_data()
        if part is not None:
            # 丢弃不要的类
            self.part = part
            self.drop_classes = []
            self.drop()
            self.map = {}
            self.set_map()

    def __getitem__(self, index):
        file_path, label = self.data.iloc[index]
        if hasattr(self, 'part'):
            label = self.map[label]
        # load .npz file
        feature = np.load(file_path)['flowpic'].astype(float)  # uint16 -> float64
        feature = torch.FloatTensor(feature)  # dtype: torch.float32
        label = torch.LongTensor([label])  # dtype: torch.int64
        return feature, label

    def drop(self):
        assert len(self.data)
        for class_ in self.classes:
            if class_ not in self.part:
                self.drop_classes.append(class_)
        for drop_class in self.drop_classes:
            self.data = self.data.drop(self.data[self.data['label'] == drop_class].index)

    def set_map(self):
        index = 0
        for class_ in self.part:
            self.map[class_] = index
            index += 1

    def load_data(self):
        tor = 'tor' if self.tor else 'nonTor'
        file = 'train.csv' if self.train else 'test.csv'
        return pd.read_csv(os.path.join(self.root, tor, file))


class ISCX(BaseEIMTCFlowPicDataset):
    """
        flag: 对于ISCXTor2016，意味着tor或者nonTor，对于ISCXVPN2016，意味着VPM或者NonVPN
        part：形如[0,4,6]，只使用数据集中label_id为0，4，6的数据，其余的数据将会在load_data()
        之后被丢弃，label_id也会进行重新映射
        """

    def __init__(self, root, train, flag: bool = False, part: List = None):
        super(ISCX, self).__init__(root, train)
        self.flag = flag
        self.data = self.load_data()
        self.num_classes = self.get_num_classes()
        if part is not None:
            # 丢弃不要的类
            self.part = part
            self.drop_classes = []
            self.drop()
            self.map = {}
            self.set_map()
            # 提供参数给模型初始化
            self.num_classes = len(self.part)

    def __getitem__(self, index):
        file_path, label = self.data.iloc[index]
        if hasattr(self, 'part'):
            label = self.map[label]
        # 将windows下的路径转换为linux下的
        if os.name =='posix':
            file_path = file_path.replace('D:\\data\\trace\\processed','/home/cape/data/trace')
            file_path = file_path.replace('\\','/')
        # load .npz file
        feature = np.load(file_path)['flowpic'].astype(float)  # uint16 -> float64
        feature = torch.FloatTensor(feature)  # dtype: torch.float32
        label = torch.LongTensor([label])  # dtype: torch.int64
        return feature, label

    def drop(self):
        assert len(self.data)
        for class_ in range(self.get_num_classes()):
            if class_ not in self.part:
                self.drop_classes.append(class_)
        for drop_class in self.drop_classes:
            self.data = self.data.drop(self.data[self.data['label'] == drop_class].index)

    def set_map(self):
        index = 0
        for class_ in self.part:
            self.map[class_] = index
            index += 1

    def load_data(self):
        if 'VPN'.lower() in self.root.lower():
            self.flag = 'VPN' if self.flag else 'NonVPN'
        elif 'tor'.lower() in self.root.lower():
            self.flag = 'tor' if self.flag else 'nonTor'
        file = 'train.csv' if self.train else 'test.csv'
        return pd.read_csv(os.path.join(self.root, self.flag, file))

    def get_num_classes(self):
        dirs_count = 0
        files_or_dirs = glob.glob(os.path.join(self.root, self.flag, '*'))
        for file_or_dir in files_or_dirs:
            if os.path.isdir(file_or_dir):
                dirs_count += 1
        return dirs_count

    def name(self):
        """

        Returns: 便于保存记录

        """
        base = 'ISCX'
        if 'VPN' in self.root:
            # 这里的flag已经是字符串了
            subclass = 'VPN2016_' + self.flag
        else:
            subclass = 'Tor2016_' + self.flag
        part = str(self.part) if hasattr(self, 'part') else ''
        return base + subclass + part


if __name__ == "__main__":
    # d = ISCXTorData(flag='train')
    # train_loader = DataLoader(dataset=d, batch_size=1, shuffle=True)
    # for x, y in train_loader:
    #     print(x.shape, y.shape)
    #     print(y)
    #     exit(0)

    d = ISCX(root='/home/cape/data/trace/ISCXTor2016/MyFlowPic', train=True, flag=False, part=None)
    # train_loader = DataLoader(dataset=d, batch_size=1, shuffle=True)
    # count = 0
    print(d.get_num_classes())  # 1208
    print(len(d.data))  # 1208
    print(d.name())
    pass
