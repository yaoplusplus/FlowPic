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
    hist = np.histogram2d(data.arrival_time, data.packet_size,
                          bins=hist_bins, range=bin_range)
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
        return len(self.dataset_a.data)

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
            self.data = self.data.drop(
                self.data[self.data['label'] == drop_class].index)

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
        feature = np.load(file_path)['flowpic'].astype(
            float)  # uint16 -> float64
        feature = torch.FloatTensor(feature)  # dtype: torch.float32
        label = torch.LongTensor([label])  # dtype: torch.int64
        return feature, label

    def drop(self):
        assert len(self.data)
        for class_ in self.classes:
            if class_ not in self.part:
                self.drop_classes.append(class_)
        for drop_class in self.drop_classes:
            self.data = self.data.drop(
                self.data[self.data['label'] == drop_class].index)

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

    def __init__(self, dataset: str, feature_method: str, flag: bool, train: bool, root='../dataset',
                 part: List = None):
        # dataset指向以数据集为名称的文件夹，feature指代FlowPic或者MyFlowPic或其他方法，也就是子文件夹名,flag指向nonTor/tor、VPN/nonVPN
        self.feature_method = feature_method
        super(ISCX, self).__init__(os.path.join(
            root, self.feature_method), train)
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
        if os.name == 'posix':
            file_path = file_path.replace(
                'D:\\data\\trace\\processed', '/home/cape/data/trace')
            file_path = file_path.replace('\\', '/')
        # load .npz file
        feature = np.load(file_path)['flowpic'].astype(
            float)  # uint16 -> float64
        feature = torch.FloatTensor(feature)  # dtype: torch.float32
        label = torch.LongTensor([label])  # dtype: torch.int64
        return feature, label

    def drop(self):
        assert len(self.data)
        for class_ in range(self.get_num_classes()):
            if class_ not in self.part:
                self.drop_classes.append(class_)
        for drop_class in self.drop_classes:
            self.data = self.data.drop(
                self.data[self.data['label'] == drop_class].index)

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
        files_or_dirs = glob.glob(os.path.join(
            self.root, self.feature_method, self.flag, '*'))
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


class SimpleDataset(Dataset):
    def __init__(self, dataset: str, feature_method: str, train: bool, root):
        """
        dataset: ISCXTor2016_tor, ISCXTor2016_nonTor, ISCXVPN2016_VPN
        feature_method : FlowPic, MyFlowPic, Joint, JointFeature
        """
        self.dataset = dataset
        self.feature_method = feature_method
        self.root = os.path.join(root, self.dataset, self.feature_method)
        self.train = train
        self.data = self.load_data()
        self.num_classes = self.get_num_classes()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file_path, label = self.data.iloc[index]
        key = 'feature' if self.feature_method == 'JointFeature' else 'flowpic'
        feature = np.load(file_path)[key].astype(float)  # uint16 -> float64
        feature = torch.FloatTensor(feature)  # dtype: torch.float32
        label = torch.LongTensor([label])  # dtype: torch.int64
        return feature, label

    def name(self):
        return '_'.join([self.dataset, self.feature_method])

    def get_num_classes(self):
        dirs_count = 0
        files_or_dirs = glob.glob(os.path.join(self.root, '*'))
        for file_or_dir in files_or_dirs:
            if os.path.isdir(file_or_dir):
                dirs_count += 1
        return dirs_count

    def load_data(self):
        # metadata = pd.read_csv(os.path.join(self.root, 'metadata.csv'))
        # train_data = pd.read_csv(os.path.join(self.root, 'train.csv'))
        # test_data = pd.read_csv(os.path.join(self.root, 'test.csv'))
        # assert (len(train_data) + len(test_data)) == len(metadata), '训练/测试数据总数与metadata不符'
        file = 'train.csv' if self.train else 'test.csv'
        return pd.read_csv(os.path.join(self.root, file))


class MultiFeatureISCX:
    def __init__(self, dataset: str, train: bool, feature=['FlowPic', 'MyFlowPic']):
        self.dataset = dataset
        self.train = train
        self.feature = feature

        self.load_dataset()
        assert self.dataset_a.get_num_classes() == self.dataset_b.get_num_classes()
        assert len(self.dataset_a) == len(self.dataset_b)
        self.num_classes = self.dataset_a.get_num_classes()

    def load_dataset(self):
        # 横跨FlowPic和MyFlowPic
        self.dataset_a = SimpleDataset(
            root='../dataset/processed/', dataset=self.dataset, train=self.train, feature_method=self.feature[0])
        self.dataset_b = SimpleDataset(
            root='../dataset/processed/', dataset=self.dataset, train=self.train, feature_method=self.feature[1])

    def __len__(self):
        return len(self.dataset_a)

    def __getitem__(self, index):
        # shuffle应该是false?不然两个数据集在一个index下，访问的数据可能不一样
        flowpic_a, label = self.dataset_a.__getitem__(index)
        flowpic_b, label_ = self.dataset_b.__getitem__(index)
        # 这里转置是因为数据集的X，y坐标
        # print(np.nonzero(flowpic_a.T))
        # print(np.nonzero(flowpic_b))
        assert torch.equal(np.nonzero(flowpic_a.T), np.nonzero(flowpic_b))
        assert label == label_
        return torch.concat([flowpic_a, flowpic_b], dim=0), label

    def name(self):
        return 'MultiFeature' + '_' + self.dataset_a.name()

    def get_num_classes(self):
        return self.num_classes
