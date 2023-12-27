import glob
import os

import PIL.Image as Image
import torch
import numpy as np
import pandas as pd
from typing import List, Optional, Callable

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import data_augmentations
from base._basedataset import BaseEIMTCFlowPicDataset
from utils import get_num_classes


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
        files_or_dirs = glob.glob(os.path.join(self.root, self.feature_method, self.flag, '*'))
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
    def __init__(self, dataset: str, feature_method: str, train: bool, root, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = torch.Tensor, custom_csv: str = None):
        """
        dataset: ISCXTor2016_tor, ISCXTor2016_nonTor, ISCXVPN2016_VPN
        feature_method : FlowPic, MyFlowPic, Joint, JointFeature
        train\test_csv_custom : 指定train.csv 或者 test.csv 例如 train-ChangeRTT.csv test.csv
        """
        self.dataset = dataset
        self.feature_method = feature_method
        self.train = train
        # 是否分别从两个文件夹的csv文件读取数据的标志位
        self.part_csv = True if dataset == 'VoIP_Video_Application_NonVPN' and 'FlowPicOverlapped' in feature_method else False
        if dataset != 'Video_VoIP_splited':
            self.root = os.path.join(root, self.dataset, self.feature_method)
        else:
            self.suffix = 'train' if self.train else 'test'
            self.root = os.path.join(root, self.dataset, self.feature_method, self.suffix)
        if self.part_csv:
            self.root = self.root + '_' + 'train' if self.train else self.root + '_' + 'test'
        self.custom_csv = custom_csv
        self.load_data()
        self.num_classes = get_num_classes(self.root)  # 因为csv分散到两个文件夹了，此处逻辑伴随改变
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file_path, label = self.data.iloc[index]
        file_path = os.path.join(self.root, file_path)
        key = 'feature' if 'JointFeature' in self.feature_method else 'flowpic'

        if self.train and self.transform is not None:  # 至少有一个是ToTensor
            # 自定义的transform，输入是.npz加载的对象，含有flowpic和info两部分
            if hasattr(self.transform, 'name') and self.transform.name in ['ChangeRTT', 'TimeShift', 'PacketLoss']:
                feature = self.transform(np.load(file_path, allow_pickle=True))
            else:
                # 内置的transform，输入是PIL.Image.Image
                feature = np.load(file_path)[key].astype(np.float32)  # uint16 -> np.float32
                feature = torch.Tensor(feature)
                feature = feature.unsqueeze(0) # ColorJitter要求输入形状为(num_channels,H,W）
                feature = self.transform(feature)
        else:
            feature = np.load(file_path, allow_pickle=True)[key].astype(np.float32)
        # if self.target_transform:
        # label = np.array([[label]])
        # label = self.target_transform(label)

        # feature = torch.FloatTensor(feature)  # dtype: torch.float32 # 这个放到transform里
        label = self.target_transform([label])  # dtype: torch.int64 # TODO 这个放到transform里

        return feature, label

    def name(self):
        return '_'.join([self.dataset, self.feature_method])

    def load_data(self):
        if self.custom_csv:
            file = self.custom_csv
        else:
            file = 'train.csv' if self.train else 'test.csv'
        self.data = pd.read_csv(os.path.join(self.root, file))


class SimpleSplitDataset(Dataset):
    # like── Video_VoIP_splited          -dataset
    #     ├── DELTA_T=15-IMG_DIM=1500    -feature_method
    #     │      ├── test
    #     │      │      ├── buster_voip  -class
    def __init__(self, dataset: str, feature_method: str, train: bool, root, transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, train_csv_custom: str = None,
                 test_csv_custom: str = None):
        """
        dataset: ISCXTor2016_tor, ISCXTor2016_nonTor, ISCXVPN2016_VPN
        feature_method : FlowPic, MyFlowPic, Joint, JointFeature
        train\test_csv_custom : 指定train.csv 或者 test.csv 例如 train-ChangeRTT.csv test.csv
        """
        self.dataset = dataset
        self.feature_method = feature_method
        self.train = train
        self.subdir = 'train' if self.train else 'test'
        self.root = os.path.join(root, self.dataset, self.feature_method, self.subdir)
        self.train_csv_custom = train_csv_custom
        self.test_csv_custom = test_csv_custom
        self.load_data()
        self.num_classes = get_num_classes(self.root)  # 因为csv分散到两个文件夹了，此处逻辑伴随改变
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file_path, label = self.data.iloc[index]
        file_path = os.path.join(self.root, file_path)
        key = 'feature' if 'JointFeature' in self.feature_method else 'flowpic'

        if self.transform is not None and self.train:  # 至少有一个是ToTensor
            # 自定义的transform，输入是.npz加载的对象，含有flowpic和info两部分
            if self.transform.name in ['ChangeRTT', 'TimeShift', 'PacketLoss']:
                feature = self.transform(np.load(file_path, allow_pickle=True))
            else:
                # 内置的transform，输入是PIL.Image.Image
                feature_array = np.load(file_path)[key].astype(np.float32)  # uint16 -> np.float32
                feature_PILImg = Image.fromarray(feature_array)
                feature = self.transform(feature_PILImg)
        else:
            feature = np.load(file_path, allow_pickle=True)[key].astype(np.float32)
        # if self.target_transform:
        # label = np.array([[label]])
        # label = self.target_transform(label)

        # feature = torch.FloatTensor(feature)  # dtype: torch.float32 # 这个放到transform里
        label = torch.LongTensor([label])  # dtype: torch.int64 # TODO 这个放到transform里

        return feature, label

    def name(self):
        return '_'.join([self.dataset, self.feature_method])

    def load_data(self):
        #     file = 'train.csv' if self.train else 'test.csv'
        # self.data = pd.read_csv(os.path.join(self.root, file))
        pass


class MultiDataset(Dataset):

    def __init__(self, root: str, dataset: str, train: bool, features: List = ['FlowPic', 'MyFlowPic']):
        self.dataset = dataset
        self.features = features
        self.roots = [os.path.join(root, self.dataset, self.features[0]),
                      os.path.join(root, self.dataset, self.features[1])]
        self.train = train
        self.load_data()
        self.num_classes = get_num_classes(self.roots[0])
        assert os.path.exists(os.path.exists())

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        # 根据两个数据集的.csv文件来加载就行
        file_paths, label = self.data[0].iloc[index]
        file_path, label = self.data[1].iloc[index]

        key = 'feature' if 'JointFeature' in self.feature_method else 'flowpic'
        feature = np.load(file_path)[key].astype(float)  # uint16 -> float64
        feature = torch.FloatTensor(feature)  # dtype: torch.float32
        label = torch.LongTensor([label])  # dtype: torch.int64
        return feature, label

    def name(self):
        return 'MultiDataset'

    def load_data(self):
        file = 'train.csv' if self.train else 'test.csv'
        self.data = [pd.read_csv(os.path.join(self.roots[0], file)),
                     pd.read_csv(os.path.join(self.roots[1], file))]
        assert len(self.data[0]) == len(self.data[1])
