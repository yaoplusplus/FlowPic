import glob
import os
import time
from typing import List

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm

from classifier import FlowPicNet
from mydataset import SimpleDataset
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_config_from_yaml(file_path):
    with open(file_path, 'r', encoding='UTF-8') as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return config


def save_config_to_yaml(config, file_path):
    with open(file_path, 'w', encoding='UTF-8') as yaml_file:
        yaml.dump(config, yaml_file)


def get_time():
    """
    return current time string
    :return: time (string) like 2023-08-19_20-18-9
    """
    time_ = time.strftime("%Y-%m-%d %X", time.localtime()).replace(' ', '_')
    time_ = time_.replace(':', '-')
    return time_


def compare_model_params(initial_params, trained_params):
    for name, initial_param in initial_params.items():
        trained_param = trained_params[name]
        if not torch.equal(initial_param, trained_param):
            print(f"Parameter {name} has been optimized.")
        else:
            print(f"Parameter {name} has NOT been optimized.")


def show_hist(hist: torch.Tensor):
    plt.imshow(hist, cmap='binary', interpolation='nearest',
               origin='lower', extent=[0, 1500, 0, 1500], )

    plt.colorbar(label='Frequency')  # 添加颜色条，显示频率
    plt.title('2D Histogram')

    plt.show()


def show_tensor(tensor: torch.Tensor):
    # 可视化 torch.Tensor
    plt.pcolormesh(tensor.numpy())  # 将 torch.Tensor 转换为 NumPy 数组并绘制
    plt.colorbar()
    plt.xlim(0, tensor.shape[0])
    plt.ylim(0, tensor.shape[1])
    plt.set_cmap('binary')  # black and white
    plt.title('Visualization of a flow')
    plt.show()


def get_dataloader_datasetname_numclasses(root, dataset: str, feature_method: str, batch_size: int = 128,
                                          shuffle: bool = True):
    # 从utils.py文件所在位置锚定数据集位置
    # root = os.path.join(os.path.dirname(__file__), 'dataset', 'processed') ISCX*** 's dir
    train_dataset = SimpleDataset(
        dataset=dataset, feature_method=feature_method, root=root, train=True)
    test_dataset = SimpleDataset(
        dataset=dataset, feature_method=feature_method, root=root, train=False)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, train_dataset.name(), train_dataset.get_num_classes()


def get_num_classes(path):
    dirs_count = 0
    files_or_dirs = glob.glob(os.path.join(path, '*'))
    for file_or_dir in files_or_dirs:
        if os.path.isdir(file_or_dir):
            dirs_count += 1
    return dirs_count


datasets = {
    'ISCXVPN2016_VPN': {
        'dataset_dir': './dataset/raw/ISCXVPN2016_VPN',
        'features_dir': './dataset/processed/ISCXVPN2016_VPN',
        'classes': {'video-streaming': 0, 'chat': 1, 'email': 2, 'ftp': 3, 'p2p': 4, 'voip': 5},
        'ignore_files': ['vpn_spotify_A.pcap']
    },
    # ISCXTor2016 https://www.unb.ca/cic/datasets/tor.html
    'ISCXTor2016_tor': {
        'dataset_dir': './dataset/raw/ISCXTor2016_tor',
        'features_dir': './dataset/processed/ISCXTor2016_tor',
        'classes': {'browsing': 0, 'chat': 1, 'email': 2, 'ftp': 3, 'p2p': 4, 'audio': 5, 'voip': 6, 'video': 7},
        'ignore_files': []
    },

    'ISCXTor2016_nonTor': {
        'dataset_dir': './dataset/raw/ISCXTor2016_nonTor',
        'features_dir': './dataset/processed/ISCXTor2016_nonTor',
        'classes': {'browsing': 0, 'chat': 1, 'email': 2, 'ftp': 3, 'p2p': 4, 'voip': 5},
        # streaming -AUDIO and VIDEO
        'ignore_files': ['ssl.pcap', 'spotify.pcap', 'spotify2.pcap', 'spotify2-1.pcap', 'spotify2-2.pcap',
                         'spotifyAndrew.pcap']
        # 实际上我将这些文件暂存在一个子文件夹里了
    },
}


def make_joint_features(root: str, dataset: str, feature_methods: List, feature_extractor: str = 'FlowPicNet',
                        para_dict: str = None,
                        ):
    """
    提取特征向量并保存
    root : like '/home/cape/data/trace/new_processed',
    dataset: dataset name like 'ISCXTor2016_tor'
    feature_method: FlowPic or MyFlowPic
    para_dict: 模型参数文件 like './90.89.pth'
    feature_extractor: 模型类名称 like 'FlowPicNet', 'LeNet'
    """
    path = os.path.join(root, dataset, feature_methods[0])
    assert os.path.exists(path), '不存在的路径'
    flowpic_files = glob.glob(f'{path}/*/*.npz')

    # 特征提取器
    feature_extractor: torch.nn.Module = eval(feature_extractor)(num_classes=get_num_classes(path),
                                                                 mode='feature_extractor').to(device)
    if para_dict:
        feature_extractor = (torch.load(para_dict))
    feature_extractor.eval()

    for file in tqdm(flowpic_files):
        split_file_path = file.split('/')
        # load hist and label
        # label = datasets[dataset]['classes'](split_file_path[-2])
        # [1500,1500] -> [1,1,1500,1500]
        flowpic = torch.Tensor(np.load(file)['flowpic'].astype(float)).to(device).unsqueeze(dim=0).unsqueeze(dim=0)
        myflowpic = torch.Tensor(
            np.load(file.replace(feature_methods[0], feature_methods[1]))['flowpic'].astype(float)).to(
            device).unsqueeze(dim=0).unsqueeze(dim=0)

        # 获取并串联特征
        with torch.no_grad():
            flowpic_feature = feature_extractor(flowpic)
            myflowpic_feature = feature_extractor(myflowpic)
            joint_fature = torch.concat([flowpic_feature, myflowpic_feature], dim=1)
            # save file
            save_path = os.path.join(root, dataset, 'JointFeature', split_file_path[-2])
            os.makedirs(save_path, exist_ok=True)
            np.savez_compressed(os.path.join(save_path, split_file_path[-1]), feature=joint_fature.cpu().numpy())


if __name__ == '__main__':
    make_joint_features(root='/home/cape/data/trace/new_processed', dataset='ISCXTor2016_tor',
                        feature_methods=['FlowPic', 'MyFlowPic'],
                        para_dict='/home/cape/code/FlowPic/checkpoints/FlowPicNet-ISCXVPN2016_VPN_FlowPic-Adam-ReduceLROnPlateau-2023-11-08_12-42-16/0.9152.pth',
                        feature_extractor='FlowPicNet')
    # feature = np.load(
    #     '/home/cape/code/FlowPic/dataset/processed/ISCXTor2016_tor/JointFeature/audio/flowpic-1437494738355-10.0.2.15-57188-82.161.239.177-110-6-src2dst.npz')['feature']
    # print(feature.shape)
    pass
