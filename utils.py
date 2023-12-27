import glob
import os
import time
from typing import List

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm

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


def show_hist(hist: torch.Tensor, size=32):
    plt.imshow(hist, cmap='binary', interpolation='nearest',
               origin='lower', extent=[0, size, 0, size], )

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
                        para_dict: str = None, folder_name: str = 'JointFeature'
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
    feature_extractor = torch.load(para_dict)
    feature_extractor.eval()

    for file in tqdm(flowpic_files):
        split_file_path = file.split('/')
        # load hist and label
        # label = datasets[dataset]['classes'](split_file_path[-2])
        # [1500,1500] -> [1,1,1500,1500]
        flowpic = torch.Tensor(np.load(file)['flowpic'].astype(
            float)).to(device).unsqueeze(dim=0).unsqueeze(dim=0)
        myflowpic = torch.Tensor(
            np.load(file.replace(feature_methods[0], feature_methods[1]))['flowpic'].astype(float)).to(
            device).unsqueeze(dim=0).unsqueeze(dim=0)

        # 获取并串联特征
        with torch.no_grad():
            flowpic_feature = feature_extractor.extractor(flowpic)
            myflowpic_feature = feature_extractor.extractor(myflowpic)
            joint_fature = torch.concat(
                [flowpic_feature, myflowpic_feature], dim=1)
            # save file
            save_path = os.path.join(
                root, dataset, folder_name, split_file_path[-2])
            os.makedirs(save_path, exist_ok=True)
            np.savez_compressed(os.path.join(
                save_path, split_file_path[-1]), feature=joint_fature.cpu().numpy())


if __name__ == '__main__':
    tor_model = '/home/cape/code/FlowPic/checkpoints/FlowPicNet-ISCXTor2016_tor_MyFlowPic-Adam-ReduceLROnPlateau-2023-11-09_01-29-28/0.8443.pt'
    # JointFeature_trained_nonTor_model
    nonTor_model = '/home/cape/code/FlowPic/checkpoints/FlowPicNet-ISCXTor2016_nonTor_MyFlowPic-Adam-ReduceLROnPlateau-2023-11-09_01-29-16/0.8582.pt'
    vpn_model = '/home/cape/code/FlowPic/checkpoints/FlowPicNet-ISCXVPN2016_VPN_MyFlowPic-Adam-ReduceLROnPlateau-2023-11-08_16-53-26/0.9182.pt'
    app_model = ''
    make_joint_features(root='/home/cape/data/trace/new_processed', dataset='VoIP_Video_Application_NonVPN',
                        feature_methods=['FlowPic', 'MyFlowPic'],
                        para_dict=app_model,
                        feature_extractor='FlowPicNet', folder_name='JointFeature_trained_app_model')
    pass


def map_hist(hist):
    nonzero_indexes = np.nonzero(hist)
    for x, y in zip(nonzero_indexes[0], nonzero_indexes[1]):
        if hist[x][y] > 255:
            hist[x][y] = 255
    return hist.astype(np.uint8)


def print_hist(hist):
    # tensor
    if isinstance(hist, torch.Tensor):
        if len(hist.shape) == 3 and hist.shape[0] == 1:
            hist = hist.squeeze(0)
            nonzero_indexes = np.nonzero(hist)  # [82,2]
            for index in nonzero_indexes:
                print(index[0], index[1], hist[index[0]][index[1]])
            return

    nonzero_indexes = np.nonzero(hist)
    for x, y in zip(nonzero_indexes[0], nonzero_indexes[1]):
        print(x, y, hist[x][y])
