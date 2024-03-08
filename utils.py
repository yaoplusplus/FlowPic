import glob
import os
import time
from typing import List, TypeVar

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from torch import Tensor
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_PACKET_SIZE = 1500
NDArray = np.ndarray


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

    # plt.colorbar(label='Frequency')  # 添加颜色条，显示频率
    # plt.title('2D Histogram')
    plt.axis('off')
    plt.savefig(f'/home/cape/temp/show/flowpic-{sum(hist[0])}.png', bbox_inches='tight', pad_inches=0.0)

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


def get_flowpic(
        timetofirst: NDArray,
        pkts_size: NDArray,
        dim: int = 32,
        max_block_duration: int = 15,
) -> NDArray:
    """Generate a Flowpic from time series

    Arguments:
        timetofirst: time series (in seconds) of the intertime between a packet and the first packet of the flow
        pkts_size: time series of the packets size
        dim: pixels size of the output representation
        max_block_duration: how many seconds of the input time series to process

    Return:
        a 2d numpy array encoding a flowpic
    """
    indexes = np.where(timetofirst < max_block_duration)[0]
    timetofirst = timetofirst[indexes]
    pkts_size = np.clip(pkts_size[indexes], a_min=0, a_max=MAX_PACKET_SIZE)

    timetofirst_norm = (timetofirst / max_block_duration) * dim
    # 这里居然对size进行了归一化，这是可以的吗
    pkts_size_norm = (pkts_size / MAX_PACKET_SIZE) * dim
    bins = range(dim + 1)
    mtx, _, _ = np.histogram2d(x=pkts_size_norm, y=timetofirst_norm, bins=(bins, bins))

    # Quote from Sec.2.1 of the IMC22 paper
    # > If more than max value (255) packets of
    # > a certain size arrive in a time slot,
    # > we set the pixel value to max value
    # 这里的astype("uint8")导致了模型forward的失败
    mtx = np.clip(mtx, a_min=0, a_max=255).astype(np.float32)
    return mtx


def get_flowpic_tensor(
        timetofirst: NDArray,
        pkts_size: NDArray,
        dim: int = 32,
        max_block_duration: int = 15
) -> torch.Tensor:
    # 将 NumPy 数组转换为 PyTorch 张量
    timetofirst_tensor = torch.tensor(timetofirst, dtype=torch.float32)
    pkts_size_tensor = torch.tensor(np.clip(pkts_size, a_min=0, a_max=MAX_PACKET_SIZE), dtype=torch.float32)

    # 归一化
    timetofirst_norm = (timetofirst_tensor / max_block_duration) * dim
    pkts_size_norm = (pkts_size_tensor / MAX_PACKET_SIZE) * dim

    # 使用 PyTorch 的 histc 函数替代 np.histogram2d
    histogram_tensor = torch.histc2d(x=timetofirst_norm, y=pkts_size_norm, bins=dim + 1)
    # Clip 和 astype 操作
    # histogram_clipped = np.clip(histogram_np, a_min=0, a_max=255).astype(np.float32)
    return histogram_tensor


def int2one_hot(input: int, num_classes: int) -> Tensor:
    res = torch.zeros(num_classes, dtype=torch.float32)
    res[input] = 1
    return res


if __name__ == '__main__':
    # tor_model = '/home/cape/code/FlowPic/checkpoints/FlowPicNet-ISCXTor2016_tor_MyFlowPic-Adam-ReduceLROnPlateau-2023-11-09_01-29-28/0.8443.pt'
    # # JointFeature_trained_nonTor_model
    # nonTor_model = '/home/cape/code/FlowPic/checkpoints/FlowPicNet-ISCXTor2016_nonTor_MyFlowPic-Adam-ReduceLROnPlateau-2023-11-09_01-29-16/0.8582.pt'
    # vpn_model = '/home/cape/code/FlowPic/checkpoints/FlowPicNet-ISCXVPN2016_VPN_MyFlowPic-Adam-ReduceLROnPlateau-2023-11-08_16-53-26/0.9182.pt'
    # app_model = ''
    # make_joint_features(root='/home/cape/data/trace/new_processed', dataset='VoIP_Video',
    #                     feature_methods=['FlowPic', 'MyFlowPic'],
    #                     para_dict=app_model,
    #                     feature_extractor='FlowPicNet', folder_name='JointFeature_trained_app_model')
    # 测试反向传播
    times = np.random.random(size=1110)
    pkts_size = np.random.uniform(low=0, high=1500, size=1110)
    torch.histogramdd(times, pkts_size, bins=[33, 33])
    pass
