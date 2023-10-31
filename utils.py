import time

import torch
import yaml
from matplotlib import pyplot as plt


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
    plt.imshow(hist, cmap='binary', interpolation='nearest', origin='lower', extent=[0, 1500, 0, 1500], )

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
