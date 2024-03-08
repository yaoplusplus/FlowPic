from datetime import datetime

import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader

from trainer import Trainer

if __name__ == '__main__':
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    # 没有使用GPU的时候设置的固定生成的随机数
    np.random.seed(5799)
    # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.manual_seed(5799)
    # torch.cuda.manual_seed()为当前GPU设置随机种子
    torch.cuda.manual_seed(5799)

    t = Trainer(r'./config.yaml')
    # 修改数据集
    ###
    # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize(32)])
    # mnist_train = torchvision.datasets.MNIST(root='/home/cape/data', train=True, transform=transform, download=True)
    # mnist_test = torchvision.datasets.MNIST(root='/home/cape/data', train=False, transform=transform, download=False)
    # train_dl = DataLoader(mnist_train, batch_size=256, shuffle=True)
    # test_dl = DataLoader(mnist_test, batch_size=256, shuffle=True)
    # t.num_classes = 10
    # t.train_dl = train_dl
    # t.val_dl = test_dl
    ###

    # print(len(t.train_dl))
    t.train()

    if t.test_dl is not None:
        t.test()
