import torch
import numpy as np
from trainer import Trainer

if __name__ == '__main__':
    # 没有使用GPU的时候设置的固定生成的随机数
    np.random.seed(5799)
    # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.manual_seed(5799)
    # torch.cuda.manual_seed()为当前GPU设置随机种子
    torch.cuda.manual_seed(5799)

    t = Trainer(r'./config.yaml')
    # print(len(t.train_dl))
    t.train()
    if t.test_dl is not None:
        t.test()
