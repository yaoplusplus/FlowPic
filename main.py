from trainer import Trainer
from utils import get_time

if __name__ == '__main__':
    t = Trainer(r'./config.yaml')
    # print(len(t.train_dl))
    t.train()
