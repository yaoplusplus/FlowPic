import os
import shutil
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torchmetrics
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as trans
from torchvision.datasets import MNIST
from tqdm import tqdm
import classifier
from mydataset import ISCX2016Tor, ISCXTor2016EIMTC, MultiFeatureISCX
from classifier import FlowPicNet, FlowPicNet_adaptive, LeNet
from utils import load_config_from_yaml, save_config_to_yaml, get_time, get_dataloader_datasetname_numclasses

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def metrics_init(num_classes):
    test_acc = torchmetrics.Accuracy(
        task="multiclass", num_classes=num_classes)
    test_recall = torchmetrics.Recall(
        task="multiclass", average='none', num_classes=num_classes)
    test_precision = torchmetrics.Precision(
        task="multiclass", average='none', num_classes=num_classes)
    test_auc = torchmetrics.AUROC(
        task="multiclass", average="macro", num_classes=num_classes)
    return test_acc.to(device), test_recall.to(device), test_precision.to(device), test_auc.to(device)


class Trainer:
    """
    Trainer class

    args:
        self.part : 数据集初始化参数、保存配置时也用到了
    """

    def __init__(self, config, logdir):
        torch.cuda.empty_cache()
        # 1. 参数实例化
        self.config = load_config_from_yaml(config)
        self.logdir = logdir
        self.init_by_config()
        self.test_acc, self.test_recall, self.test_precision, self.test_auc = metrics_init(
            num_classes=self.num_classes)
        self.epochs = self.config['epochs']

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        """
        self.model.train()

        losses = 0
        # for feature, label in tqdm(self.train_dl, leave=False):
        for feature, label in tqdm(self.train_dl, leave=False):
            feature = feature.to(device)
            label = label.to(device)
            feature = feature.unsqueeze(1)  # MNIST不需要
            out = self.model(feature)
            onehot_label = one_hot(label, num_classes=self.num_classes).float()

            onehot_label = onehot_label.squeeze(1).float()
            loss = self.loss_func(out, onehot_label)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses += loss.item()
        batch_loss = losses / len(self.train_dl)
        # 一个epoch才应该更新lr_scheduler，而不是每一个iter
        if hasattr(self, 'lr_scheduler'):
            if self.lr_scheduler.__class__.__name__ == 'StepLR':
                self.lr_scheduler.step()
            elif self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                self.lr_scheduler.step(batch_loss)
        # tqdm.write(f'lr: {self.lr_scheduler.get_last_lr()}')
        return batch_loss

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        """
        self.model.eval()
        with torch.no_grad():
            losses = 0
            corrects = 0
            # for feature, label in tqdm(self.val_dl, leave=False):
            for feature, label in tqdm(self.val_dl, leave=False):
                feature = feature.to(device)
                label = label.to(device)
                feature = feature.unsqueeze(1)  # MNIST不需要
                out = self.model(feature)  # 没经过 softmax
                pre = nn.Softmax()(out).argmax(dim=1)
                onehot_label = one_hot(
                    label, num_classes=self.num_classes).float()

                onehot_label = onehot_label.squeeze(1).float()
                loss = self.loss_func(out, onehot_label)
                losses += loss.cpu().item()
                if len(pre.shape) > len(label.shape):
                    pre = pre.squeeze()
                elif len(pre.shape) < len(label.shape):
                    pre = pre.unsqueeze(1)
                self.test_acc(label, pre)
                # corrects += (label == pre).sum()

            acc = self.test_acc.compute()
            self.test_acc.reset()
            # print(corrects / len(self.val_dl.dataset))
            return losses / len(self.val_dl), acc.cpu().item()

    def init_by_config(self):
        # 基础参数
        self.dataset = self.config['dataset']
        self.feature_method = self.config['feature_method']
        self.batch_size = self.config['batch_size']
        self.lr = self.config['lr']
        self.shuffle = self.config['shuffle']
        if 'part' in self.config.keys():
            self.part = self.config['part']
        else:
            self.part = None
        # 基础组件

        self.train_dl, self.val_dl, self.dataset_name, self.num_classes = \
            get_dataloader_datasetname_numclasses(
                dataset=self.dataset, feature_method=self.feature_method, batch_size=self.batch_size, shuffle=self.shuffle)
        self.loss_func = eval(self.config['loss_func']).to(device)  # 损失函数实例
        self.model = eval(self.config['model']).to(device)  # 模型实例
        self.opt = eval(self.config['optim'])  # 优——化器实例
        # self.config['writer'] = eval(self.config['writer'])
        assert isinstance(self.model, torch.nn.Module)
        if self.config['lr_scheduler']:
            self.lr_scheduler = eval(self.config['lr_scheduler'])
        assert isinstance(self.loss_func, nn.modules.loss._Loss)
        # assert ismethod(config['optim'], type(torch.nn.modules))

    def archive(self):
        # 配置保存
        checkpoint_folder_name_ = self.model.name() \
            + '-' + self.dataset_name \
            + '-' + self.config['optim'].split('(')[0].split('.')[1] \
            + '-' + self.config['lr_scheduler'].split('(')[0] \
            + '-' + get_time()

        self.config['checkpoint_folder_name'] = checkpoint_folder_name_
        self.folder = f"./checkpoints/{self.config['checkpoint_folder_name']}"
        os.makedirs(self.folder, exist_ok=True)
        shutil.copy('./config.yaml', f'{self.folder}/config.yaml')
        # save_config_to_yaml(self.config, f"checkpoints/{self.config['checkpoint_folder_name']}/config.yaml")

    def train(self):
        self.archive()
        self.acc = []
        for epoch in range(self.epochs):
            self.output = {'epoch': [], 'cur_lr': [],
                           'train_loss': [], 'val_loss': [], 'acc': []}
            cur_lr = self.opt.param_groups[0]['lr']
            tqdm.write(f'epoch: {epoch},\ncur_lr: {cur_lr}')
            self.output['epoch'].append(epoch)
            self.output['cur_lr'].append(cur_lr)
            # _valid_epoch_loss, _valid_epoch_acc = self._valid_epoch(epoch)
            train_loss = self._train_epoch(epoch)
            _valid_epoch_loss, _valid_epoch_acc = self._valid_epoch(epoch)
            self.acc.append(_valid_epoch_acc)
            if _valid_epoch_acc > 0.85 and _valid_epoch_acc - max(self.acc) > 0.01 and epoch%3 == 0:
                torch.save(self.model.state_dict(), os.path.join(
                    self.folder, f'{_valid_epoch_acc:.4f}'+'.pth'))
                pass
            tqdm.write(
                f"train_loss: {train_loss}\nval_loss: {_valid_epoch_loss}\nval_acc: {_valid_epoch_acc * 100:.4f}%\n")
            self.output['train_loss'].append(train_loss)
            self.output['val_loss'].append(_valid_epoch_loss)
            self.output['acc'].append(_valid_epoch_acc)
            # print(type(epoch), type(cur_lr), type(train_loss), type(_valid_epoch_loss), type(_valid_epoch_acc.cpu().item()))
            self.write_output()

    def write_output(self):
        self.output = pd.DataFrame.from_dict(self.output)
        csv_path = os.path.join(
            'checkpoints', self.config['checkpoint_folder_name'], 'output.csv')
        if os.path.exists(csv_path):
            # 初次打开文件，包含表头
            self.output.to_csv(csv_path, mode='a+', index=False, header=False)
        else:
            # 文件已创建，不包含表头
            self.output.to_csv(csv_path, mode='a+', index=False)
