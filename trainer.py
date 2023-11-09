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
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as trans
from torchvision.datasets import MNIST
from tqdm import tqdm
import classifier

from classifier import FlowPicNet, FlowPicNet_adaptive, LeNet, PureClassifier
from utils import load_config_from_yaml, save_config_to_yaml, get_time, device
from mydataset import SimpleDataset


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
    return train_loader, test_loader, train_dataset.name(), train_dataset.num_classes


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

    def __init__(self, config):
        self.saved_acc = [0]
        torch.cuda.empty_cache()
        # 1. 参数实例化
        self.config = load_config_from_yaml(config)
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
        for feature, label in tqdm(self.train_dl, leave=False):
            feature = feature.to(device)
            label = label.to(device)
            if self.model.name() == 'PureClassifier':
                feature = feature.squeeze(1)  # [256,1,9000] -> [256,9000] (Linear Layer对输入的形状要求: [batch_size,size])
            else:
                feature = feature.unsqueeze(1)  # CNN对输入的形状要求:[batch_size,n_channels,height,width]
            out = self.model(feature)
            onehot_label = one_hot(label, num_classes=self.num_classes).float()

            onehot_label = onehot_label.squeeze(1).float()
            loss = self.loss_func(out, onehot_label)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            losses += loss.item()

            pre = nn.Softmax()(out).argmax(dim=1)
            if len(pre.shape) > len(label.shape):
                pre = pre.squeeze()
            elif len(pre.shape) < len(label.shape):
                pre = pre.unsqueeze(1)
            self.test_acc(label, pre)
            # corrects += (label == pre).sum()

        acc = self.test_acc.compute()
        self.test_acc.reset()
        batch_loss = losses / len(self.train_dl)
        # 一个epoch才应该更新lr_scheduler，而不是每一个iter
        if hasattr(self, 'lr_scheduler'):
            if self.lr_scheduler.__class__.__name__ == 'StepLR':
                self.lr_scheduler.step()
            elif self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                self.lr_scheduler.step(batch_loss)
        # tqdm.write(f'lr: {self.lr_scheduler.get_last_lr()}')
        return batch_loss, acc.cpu().item()

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
                if self.model.name() == 'PureClassifier':
                    feature = feature.squeeze(1)  # [256,1,9000] -> [256,9000] (Linear Layer对输入的形状要求: [batch_size,size])
                else:
                    feature = feature.unsqueeze(1)  # CNN对输入的形状要求:[batch_size,n_channels,height,width]
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
        self.dataset_root = self.config['dataset_root']
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
                root=self.dataset_root,
                dataset=self.dataset, feature_method=self.feature_method, batch_size=self.batch_size,
                shuffle=self.shuffle)
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

    def init_logger(self):
        self.valid_acc_logger = SummaryWriter(log_dir=os.path.join(self.folder, 'valid_acc'))
        self.train_acc_logger = SummaryWriter(log_dir=os.path.join(self.folder, 'train_acc'))
        self.train_loss_logger = SummaryWriter(log_dir=os.path.join(self.folder, 'train_loss'))
        self.valid_loss_logger = SummaryWriter(log_dir=os.path.join(self.folder, 'valid_loss'))

    def train(self):

        self.archive()  # not save config and output while debugging
        self.init_logger()
        self.output = {'epoch': [], 'cur_lr': [],
                       'train_loss': [], 'val_loss': [], 'valid_acc': [], 'train_acc': []}
        for epoch in range(1, self.epochs + 1):
            cur_lr = self.opt.param_groups[0]['lr']
            tqdm.write(f'epoch: {epoch},\ncur_lr: {cur_lr}')

            # 训练
            train_epoch_loss, train_epoch_acc = self._train_epoch(epoch)
            # 验证
            valid_epoch_loss, valid_epoch_acc = self._valid_epoch(epoch)

            tqdm.write(
                f"\ntrain_loss: {train_epoch_loss}\nval_loss: {valid_epoch_loss}\nval_acc: {valid_epoch_acc * 100:.4f}%\n{train_epoch_acc * 100:.4f}")
            # 保存输出到csv
            self.output['epoch'].append(epoch)
            self.output['cur_lr'].append(cur_lr)
            self.output['train_loss'].append(train_epoch_loss)
            self.output['val_loss'].append(valid_epoch_loss)
            self.output['valid_acc'].append(valid_epoch_acc)
            self.output['train_acc'].append(valid_epoch_acc)
            #  由于保存输出会转换self.output的格式，所以先保存模型
            self.save()
            self.write_output()

            # 记录输出到tensorboard
            self.valid_acc_logger.add_scalar('data', valid_epoch_acc, epoch)
            self.train_acc_logger.add_scalar('data', train_epoch_acc, epoch)
            self.valid_loss_logger.add_scalar('data', valid_epoch_loss, epoch)
            self.train_loss_logger.add_scalar('data', train_epoch_loss, epoch)

            # 保存模型参数

    def save(self):
        flag = False
        # 保存精度高的和最后几个epoch的模型参数
        if self.output['epoch'][-1] % 3 == 0:
            if self.output['valid_acc'][-1] > 0.8:
                if self.output['valid_acc'][-1] - max(self.saved_acc) > 0.02:
                    flag = True
            if self.output['epoch'][-1] > self.epochs * 0.9:
                flag = True
        if flag:
            self.saved_acc.append(self.output['valid_acc'][-1])
            torch.save(self.model, os.path.join(  # 由于模型改动很频繁且模型不是很大，直接保存模型
                self.folder, f"{self.output['valid_acc'][-1]:.4f}" + '.pt'))
            tqdm.write('save model')

    def write_output(self):
        output2write = {'epoch': self.output['epoch'][-1], 'cur_lr': self.output['cur_lr'][-1],
                        'train_loss': [self.output['train_loss'][-1]], 'val_loss': [self.output['val_loss'][-1]],
                        'valid_acc': [self.output['valid_acc'][-1]],
                        'train_acc': [self.output['train_acc'][-1]]}
        output2write = pd.DataFrame.from_dict(output2write)
        csv_path = os.path.join(
            'checkpoints', self.config['checkpoint_folder_name'], 'output.csv')
        if os.path.exists(csv_path):
            # 初次打开文件，包含表头
            output2write.to_csv(csv_path, mode='a+', index=False, header=False)
        else:
            # 文件已创建，不包含表头
            output2write.to_csv(csv_path, mode='a+', index=False)
