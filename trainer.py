import csv
import os
import ast
from PIL import ImageOps, Image
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
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms as trans

import data_augmentations
from torchvision.datasets import MNIST
from tqdm import tqdm
import classifier

from classifier import FlowPicNet, FlowPicNet_adaptive, LeNet, PureClassifier, MiniFlowPicNet_32, \
    MiniFlowPicNet_adaptive
from utils import load_config_from_yaml, save_config_to_yaml, get_time, device, print_hist, show_hist, map_hist
from mydataset import SimpleDataset


def get_dataloader_datasetname_numclasses(root, dataset: str, feature_method: str, batch_size: int = 128,
                                          transform=None, target_transform=None,
                                          setting=None,
                                          train_csv_file: str = None,
                                          test_csv_file: str = None,
                                          val_csv_file: str = None):
    if 'split' not in dataset and False:
        print('非预先划分的数据集, 也不进行数据增强,从数据中按比例划分训练、验证、测试集')
        val = False

        train_radio = 0.8
        test_radio = 0.1

        _dataset = SimpleDataset(
            dataset=dataset, feature_method=feature_method, root=root, train=True, transform=transform,
            target_transform=target_transform, custom_csv=train_csv_file)
        total_size = len(_dataset)

        if val:
            train_size = total_size * train_radio
            test_size = total_size * test_radio
            val_size = total_size - train_size - test_size
            train_dataset, temp_dataset = random_split(
                _dataset, [train_size, test_size + val_size])
            test_dataset, val_dataset = random_split(
                temp_dataset, [test_size, val_size])
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False)
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False)
            return train_loader, test_loader, val_loader, _dataset.name(), _dataset.num_classes
        else:
            test_size = int(total_size * test_radio)
            train_size = total_size - test_size
            train_dataset, test_dataset = random_split(
                _dataset, [train_size, test_size])
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False)
            return train_loader, test_loader, None, _dataset.name(), _dataset.num_classes

    # elif 'ChangeRTT' in train_csv_file or 'TimeShift' in train_csv_file or 'PacketLoss' in train_csv_file:
    #     print('非预先划分的数据集, 只使用数据增强过的训练集')
    #     train_dataset = SimpleDataset(
    #         dataset=dataset, feature_method=feature_method, root=root, train=True, transform=transform,
    #         target_transform=target_transform, custom_csv=train_csv_file)
    #     test_dataset = SimpleDataset(
    #         dataset=dataset, feature_method=feature_method, root=root, train=False, transform=transform,
    #         target_transform=target_transform, custom_csv=test_csv_file)
    #     val_dataset = SimpleDataset(
    #         dataset=dataset, feature_method=feature_method, root=root, train=False, transform=transform,
    #         target_transform=target_transform, custom_csv=val_csv_file)
    #
    #     train_loader = DataLoader(
    #         train_dataset, batch_size=batch_size, shuffle=True)
    #     test_loader = DataLoader(
    #         test_dataset, batch_size=batch_size, shuffle=False)
    #     val_loader = DataLoader(
    #         val_dataset, batch_size=batch_size, shuffle=False)
    #     return train_loader, test_loader, val_loader, train_dataset.name(), train_dataset.num_classes
    else:
        print('预先划分的数据集，训练集和验证集是数据增强之后的数据（论文要求），测试集是原始数据')
        train_dataset = SimpleDataset(
            dataset=dataset, feature_method=feature_method, root=root, train=True, transform=transform,
            target_transform=target_transform, custom_csv=train_csv_file)
        test_dataset = SimpleDataset(
            dataset=dataset, feature_method=feature_method, root=root, train=False, transform=transform,
            target_transform=target_transform, custom_csv=test_csv_file)
        val_size = int(0.2 * len(train_dataset))
        train_size = len(train_dataset) - val_size

        # 使用random_split函数分割数据集
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size])

        # 创建DataLoader实例
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, val_loader, test_dataset.name(), test_dataset.num_classes


def metrics_init(num_classes):
    test_acc = torchmetrics.Accuracy(
        task="multiclass", num_classes=num_classes)
    test_recall = torchmetrics.Recall(
        task="multiclass", average='none', num_classes=num_classes)
    test_precision = torchmetrics.Precision(
        task="multiclass", average='none', num_classes=num_classes)
    test_auc = torchmetrics.AUROC(
        task="multiclass", average="macro", num_classes=num_classes)
    confusion_matrix = torchmetrics.ConfusionMatrix(
        task="multiclass", num_classes=num_classes)
    return test_acc.to(device), confusion_matrix, test_recall.to(device), test_precision.to(device), test_auc.to(device)


class Trainer:
    """
    Trainer class

    args:
        self.part : 数据集初始化参数、保存配置时也用到了
    """

    def __init__(self, config:str):
        self.saved_acc = [0]
        torch.cuda.empty_cache()
        # 1. 参数实例化
        self.config = load_config_from_yaml(config)
        self.init_by_config()

        self.test_acc, self.confusion_matrix, self.test_recall, self.test_precision, self.test_auc, = metrics_init(
            num_classes=self.num_classes)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        """
        self.model.train()

        losses = 0
        for feature, label in tqdm(self.train_dl, leave=False):
            feature = feature.to(device)
            label = label.to(device)
            if self.model.name() == 'PureClassifier':
                # [256,1,9000] -> [256,9000] (Linear Layer对输入的形状要求: [batch_size,size])
                feature = feature.squeeze(1)
            # CNN对输入的形状要求:[batch_size,n_channels,height,width]
            if feature.shape[1] != 1:
                feature = feature.unsqueeze(1)
            out = self.model(feature)
            # label = label.unsqueeze(1)
            onehot_label = one_hot(label, num_classes=self.num_classes).float()

            onehot_label = onehot_label.squeeze(1).float()
            loss = self.loss_func(out, onehot_label)

            self.opt.zero_grad()
            loss.backward()
            # 打印debug信息
            if self.debug and self.grad:
                self.debug_info()
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
        if self.lr_scheduler and epoch > 100:
            if self.lr_scheduler.__class__.__name__ == 'StepLR':
                self.lr_scheduler.step()
            elif self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                self.lr_scheduler.step(batch_loss)
        # tqdm.write(f'lr: {self.lr_scheduler.get_last_lr()}')
        return batch_loss, acc.cpu().item()

    def _valid_epoch(self):
        """
        Validate after training an epoch
        """
        self.model.eval()
        with torch.no_grad():
            losses = 0
            # for feature, label in tqdm(self.val_dl, leave=False):
            for feature, label in tqdm(self.val_dl, leave=False):
                feature = feature.to(device)
                label = label.to(device)
                if self.model.name() == 'PureClassifier':
                    # [256,1,9000] -> [256,9000] (Linear Layer对输入的形状要求: [batch_size,size])
                    feature = feature.squeeze(1)
                # CNN对输入的形状要求:[batch_size,n_channels,height,width]
                if feature.shape[1] != 1:
                    feature = feature.unsqueeze(1)
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

            acc = self.test_acc.compute()
            self.test_acc.reset()
            return losses / len(self.val_dl), acc.cpu().item()

    def init_by_config(self):
        # 基础参数
        self.epochs = self.config['epochs']
        self.debug = self.config['debug']
        self.grad = self.config['grad']
        self.dataset_root = self.config['dataset_root']
        self.dataset = self.config['dataset']
        self.feature_method = self.config['feature_method']
        # 策略相关的数据集设置
        self.setting = self.config['setting']
        self.train_csv_file = self.config['train_csv_file']
        self.test_csv_file = self.config['test_csv_file']
        self.val_csv_file = self.config['val_csv_file']
        self.batch_size = self.config['batch_size']
        self.lr = self.config['lr']
        self.shuffle = self.config['shuffle']
        if 'part' in self.config.keys():
            self.part = self.config['part']
        else:
            self.part = None

        # 基础组件
        # self.transform = []
        # for transform in self.config['transforms']:
        #     if hasattr(trans, transform):
        #         # 输入是PIL image
        #         self.transform.append(getattr(trans, transform))
        #     else:
        #         # 输入是info和flowpic的压缩包（.npz文件load的对象）
        #         self.transform.append(getattr(data_augmentations, transform))

        self.init_transform()
        self.train_dl, self.test_dl, self.val_dl, self.dataset_name, self.num_classes = \
            get_dataloader_datasetname_numclasses(
                root=self.dataset_root,
                dataset=self.dataset, feature_method=self.feature_method, batch_size=self.batch_size,
                transform=self.transform, target_transform=self.target_transform, setting=self.setting,
                train_csv_file=self.train_csv_file, test_csv_file=self.test_csv_file, val_csv_file=self.val_csv_file)
        if self.val_dl is None:
            # 李代桃僵
            self.val_dl = self.test_dl
            self.test_dl = None

        self.loss_func = eval(self.config['loss_func']).to(device)  # 损失函数实例
        self.model = eval(self.config['model']).to(device)  # 模型实例
        self.opt = eval(self.config['optim'])  # 优——化器实例
        # self.config['writer'] = eval(self.config['writer'])
        assert isinstance(self.model, torch.nn.Module)
        self.lr_scheduler = eval(
            self.config['lr_scheduler']) if 'lr_scheduler' in self.config else None
        assert isinstance(self.loss_func, nn.modules.loss._Loss)
        # assert ismethod(config['optim'], type(torch.nn.modules))

    def init_transform(self):
        def _get_transformer(transform):
            res = None
            if transform is not None:
                if transform in ['PacketLoss', 'ChangeRTT', 'TimeShift']:
                    res = getattr(data_augmentations, transform)()
                else:
                    res = eval(transform)
            return res

        self.transform = _get_transformer(self.config['transform'])
        self.target_transform = _get_transformer(self.config['target_transform']) if self.config[
                                                                                         'target_transform'] is not None else torch.LongTensor

    def archive(self):
        # 配置保存
        lr_scheduler = self.config['lr_scheduler'].split(
            '(')[0] if self.lr_scheduler else ''
        balanced = 'balanced' if 'balanced' in self.train_csv_file else ''
        checkpoint_folder_name_ = self.model.name() \
                                  + '-' + self.dataset_name \
                                  + '-' + self.config['optim'].split('(')[0].split('.')[1] \
                                  + '-' + lr_scheduler \
                                  + '-' + balanced \
                                  + '-' + get_time()

        self.config['checkpoint_folder_name'] = checkpoint_folder_name_
        self.folder = f"./checkpoints/{self.config['checkpoint_folder_name']}"

        os.makedirs(self.folder, exist_ok=True)
        shutil.copy('./config.yaml', f'{self.folder}/config.yaml')
        # save_config_to_yaml(self.config, f"checkpoints/{self.config['checkpoint_folder_name']}/config.yaml")

    def init_logger(self):
        self.valid_acc_logger = SummaryWriter(
            log_dir=os.path.join(self.folder, 'valid_acc'))
        self.train_acc_logger = SummaryWriter(
            log_dir=os.path.join(self.folder, 'train_acc'))
        self.train_loss_logger = SummaryWriter(
            log_dir=os.path.join(self.folder, 'train_loss'))
        self.valid_loss_logger = SummaryWriter(
            log_dir=os.path.join(self.folder, 'valid_loss'))
        self.lr_logger = SummaryWriter(log_dir=os.path.join(self.folder, 'lr'))
        # if self.debug:
        #     self.gard_logger = SummaryWriter(log_dir=os.path.join(self.folder, 'grad'))

    def train(self):
        # not save config and output while debugging
        if not self.debug:
            self.archive()
            self.init_logger()
        self.output = {'epoch': [], 'cur_lr': [],
                       'train_loss': [], 'val_loss': [], 'valid_acc': [], 'train_acc': []}
        for epoch in range(1, self.epochs + 1):
            cur_lr = self.opt.param_groups[0]['lr']
            tqdm.write(f'epoch: {epoch},\ncur_lr: {cur_lr}')

            # 训练
            train_epoch_loss, train_epoch_acc = self._train_epoch(epoch)
            # 验证
            valid_epoch_loss, valid_epoch_acc = self._valid_epoch()

            tqdm.write(
                f"\ntrain_loss: {train_epoch_loss}\nval_loss: {valid_epoch_loss}\nval_acc: {valid_epoch_acc * 100:.4f}%\ntrain_acc: {train_epoch_acc * 100:.4f}%\n")
            # 保存输出到csv
            self.output['epoch'].append(epoch)
            self.output['cur_lr'].append(cur_lr)
            self.output['train_loss'].append(train_epoch_loss)
            self.output['val_loss'].append(valid_epoch_loss)
            self.output['valid_acc'].append(valid_epoch_acc)
            self.output['train_acc'].append(valid_epoch_acc)
            #  由于保存输出会转换self.output的格式，所以先保存模型
            if not self.debug:
                # 保存模型参数
                self.save()
                self.write_output()
                # 记录输出到tensorboard
                self.valid_acc_logger.add_scalar(
                    'data', valid_epoch_acc, epoch)
                self.train_acc_logger.add_scalar(
                    'data', train_epoch_acc, epoch)
                self.valid_loss_logger.add_scalar(
                    'data', valid_epoch_loss, epoch)
                self.train_loss_logger.add_scalar(
                    'data', train_epoch_loss, epoch)
                self.lr_logger.add_scalar('data', cur_lr, epoch)

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
                self.folder, f"{self.output['valid_acc'][-1]:.4f}-epoch_{self.output['epoch'][-1]}" + '.pt'))
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

    def debug_info(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                tqdm.write(f'Parameter: {name}')
                tqdm.write(f'Gradient: {param.grad}')

    def test(self):
        self.model.eval()
        with torch.no_grad():
            losses = 0
            corrects = 0
            # for feature, label in tqdm(self.val_dl, leave=False):
            for feature, label in tqdm(self.test_dl, leave=False):
                feature = feature.to(device)
                label = label.to(device)
                if self.model.name() == 'PureClassifier':
                    # [256,1,9000] -> [256,9000] (Linear Layer对输入的形状要求: [batch_size,size])
                    feature = feature.squeeze(1)
                # CNN对输入的形状要求:[batch_size,n_channels,height,width]
                if feature.shape[1] != 1:
                    feature = feature.unsqueeze(1)
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
            print(f'test_acc: {acc.cpu().item()}')
            # write test acc to output.csv
            csvfile = open(os.path.join(
                'checkpoints', self.config['checkpoint_folder_name'], 'output.csv'), 'a')
            writer = csv.writer(csvfile)
            writer.writerow([f'test_acc: {acc.cpu().item()}'])
            csvfile.close()
            return losses / len(self.test_dl), acc.cpu().item()
