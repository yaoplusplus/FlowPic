import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torch.utils.data import DataLoader
import torchmetrics


def load_model(model_path: str, net: nn.Module = None) -> nn.Module:
    try:
        model = torch.load(model_path, map_location='cuda')
        return model
    except pickle.UnpicklingError:
        print(f"{model_path} 是保存的模型参数，直接加载失败，使用load_statedict方法加载")
        if net is not None:
            net.load_state_dict(torch.load(model_path))
            net.to('cuda')
            return net
        else:
            print('请提供nn.Modile对象，以供加载参数')
            exit(1)


def test(
        model_path: str,
        num_classes: int,
        loss_func: nn.Module,
        metric: torchmetrics.Metric = torchmetrics.Accuracy(),
        test_dl: DataLoader = None,
        device='cuda') -> None:
    model = load_model(model_path)
    model.eval()

    with torch.no_grad():
        losses = 0
        corrects = 0
        # for feature, label in tqdm(self.val_dl, leave=False):
        for feature, label in tqdm(test_dl, leave=False):
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
        loss = losses / len(self.test_dl)
        self.test_acc.reset()
        # write test acc to output.csv
        csvfile = open(os.path.join(
            'checkpoints', self.config['checkpoint_folder_name'], 'output.csv'), 'a')
        writer = csv.writer(csvfile)
        writer.writerow([f'test_loss: {loss}, test_acc: {acc.cpu().item()}'])
        csvfile.close()
        return loss, acc.cpu().item()
