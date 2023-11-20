import torch
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Dropout, ReLU, Sequential
import torchvision.models as models
from pprint import pprint
from sklearn.decomposition import PCA


class FlowPicNet(nn.Module):
    # input [batch_size,1,1500,1500]
    def __init__(self, num_classes, show_temp_out=False, mode=None):
        """
        mode : feature_extractor-提取特征向量, classifier-分类器, None-全流程
        """
        super().__init__()
        self.mode = mode
        self.num_classes = num_classes
        self.show_temp_out = show_temp_out
        # stride >1 的时候，pytorch不允许使用 padding="same"
        self.conv1 = Sequential(Conv2d(in_channels=1, out_channels=10, kernel_size=10, stride=5, padding=4), ReLU())
        self.maxpool1 = MaxPool2d(kernel_size=2)
        self.conv2 = Sequential(Conv2d(in_channels=10, out_channels=20, kernel_size=10, stride=5, padding=4), ReLU())
        self.drop1 = Dropout(0.25)
        self.maxpool2 = MaxPool2d(kernel_size=2)
        self.flatten = Flatten()
        self.linear1 = Sequential(Linear(in_features=4500, out_features=64), ReLU(), Dropout(0.5))
        # self.linear2 = Sequential(Linear(in_features=64, out_features=self.num_classes))
        self.linear2 = Sequential(Linear(in_features=64, out_features=self.num_classes))

        self.extractor = Sequential(self.conv1, self.maxpool1, self.conv2, self.drop1, self.maxpool2, self.flatten)
        self.classifier = Sequential(self.linear1, self.linear2)

    def forward(self, x):
        if self.mode == 'feature_extractor':
            return self.extractor(x)
        if self.mode == 'classifier':
            return self.classifier(x)
        else:
            shapes = {}
            x = self.conv1(x)
            shapes['conv1'] = x.shape
            x = self.maxpool1(x)
            shapes['maxpool1'] = x.shape
            x = self.conv2(x)
            shapes['conv2'] = x.shape
            x = self.drop1(x)
            x = self.maxpool2(x)
            shapes['maxpool2'] = x.shape
            # 展平，解决batch——bug的地方
            x = self.flatten(x)
            shapes['flatten'] = x.shape
            x = self.linear1(x)
            shapes['linear1'] = x.shape
            x = self.linear2(x)
            shapes['linear2'] = x.shape

            if self.show_temp_out:
                print(shapes)
            return x

    def name(self):
        base = 'FlowPicNet'
        return base


class PureClassifier(nn.Module):
    def __init__(self, in_features, out_features, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.linear1 = Sequential(Linear(in_features=in_features, out_features=out_features), ReLU(), Dropout(0.5))
        self.linear2 = Sequential(Linear(in_features=out_features, out_features=self.num_classes))

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

    def name(self):
        return 'PureClassifier'


class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),  # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),  # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, num_classes)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


class FlowPicNet_adaptive(FlowPicNet):
    """
    可以修改self.linear1的in_features的大小，以适应不同的输入尺寸
    3000*1500 -> 9000
    1500*1500 -> 4500
    256*256   -> 90
    """

    def __init__(self, num_classes, show_temp_out=False, liner1_in_feature=4500):
        super().__init__(num_classes=num_classes, show_temp_out=show_temp_out)
        self.linear1 = Sequential(Linear(in_features=liner1_in_feature, out_features=64), ReLU(), Dropout(0.5))


def resnet(num_classes):
    net = models.resnet18(weights=None)
    net.conv1 = Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=2, bias=False)
    net.fc = Linear(in_features=512, out_features=num_classes, bias=True)
    return net


def reduce_dim(x, dst_dim):
    # sklearn 方法，放在网络forward函数里会丢失梯度信息
    # estimator = PCA(n_components=n_components)
    # x = estimator.fit_transform(x)
    # return x

    # torch方法
    U, S, V = torch.pca_lowrank(x, q=dst_dim)
    # U: mxq matrix, S：q-vector V:nxq matrix
    return torch.mm(x, V)


class MiniFlowPicNet_32(nn.Module):
    def __init__(self, num_classes, show_temp_out=False):
        super().__init__()
        self.num_classes = num_classes
        self.show_temp_out = show_temp_out
        # stride >1 的时候，pytorch不允许使用 padding="same"
        self.conv1 = Sequential(Conv2d(in_channels=1, out_channels=6, kernel_size=9, stride=1, padding=2), ReLU())
        # 6*28*28
        self.maxpool1 = MaxPool2d(kernel_size=2)
        # 6*14*14
        self.conv2 = Sequential(Conv2d(in_channels=6, out_channels=16, kernel_size=9, stride=1, padding=2),
                                ReLU())
        # 6*10*10
        self.drop1 = Dropout(0.25)
        self.maxpool2 = MaxPool2d(kernel_size=2)
        # 6*5*5
        # paper里提到了PAC
        self.PCA = reduce_dim
        self.flatten = Flatten()
        self.linear1 = Sequential(Linear(in_features=120, out_features=84), ReLU(), Dropout(0.5))
        # self.linear2 = Sequential(Linear(in_features=64, out_features=self.num_classes), Softmax())
        self.linear2 = Sequential(Linear(in_features=84, out_features=self.num_classes))

    def forward(self, x):
        x = self.conv1(x)
        if self.show_temp_out:
            print('after conv1: ', x.shape)
        x = self.maxpool1(x)
        if self.show_temp_out:
            print('after maxpool1: ', x.shape)
        x = self.conv2(x)
        if self.show_temp_out:
            print('after conv2: ', x.shape)
        x = self.drop1(x)

        x = self.maxpool2(x)
        if self.show_temp_out:
            print('after maxpool2: ', x.shape)
        # 展平，解决batch——bug的地方
        x = self.flatten(x)
        if self.show_temp_out:
            print('after flatten: ', x.shape)
        assert x.shape[1] >= 120  # 实际上就是batch_size，因为要降低维度到120，函数要求任意维度都要大于120
        # PCA降维
        x = self.PCA(x.squeeze(0), dst_dim=120).unsqueeze(0)
        print('after reduce_dim: ', x.shape)
        x = self.linear1(x)
        if self.show_temp_out:
            print('after linear1: ', x.shape)
        x = self.linear2(x)
        if self.show_temp_out:
            print('after linear2: ', x.shape)

        return x

    def name(self):
        return 'FlowPicNet_32'


class JointFlowPicNet(nn.Module):
    pass


if __name__ == '__main__':
    # 测试模型输出
    model = MiniFlowPicNet_32(num_classes=4, show_temp_out=True)
    t = torch.rand([128, 1, 32, 32])
    print(model(t).shape)
    # pass

    # tensor = torch.rand([256, 400])
    # reduce_dim(tensor, n_components=120)
    # U, S, V = torch.pca_lowrank(tensor, q=120)
    # print(U.shape)
    # print(S.shape)
    # print(V.shape)
    # print(torch.mm(tensor, V).shape)
