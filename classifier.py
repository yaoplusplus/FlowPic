import torch
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Dropout, ReLU, Softmax, Sequential
import torchvision.models as models
from pprint import pprint


class FlowPicNet(nn.Module):
    # input [1,1,1500,1500]
    def __init__(self, num_classes, show_temp_out=False):
        super().__init__()
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
        # self.linear2 = Sequential(Linear(in_features=64, out_features=self.num_classes), Softmax())
        self.linear2 = Sequential(Linear(in_features=64, out_features=self.num_classes))

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
        x = self.linear1(x)
        if self.show_temp_out:
            print('after linear1: ', x.shape)
        x = self.linear2(x)
        if self.show_temp_out:
            print('after linear2: ', x.shape)

        return x
    def name(self):
        base = 'FlowPicNet'
        return base

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


class FlowPicNet_256(FlowPicNet):
    """
    只修改了self.linear1的in_features的大小，以适应256*256的输入尺寸
    """

    def __init__(self, num_classes, show_temp_out=False):
        super().__init__(num_classes=num_classes, show_temp_out=show_temp_out)
        self.linear1 = Sequential(Linear(in_features=80, out_features=64), ReLU(), Dropout(0.5))


def resnet(num_classes):
    # todo: add softmax func ?
    net = models.resnet18(weights=None)
    net.conv1 = Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=2, bias=False)
    net.fc = Linear(in_features=512, out_features=num_classes, bias=True)
    return net


class FlowPicNet_256_Reduce(FlowPicNet_256):
    """
    在FlowPicNet_256的基础上去掉了self.drop1 和self.linear1的drop_out
    """

    def __init__(self, num_classes, show_temp_out=False):
        super().__init__(num_classes=num_classes, show_temp_out=show_temp_out)
        self.linear1 = Sequential(Linear(in_features=80, out_features=64), ReLU())

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
        x = self.maxpool2(x)
        if self.show_temp_out:
            print('after maxpool2: ', x.shape)
        # 展平，解决batch——bug的地方
        x = self.flatten(x)
        if self.show_temp_out:
            print('after flatten: ', x.shape)
        x = self.linear1(x)
        if self.show_temp_out:
            print('after linear1: ', x.shape)
        x = self.linear2(x)
        if self.show_temp_out:
            print('after linear2: ', x.shape)

        return x


if __name__ == '__main__':
    # # code for test FlowPicNet
    # t = torch.rand([1, 1500, 1500])
    # print(t.shape)
    # model = net(5, True)
    # print(model(t))

    # code for test ResNet()
    model = FlowPicNet_256_Reduce(num_classes=4, show_temp_out=True)
    t = torch.unsqueeze(torch.rand([1, 256, 256]), 0)
    print(model(t))
    # pprint(model(t))
    pass
