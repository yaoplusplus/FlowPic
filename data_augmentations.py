# Rotate
# 直接用transform对图片操作，填充颜色为白色
import ast
from typing import List, Optional, Union
from abc import abstractmethod

import numpy as np

MTU = 1500

TimeStamps = List[float]
Sizes = List[int]


# Horizontal Flip
# 直接用transform对图片操作，这个涉及到ndarray和PIL.Image.Image的转换

# Color Jitter
# 亮度0.8 对比度0.8 饱和度0.8 色调0.2

# 下面的几个变换如果也用transform实现的话，我需要修改trainer的代码， feature将不是单纯的hist2d数据，而是一个包含time_stamp和sizes的复合体


class BasicTransform(object):
    def __init__(self, factor: Union[List[int], int] = None, debug: bool = False,
                 rebuild: bool = True):
        """
            Args:
            factor : [a_min.a_max]，随机变化的参数
            debug : 是否打印debug信息
            rebuild : 是否从原始数据进行transform 和 重建flowpic
        """
        self.factor = factor
        self._debug = debug
        self.rebuild = rebuild

    def __call__(self, flowpic):
        """
        Args:
            flowpic ( .npz): 一个包含'flowpic'和'info'的dict

        Returns:
            torch.Tensor: 处理后的flowpic
        """
        np.random.seed()
        # numpy.ndarray -> dist
        if self.rebuild:
            info = ast.literal_eval(str(flowpic['info']))
            # 获取相对时间
            self.time_stamps = (
                np.array(info['time_stamps']) - info['start_time']) / 1000
            self.sizes = info['sizes']
            self.image_dims = info['image_dims']
            self.bin_factor = 1500 // self.image_dims[0]
            return self.transform()
        else:
            indexes = np.nonzero(flowpic['flowpic'])
            for x, y in zip(indexes[0], indexes[1]):
                # 时间段和大小段
                pass

    # 子类继承此处函数就可以实现不同的trans
    @abstractmethod
    def transform(self):
        pass

    @property
    def debug(self):
        return self._debug

    @property
    def hist(self):
        # 解包
        assert len(self.sizes) == len(self.time_stamps)
        # -> [1500,1500]
        ts_norm = ((np.array(self.time_stamps) - self.time_stamps[0]) / (
            self.time_stamps[-1] - self.time_stamps[0])) * 1500
        H, X_edge, Y_edge = np.histogram2d(ts_norm, self.sizes,
                                           bins=(
                                               range(0, 1500 + 1, 1),
                                               range(0, 1500 + 1, 1)))
        H_new_bin = self.change_hist_bins(H.astype(np.uint16))
        # 这是为了满足ToTensor()函数的输入限制
        return H_new_bin.astype(np.float32)

    def index_map(self, index):
        new_index = index // self.bin_factor
        # 末尾几个多余的将就一下
        if new_index == self.image_dims[0]:
            new_index = self.image_dims[0] - 1
        return new_index

    def change_hist_bins(self, hist):
        new_hist = np.zeros([self.image_dims[0], self.image_dims[0]])
        # 获取hist的非零索引litral
        nonzero_indexes = np.nonzero(hist)
        for x, y in zip(*nonzero_indexes):
            # 获得new_hist上的对应索引,例如0-46.875 对应new_hist上的0
            x_ = self.index_map(index=x)
            y_ = self.index_map(index=y)
            new_hist[x_][y_] += hist[x][y]
        return new_hist

    @property
    @abstractmethod
    def name(self):
        pass


class ChangeRTT(BasicTransform):
    """
    由于是对RTT的操作，所以需要对time_stamps, sizes 进行操作，然后重建flowpic
    """

    def __init__(self, factor: Union[List[int], int] = None, debug: bool = False,
                 relative: bool = True):
        """
        Args:
            relative : tranform应用在绝对时间还是相对时间序列上的指标
        """
        super().__init__(factor, debug)
        self.relative = relative

    def transform(self):
        if self.factor == None:
            self.factor = [0.5, 1.5]
        if self.debug:
            print(f'********data********')
            print(f'times_stamps: {self.time_stamps}\nsizes: {self.sizes}\n')
        # 修改时间序列，重新就修改后的(time_stamp,size) 元组排序
        for i in range(len(self.sizes)):
            a = np.random.uniform(*self.factor)
            # TODO 针对相对时间还是绝对时间进行transform
            # 绝对时间
            self.time_stamps[i] = self.time_stamps[i] * a

        self.time_stamps, self.sizes = zip(
            *sorted(zip(self.time_stamps, self.sizes), key=lambda b: b[0]))
        if self.debug:
            print(f'******ChangeRTT******')
            print(f'times_stamps: {self.time_stamps}\nsizes: {self.sizes}\n')
        # 生成相对时间序列
        self.time_stamps = (np.array(self.time_stamps) - self.time_stamps[0])
        if self.debug:
            print(f'******Sort******')
            print(f'times_stamps: {self.time_stamps}\nsizes: {self.sizes}\n')
        return self.hist

    @property
    def name(self):
        return 'ChangeRTT'


class TimeShift(BasicTransform):
    def __init__(self, factor: Union[List[int], int] = None):
        super().__init__(factor)

    def transform(self):
        if self.factor == None:
            self.factor = [-1, 1]
        # 修改时间序列，重新就修改后的时间序列排列(time_stamp,size) 元组
        for i in range(len(self.sizes)):
            b = np.random.uniform(*self.factor)
            self.time_stamps[i] = self.time_stamps[i] + b
        self.time_stamps, self.sizes = zip(
            *sorted(zip(self.time_stamps, self.sizes), key=lambda b: b[0]))
        # 时间移动后,要重新生成相对时间序列
        self.time_stamps = (np.array(self.time_stamps) - self.time_stamps[0])
        return self.hist

    @property
    def name(self):
        return 'TimeShift'


class PacketLoss(BasicTransform):
    def __init__(self, factor: Union[List[int], int] = None):
        super().__init__(factor)

    def transform(self):
        delta_t = 0.1
        self.factor = self.time_stamps[np.random.randint(
            len(self.time_stamps))]
        # self.time_stamps = list(self.time_stamps)
        # self.sizes = list(self.sizes)
        t_max = self.factor + delta_t
        t_min = self.factor - delta_t
        length = len(self.sizes) - 1
        for i in range(len(self.sizes)):
            if i <= length:
                if t_min <= self.time_stamps[i] <= t_max:
                    self.sizes = np.delete(self.sizes, i)
                    self.time_stamps = np.delete(self.time_stamps, i)
                    length -= 1
                    # self.sizes.pop(i)
                    # self.time_stamps.pop(i)
            pass
        return self.hist

    @property
    def name(self):
        return 'PacketLoss'


class PacketNum2Nbytes(BasicTransform):
    """
    将直方图数据中的num_packet更换为nbytes之和
    """

    def __init__(self):
        # 此转换不需要factor
        super().__init__()
        self.bins = np.linspace(0, MTU, self.image_dims[0]+1)
        self.nbytes = [0 for i in range(self.image_dims[0])]

    def plot(self):

        pass

    def transform(self):
        # 根据self.bins计算每一个bin中packet的size之和——nbytes
        for ts, size in zip(self.time_stamps, self.sizes):
            for i in range(self.image_dims[0]):
                if self.bins[i] <= ts < self.bins[i+1]:
                    self.nbytes[i] += size
        # 生成直方图数据
        hist, X_edge, Y_edge = np.histogram2d(
            self.time_stamps, self.sizes, bins=self.bins)
        # 修改直方图
        indexes = np.nonzero(hist)
        for x, y in zip(indexes[0], indexes[1]):
            hist[x, y] = self.nbytes[x]
        return hist

    @property
    def name(self):
        return 'PacketNum2Nbytes'
