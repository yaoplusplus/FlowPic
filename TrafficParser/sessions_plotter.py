#!/usr/bin/env python
"""
sessions_plotter.py has 3 functions to create spectogram, histogram, 2d_histogram from [(ts, size),..] session.
绘制频谱图、直方图、二维直方图
session的形式是
"""
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

MTU = 1500


def session_spectogram(ts, sizes, name=None):
    plt.scatter(ts, sizes, marker='.')
    plt.ylim(0, MTU)
    plt.xlim(ts[0], ts[-1])
    # plt.yticks(np.arange(0, MTU, 10))
    # plt.xticks(np.arange(int(ts[0]), int(ts[-1]), 10))
    plt.title(name + " Session Spectogram")
    plt.ylabel('Size [B]')
    plt.xlabel('Time [sec]')

    plt.grid(True)
    plt.show()


def session_atricle_spectogram(ts, sizes, fpath=None, show=True, tps=None):
    if tps is None:
        max_delta_time = ts[-1] - ts[0]
    else:
        max_delta_time = tps

    ts_norm = ((np.array(ts) - ts[0]) / max_delta_time) * MTU
    plt.figure()
    plt.scatter(ts_norm, sizes, marker=',', c='k', s=5)
    plt.ylim(0, MTU)
    plt.xlim(0, MTU)
    plt.ylabel('Packet Size [B]')
    plt.xlabel('Normalized Arrival Time')
    plt.set_cmap('binary')
    plt.axes().set_aspect('equal')
    plt.grid(False)
    if fpath is not None:
        # plt.savefig(OUTPUT_DIR + fname, bbox_inches='tight', pad_inches=1)
        plt.savefig(fpath, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def session_histogram(sizes, plot=False):
    hist, bin_edges = np.histogram(sizes, bins=range(0, MTU + 1, 1))
    if plot:
        plt.bar(bin_edges[:-1], hist, width=1)
        plt.xlim(min(bin_edges), max(bin_edges) + 100)
        plt.show()
    return hist.astype(np.uint16)


def session_2d_histogram(ts, sizes, test=False, plot=False, tps=None):
    # 间隔时间
    if tps is None:
        max_delta_time = ts[-1] - ts[0]  # default path
    else:
        max_delta_time = tps

    # ts_norm = map(int, ((np.array(ts) - ts[0]) / max_delta_time) * MTU)
    ts_norm = ((np.array(ts) - ts[0]) / max_delta_time) * MTU  # 标准化数据后，数据从0开始,到MTU
    if test:
        H, xedges, yedges = np.histogram2d(sizes, ts_norm, bins=1000)
        # H, xedges, yedges = np.histogram2d(sizes, ts_norm, bins=(range(0, 101, 1), range(0, 101, 1)))
    else:
        H, xedges, yedges = np.histogram2d(sizes, ts_norm, bins=(range(0, MTU + 1), range(0, MTU + 1)))

    if plot:
        plt.pcolormesh(xedges, yedges, H)
        plt.colorbar()
        if test:
            plt.xlim(0, 10000)
            plt.ylim(0, 10000)
        else:
            plt.xlim(0, MTU)
            plt.ylim(0, MTU)

        plt.set_cmap('binary')  # black and white
        # plt.set_cmap('rainbow')  # multi color
        plt.show()
        # 这里是整形的原因是因为数据被乘以了MTU
    return H.astype(np.uint16)


def test_histogram():
    n = 40
    x = np.linspace(0, 1500, n)
    y = (x ** 2) / 1000 + np.random.rand(n) - 0.5
    H, yedges, xedges = np.histogram2d(y, x, bins=[range(1501), range(1501)])
    plt.pcolormesh(xedges, yedges, H)
    plt.set_cmap('gray_r')
    # plt.pcolormesh(xedges, yedges, H)
    plt.colorbar()
    plt.xlim(0, 1500)
    plt.ylim(0, 1500)
    # plt.set_cmap('rainbow')  # multi color
    plt.show()


if __name__ == '__main__':
    np.random.seed(100)
    # x = np.random.randint(0, 100000, [10000])
    # y = np.random.randint(0, 100000, [1000])
    # print(x)
    # print('*' * 20)
    # print(y)
    # print('*' * 20)
    # count = []
    # for i in y:
    #     if i <= 2000:
    #         count.append(i)
    # print(len(count))
    # H = session_2d_histogram(x, y, test=True, plot=True)
    test_histogram()
