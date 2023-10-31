import os
import csv
import time
from pprint import pprint
import numpy as np
from flowcontainer.extractor import extract
from typing import List

from tqdm import tqdm

root = r'D:\data\trace'
dataset = r'ISCXTor2016\tor'
stime = time.time()


def index_label(file):
    filename = os.path.basename(file)
    filename_no_suffix = os.path.splitext(filename)[0]
    base_name_loc = filename_no_suffix.find('_')
    label = filename_no_suffix[base_name_loc + 1:]
    return label


def extract_single(pcap_file, print_dict=False, bidirection=False):
    """

    Args:
        bidirection: pcap中的双向流
        pcap_file: pcap_file to be extracted
        print_dict: print stream info or not

    Returns: list[dict]
    """
    label = index_label(pcap_file)
    streams_list = []

    path = os.path.join(root, dataset, 'output-dir', pcap_file)

    split_flag_ = False if os.path.getsize(path) > 1000000 else True

    result = extract(infile=path,
                     filter='ip',
                     extension=[],
                     split_flag=split_flag_,
                     verbose=True
                     )
    for key in tqdm(result):
        # 一个pcap 可以对应多个stream（使用五元组进行区分）
        # The return value result is a dict, the key is a tuple (filename,protocol,stream_id)
        # and the value is a Flow object, user can access Flow object as flowcontainer.flows.Flow's attributes refer.

        stream = result[key]
        # 使用字典初始化
        stream_list = [label,
                       stream.src,
                       stream.sport,
                       stream.dst,
                       stream.dport,
                       key[1],
                       stream.time_start,
                       len(stream.payload_lengths)]
        if bidirection:
            mask = np.array(stream.payload_lengths, dtype=float) > 0
            streams_list = []
            sd_timestamps = list(np.array(stream.payload_timestamps, dtype=float)[mask])
            sd_sizes = list(np.array(stream.payload_lengths, dtype=int)[mask])
            ds_timestamps = list(np.array(stream.payload_timestamps, dtype=float)[~mask])
            ds_sizes = list(np.array(stream.payload_lengths, dtype=int)[~mask])

            sd_stream_list = [label,
                              stream.src,
                              stream.sport,
                              stream.dst,
                              stream.dport,
                              key[1],
                              sd_timestamps[0],
                              len(sd_timestamps)]
            ds_stream_list = [label,
                              stream.dst,
                              stream.dport,
                              stream.src,
                              stream.sport,
                              key[1],
                              ds_timestamps[0],
                              len(ds_timestamps)]

            # 时间序列从0开始
            sd_timestamps = [timestamp - sd_stream_list[-2] for timestamp in sd_timestamps]
            ds_timestamps = [timestamp - ds_stream_list[-2] for timestamp in ds_timestamps]

            # 拼接时间和大小序列
            sd_stream_list += sd_timestamps
            sd_stream_list.append(' ')
            sd_stream_list += sd_sizes

            ds_stream_list += ds_timestamps
            ds_stream_list.append(' ')
            ds_stream_list += ds_sizes

            streams_list.append(sd_stream_list)
            streams_list.append(ds_stream_list)
            return streams_list

        # timestamps_list 和 sizes_list将会被转换为字符串，后面可以再转换回去
        stream_list += stream.payload_timestamps
        # 时间序列和大小序列之间的空白格
        stream_list.append('')
        stream_list += stream.payload_lengths
        # 时间戳相对处理
        modified_stream_list = [x - stream.time_start if i in range(8, 8 + len(stream.payload_timestamps)) else x
                                for i, x in enumerate(stream_list)]
        if print_dict:
            print('Flow {0} info:'.format(key))
            # access ip src
            print('src ip:', stream.src)
            # access ip dst
            print('dst ip:', stream.dst)
            # access srcport
            print('sport:', stream.sport)
            # access_dstport
            print('dport:', stream.dport)
            # access payload packet lengths
            print('payload lengths :', stream.payload_lengths)
            # access payload packet timestamps sequence:
            print('payload timestamps:', stream.payload_timestamps)
            # access ip packet lengths, (including packets with zero payload, and ip header)
            print('ip packets lengths:', stream.ip_lengths)
            # access ip packet timestamp sequence, (including packets with zero payload)
            print('ip packets timestamps:', stream.ip_timestamps)

            # access default lengths sequence, the default length sequences is the payload lengths sequences
            print('default length sequence:', stream.lengths)
            # access default timestamp sequence, the default timestamp sequence is the payload timestamp sequences
            print('default timestamp sequence:', stream.timestamps)

            print('start timestamp:{0}, end timestamp :{1}'.format(stream.time_start, stream.time_end))

            # access the proto
            print('proto:', stream.ext_protocol)
            ##access sni of the flow if any else empty str
            print('extension:', stream.extension)
        streams_list.append(modified_stream_list)
    return streams_list


def extract_dir(dir: str):
    """
    Args:
        dir: pcap文件保存的目录

    Returns: List[stream_list],一个stream_list相当于csv文件的一行

    """
    streams_list = []
    for file in tqdm(os.listdir(dir)):
        streams_list += extract_single(file)
    return streams_list


def streams_list2csv(streams_list: List[List], file_name):
    with open(os.path.join(root, dataset, file_name), 'a', newline='') as f:
        csv_writer = csv.writer(f)
        # pprint(streams_list)
        for stream_list in streams_list:
            csv_writer.writerow(stream_list)


def pcap2csv():
    work_dir = os.path.join(root, dataset, 'output-dir')
    pprint(f'working at dir: {work_dir}')
    streams_list = extract_dir(work_dir)
    streams_list2csv(streams_list, file_name='processed.csv')


def test():
    # print(index_label('11905223351545221150_firefox_snowflake.pcap')) ok
    streams_list = extract_single(os.path.join(root, dataset, 'VIDEO_Vimeo_Gateway.pcap'),
                                  bidirection=False)
    for stream in streams_list:
        print(stream[0:8])
    # pprint(streams_list[-10:-1])
    # print('*'*20)
    # pprint(streams_list[0:10])
    # 统计stream的包数

    # pcap2csv() # 指定目录下所有pcap提取、转换为一个csv文件
    pass


if __name__ == '__main__':
    # extract_single(
    #     r'D:\data\trace\nprint-datasets\application_case_study\output-dir\5059893426172837500_firefox_snowflake.pcap')
    test()
    pass
