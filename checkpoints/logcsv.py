import glob
import os
import csv
from pprint import pprint
import pandas as pd
import yaml
from tqdm import tqdm

# os.chdir(os.path.dirname(__file__))

# def comment2csv(file_name):

#     data = {'epoch':[],'cur_lr':[],'train_loss':[],'val_loss':[],'acc':[]}

#     with open(file_name, mode='r', encoding='UTF8') as f:
#         lines = f.readlines()
#         # print(lines)
#         for index, line in enumerate(lines):
#             if not line =='\n':
#                 if 'epoch' in line:
#                     data['epoch'].append(eval(line[7]))
#                 #cur_lr
#                 elif 'cur_lr' in line:
#                     data['cur_lr'].append(eval(line[8:]))
#                 elif 'train_loss' in line:
#                     data['train_loss'].append(eval(line[12:]))
#                 elif 'val_loss' in line:
#                     data['val_loss'].append(eval(line[10:]))
#                 elif 'acc' in line:
#                     data['acc'].append(eval(line[9:-2]))
#         df = pd.DataFrame.from_dict(data)
#         file_name =   file_name+'.csv'
#         write_csv(df, file_name)


def load_config_from_yaml(file_path):
    with open(file_path, 'r', encoding='UTF-8') as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return config


def find_quic_config():
    for file in tqdm(glob.glob('./*/config.yaml')):
        config = load_config_from_yaml(file)
        # pprint(config)
        for key in ['dataset','dataset_root','feature_method']:
            if key in config.keys():
                # print(key, value)
                if 'DELTA_T=60-IMG_DIM=32' in config[key].upper():
                    print(file)
                    pprint(config)
                    print('*'*10)


if __name__ == '__main__':
    print(os.getcwd())
    find_quic_config()
