import os
import csv
import pandas as pd
import ruamel.yaml

os.chdir(os.path.dirname(__file__))


def write_csv(df, csv_path):
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a+', index=False, header=False)
    else:
        df.to_csv(csv_path, mode='a+', index=False)


def comment2csv(file_name):
    
    data = {'epoch':[],'cur_lr':[],'train_loss':[],'val_loss':[],'acc':[]}
    
    with open(file_name, mode='r', encoding='UTF8') as f:
        lines = f.readlines()
        # print(lines)
        for index, line in enumerate(lines):
            if not line =='\n':
                if 'epoch' in line:
                    data['epoch'].append(eval(line[7]))
                #cur_lr
                elif 'cur_lr' in line:
                    data['cur_lr'].append(eval(line[8:]))
                elif 'train_loss' in line:
                    data['train_loss'].append(eval(line[12:]))
                elif 'val_loss' in line:
                    data['val_loss'].append(eval(line[10:]))
                elif 'acc' in line:
                    data['acc'].append(eval(line[9:-2]))
        df = pd.DataFrame.from_dict(data)
        file_name =   file_name+'.csv'
        write_csv(df, file_name)
        
        
if __name__ == '__main__':
    print(os.getcwd())
    comment2csv('FlowPicNet-ISCXTor2016-nonTor-EIMTC-Adam_0.001-2023-10-23_10-11-29.out')
