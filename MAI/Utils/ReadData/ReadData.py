import os
from glob import glob
from itertools import chain

import numpy as np
import pandas as pd


def main(path="MAI/Utils/ReadData/NewImages"):

    all_xray_df = pd.read_csv('MAI/Utils/ReadData/Data_Entry_2017_v2020.csv')
    print(all_xray_df.columns)
    all_image_paths = {}
    for i in range(1, 13):
        print(os.path.join(path, f'images_0{i}', '*.png'))
        all_image_paths.update({os.path.basename(x): x for x in
                                glob(os.path.join(path, f'images_0{i}', '*.png'))})

    print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
    all_xray_df['path'] = all_xray_df['ImageIndex'].map(all_image_paths.get)

    all_xray_df['Finding Labels'] = all_xray_df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
    all_xray_df = all_xray_df[all_xray_df['Finding Labels'] != '']

    all_labels = np.unique(list(chain(*all_xray_df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
    all_labels = [x for x in all_labels if len(x) > 0]
    print('All Labels ({}): {}'.format(len(all_labels), all_labels))

    for c_label in all_labels:
        if len(c_label) > 1:  # leave out empty labels
            all_xray_df[c_label] = all_xray_df['Finding Labels']\
                .map(lambda finding: 1.0 if c_label in finding else 0)\
                .copy()

    with open("MAI/Utils/ReadData/test_list.txt", 'r') as f:
        lines = f.readlines()
        lines = list([line.rstrip('\n') for line in lines])
        train_val_df = all_xray_df[all_xray_df['ImageIndex'].isin(lines)].copy()

    with open("MAI/Utils/ReadData/test_list.txt", 'r') as f:
        lines = f.readlines()
        lines = list([line.rstrip('\n') for line in lines])
        test_df = all_xray_df[all_xray_df['ImageIndex'].isin(lines)].copy()

    print('train', train_val_df.shape[0], 'test', test_df.shape[0])

    train_val_df['newLabel'] = train_val_df.apply(lambda x: x['Finding Labels'].split('|'), axis=1)
    test_df['newLabel'] = test_df.apply(lambda x: x['Finding Labels'].split('|'), axis=1)

    return train_val_df, test_df, all_labels
