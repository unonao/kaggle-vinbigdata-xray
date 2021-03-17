import argparse
import json
import os
import datetime

import numpy as np
import pandas as pd
from glob import glob
import shutil

low_thr  = 0.05
high_thr = 0.92

# 引数で config の設定を行う
parser = argparse.ArgumentParser()
parser.add_argument('--configdet', default='./output/original_512_0_sub.csv')
parser.add_argument('--configfilter', default='./class_filter/output/efficientnet_b1_512_tf_efficientnet_b1_ns_test.csv')
options = parser.parse_args()
pred_14cls = pd.read_csv(options.configdet)
pred_2cls = pd.read_csv(options.configfilter)

pred = pd.merge(pred_14cls, pred_2cls, on='image_id', how='left')
print(pred['PredictionString'].value_counts().iloc[[0]])


def filter_2cls(row, low_thr=low_thr, high_thr=high_thr):
    prob = row['pred']
    if prob<low_thr:
        row['PredictionString'] = row['PredictionString']
    elif low_thr <= prob < high_thr:
        if "14 1 0 0 1 1" not in row['PredictionString']:
            row['PredictionString'] += f' 14 {prob} 0 0 1 1'
    elif high_thr<=prob:
        row['PredictionString'] = '14 1 0 0 1 1'
    else:
        raise ValueError('Prediction must be from [0-1]')
    return row

sub = pred.apply(filter_2cls, axis=1)
sub[['image_id', 'PredictionString']].to_csv("sub_2class_filter.csv",index=False)
print(sub['PredictionString'].value_counts().iloc[[0]])
