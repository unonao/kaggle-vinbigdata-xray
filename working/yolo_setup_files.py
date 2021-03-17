#!pip install iterative-stratification
#!pip install ensemble-boxes
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from ensemble_boxes import nms, weighted_boxes_fusion

import argparse
import json
import os
import datetime
import yaml

import numpy as np, pandas as pd
from glob import glob
import shutil, os
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm
import seaborn as sns

# 引数で config の設定を行う
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.json')
options = parser.parse_args()
CFG = json.load(open(options.config))
config_filename = os.path.splitext(os.path.basename(options.config))[0]


list_remove = []
image_remove = []

MAIN_PATH = '../input/vinbigdata-chest-xray-abnormalities-detection/'
YOLO_DATA_DIR = f'/home/kaggle-vinbigdata-xray/working/chest_yolo/{config_filename}' # yolov5 ディレクトリから見た位置
SUB_PATH = os.path.join(MAIN_PATH, 'sample_submission.csv')
TRAIN_PATH = f'../input/vinbigdata-chest-xray-resized-png-{CFG["dim"]}x{CFG["dim"]}{CFG["way"]}/train'
TRAIN_META_PATH = f'../input/vinbigdata-chest-xray-resized-png-{CFG["dim"]}x{CFG["dim"]}/train_meta.csv'
def main():

    # load train.csv
    train_df = pd.read_csv(os.path.join(MAIN_PATH,'train.csv'))
    train_df['image_path'] = f'{TRAIN_PATH}'+train_df.image_id+'.png'
    train_meta_df = pd.read_csv(TRAIN_META_PATH)
    train_meta_df.columns = ['image_id', 'h', 'w']
    train_df = train_df.merge(train_meta_df, on="image_id")


    # preprocessing
    def label_resize(org_size, img_size, *bbox):
        x0, y0, x1, y1 = bbox
        x0_new = int(np.round(x0*img_size[1]/org_size[1]))
        y0_new = int(np.round(y0*img_size[0]/org_size[0]))
        x1_new = int(np.round(x1*img_size[1]/org_size[1]))
        y1_new = int(np.round(y1*img_size[0]/org_size[0]))
        return x0_new, y0_new, x1_new, y1_new
    train_normal = train_df[train_df['class_name']=='No finding'].reset_index(drop=True)
    train_normal['x_min_resize'] = 0
    train_normal['y_min_resize'] = 0
    train_normal['x_max_resize'] = 1
    train_normal['y_max_resize'] = 1

    train_abnormal = train_df[train_df['class_name']!='No finding'].reset_index(drop=True)
    train_abnormal[['x_min_resize', 'y_min_resize', 'x_max_resize', 'y_max_resize']] = train_abnormal \
    .apply(lambda x: label_resize(x[['h', 'w']].values, [CFG["dim"],CFG["dim"]], *x[['x_min', 'y_min', 'x_max', 'y_max']].values),
        axis=1, result_type="expand")
    train_abnormal['x_center'] = 0.5*(train_abnormal['x_min_resize'] + train_abnormal['x_max_resize'])
    train_abnormal['y_center'] = 0.5*(train_abnormal['y_min_resize'] + train_abnormal['y_max_resize'])
    train_abnormal['width'] = train_abnormal['x_max_resize'] - train_abnormal['x_min_resize']
    train_abnormal['height'] = train_abnormal['y_max_resize'] - train_abnormal['y_min_resize']
    train_abnormal['area'] = train_abnormal.apply(lambda x: (x['x_max_resize']-x['x_min_resize'])*(x['y_max_resize']-x['y_min_resize']), axis=1)
    train_abnormal = train_abnormal[~train_abnormal.index.isin(list_remove)].reset_index(drop=True)


    # change by wbf
    def Preprocess_wbf(df, size=CFG["dim"], iou_thr=0.5, skip_box_thr=0.0001):
        list_image = []
        list_boxes = []
        list_cls = []
        list_h, list_w = [], []
        new_df = pd.DataFrame()
        for image_id in tqdm(df['image_id'].unique(), leave=False):
            image_df = df[df['image_id']==image_id].reset_index(drop=True)
            h, w = image_df.loc[0, ['h', 'w']].values
            boxes = image_df[['x_min_resize', 'y_min_resize',
                            'x_max_resize', 'y_max_resize']].values.tolist()
            boxes = [[j/(size-1) for j in i] for i in boxes]
            scores = [1.0]*len(boxes)
            labels = [float(i) for i in image_df['class_id'].values]
            boxes, scores, labels = weighted_boxes_fusion([boxes], [scores], [labels],
                                                        weights=None,
                                                        iou_thr=iou_thr,
                                                        skip_box_thr=skip_box_thr)
            list_image.extend([image_id]*len(boxes))
            list_h.extend([h]*len(boxes))
            list_w.extend([w]*len(boxes))
            list_boxes.extend(boxes)
            list_cls.extend(labels.tolist())
        list_boxes = [[int(j*(size-1)) for j in i] for i in list_boxes]
        new_df['image_id'] = list_image
        new_df['class_id'] = list_cls
        new_df['h'] = list_h
        new_df['w'] = list_w
        new_df['x_min_resize'], new_df['y_min_resize'], \
        new_df['x_max_resize'], new_df['y_max_resize'] = np.transpose(list_boxes)
        new_df['x_center'] = 0.5*(new_df['x_min_resize'] + new_df['x_max_resize'])
        new_df['y_center'] = 0.5*(new_df['y_min_resize'] + new_df['y_max_resize'])
        new_df['width'] = new_df['x_max_resize'] - new_df['x_min_resize']
        new_df['height'] = new_df['y_max_resize'] - new_df['y_min_resize']
        new_df['area'] = new_df.apply(lambda x: (x['x_max_resize']-x['x_min_resize'])\
                                    *(x['y_max_resize']-x['y_min_resize']), axis=1)
        return new_df

    train_abnormal = Preprocess_wbf(train_abnormal)

    # split for CV
    def split_df(df):
        kf = MultilabelStratifiedKFold(n_splits=CFG["fold_num"], shuffle=True, random_state=CFG["seed"])
        df['id'] = df.index
        annot_pivot = pd.pivot_table(df, index=['image_id'], columns=['class_id'],
                                    values='id', fill_value=0, aggfunc='count') \
        .reset_index().rename_axis(None, axis=1)
        for fold, (train_idx, val_idx) in enumerate(kf.split(annot_pivot,
                                                            annot_pivot.iloc[:, 1:(1+df['class_id'].nunique())])):
            annot_pivot[f'fold_{fold}'] = 0
            annot_pivot.loc[val_idx, f'fold_{fold}'] = 1
        return annot_pivot

    size_df = pd.read_csv(TRAIN_META_PATH)
    size_df.columns = ['image_id', 'h', 'w']

    fold_csv = split_df(train_df)
    fold_csv = fold_csv.merge(size_df, on='image_id', how='left')


    # create dataset (yolo 用の設定ファイルを作る）
    images_dir = f'{YOLO_DATA_DIR}/images'
    labels_dir = f'{YOLO_DATA_DIR}/labels'

    def create_labels(df, split_df, train_folder, size = CFG["dim"]):
        """
        全てのimageとラベルを作成する
        """
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        for image_id in tqdm(split_df.image_id.unique()):
            label_path = f'{labels_dir}/{image_id}.txt'
            with open(label_path, 'w+') as f:
                row = df[df['image_id']==image_id][['class_id', 'x_center', 'y_center', 'width', 'height']].values
                row[:, 1:] /= size
                row = row.astype('str')
                for box in range(len(row)):
                    text = ' '.join(row[box])
                    f.write(text)
                    f.write('\n')
            image_path = f'{images_dir}/{image_id}.png'
            shutil.copy(f'{train_folder}/{image_id}.png', image_path)

    create_labels(train_abnormal, fold_csv, TRAIN_PATH)
    fold_csv["img_path"] = images_dir + "/" + fold_csv["image_id"] + ".png"  # img へのパスを記す

    # 設定ファイルを作成
    def create_path_file(split_df,  yolo_data_dir, fold):
        """
        train, val へのパスを記したファイルを fold ごとに作り、それらに対応するymlファイルをを作る
        """
        train_df = split_df[split_df[f'fold_{fold}']!=0].reset_index(drop=True)
        val_df = split_df[split_df[f'fold_{fold}']==0].reset_index(drop=True)

        train_file_list = f"{yolo_data_dir}/train_list_fold_{fold}.txt"
        val_file_list = f"{yolo_data_dir}/val_list_fold_{fold}.txt"
        train_df["img_path"].to_csv(train_file_list, header=False, index=False)
        val_df["img_path"].to_csv(val_file_list, header=False, index=False)

        # train.py にわたす設定ファイル
        data = dict(
            train =  train_file_list ,
            val   =  val_file_list,
            nc    = 14,
            names = [f"{i}" for i in range(14)]
            )
        yaml_file = os.path.join(yolo_data_dir, f'yolo_{fold}.yaml')

        with open(yaml_file, 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)
        f = open(yaml_file, 'r')
        f.close()

    for f in range(CFG["fold_num"]):
        create_path_file(fold_csv, YOLO_DATA_DIR, fold=f)

if __name__ == '__main__':
    main()
