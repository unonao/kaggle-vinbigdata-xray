import argparse
import json
import os
import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import  log_loss

# 引数で config の設定を行う
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.json')
options = parser.parse_args()
CFG = json.load(open(options.config))

# logger の設定
from logging import getLogger, StreamHandler,FileHandler, Formatter, DEBUG, INFO
logger = getLogger("logger")    #logger名loggerを取得
logger.setLevel(DEBUG)  #loggerとしてはDEBUGで
#handler1を作成
handler_stream = StreamHandler()
handler_stream.setLevel(DEBUG)
handler_stream.setFormatter(Formatter("%(asctime)s: %(message)s"))
#handler2を作成
config_filename = os.path.splitext(os.path.basename(options.config))[0]
handler_file = FileHandler(filename=f'./logs/inference_{config_filename}_{CFG["model_arch"]}.log')
handler_file.setLevel(DEBUG)
handler_file.setFormatter(Formatter("%(asctime)s: %(message)s"))
#loggerに2つのハンドラを設定
logger.addHandler(handler_stream)
logger.addHandler(handler_file)

from model.transform import get_train_transforms, get_valid_transforms, get_inference_transforms
from model.dataset import XrayDataset
from model.model import XrayImgClassifierEfficientnet, XrayImgClassifierVit
from model.epoch_api import train_one_epoch, valid_one_epoch, inference_one_epoch
from model.utils import seed_everything



MAIN_PATH = '../../input/vinbigdata-chest-xray-abnormalities-detection/'
TRAIN_PATH = f'../../input/vinbigdata-chest-xray-resized-png-{CFG["dim"]}x{CFG["dim"]}{CFG["way"]}/train'
TEST_PATH = f'../../input/vinbigdata-chest-xray-resized-png-{CFG["dim"]}x{CFG["dim"]}{CFG["way"]}/test'


test = pd.DataFrame()
test['image_id'] = [os.path.splitext(f)[0] for f in os.listdir(TEST_PATH)]

def load_train_df(path):
    train_df = pd.read_csv(path)
    is_normal_df = train_df.groupby("image_id")["class_id"].agg(lambda s: (s == 14).sum()).reset_index().rename({"class_id": "num_normal_annotations"}, axis=1)
    is_normal_df["label"] = (is_normal_df["num_normal_annotations"]==3).astype(int) # 3人とも異常なしを1とする
    return is_normal_df[["image_id", "label"]]

def infer():
    logger.debug("pred start")
    train = load_train_df(os.path.join(MAIN_PATH,"train.csv"))
    seed_everything(CFG['seed'])

    folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']).split(np.arange(train.shape[0]), train.label.values)


    val_loss = []
    val_acc = []

    # 行数を揃えた空のデータフレームを作成
    cols = ["0","1"]
    oof_df = pd.DataFrame(index=[i for i in range(train.shape[0])],columns=cols)
    y_preds_df = pd.DataFrame(index=[i for i in range(test.shape[0])], columns=cols)
    oof_df['image_id'] = train['image_id']
    y_preds_df['image_id'] = test['image_id']
    y_preds_df.loc[:, cols] = 0

    for fold, (trn_idx, val_idx) in enumerate(folds):
        """
        if fold > 0:
            break
        """
        logger.debug(' fold {} started'.format(fold))
        input_shape=(CFG["dim"], CFG["dim"])

        valid_ = train.loc[val_idx,:].reset_index(drop=True)
        valid_ds = XrayDataset(valid_, TRAIN_PATH, transforms=get_inference_transforms(input_shape,CFG["transform_way"]), shape = input_shape, output_label=False)

        test_ds = XrayDataset(test, TEST_PATH, transforms=get_inference_transforms(input_shape,CFG["transform_way"]),shape=input_shape, output_label=False)


        val_loader = torch.utils.data.DataLoader(
            valid_ds,
            batch_size=CFG['valid_bs'],
            num_workers=CFG['num_workers'],
            shuffle=False,
            pin_memory=False,
        )

        tst_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=CFG['valid_bs'],
            num_workers=CFG['num_workers'],
            shuffle=False,
            pin_memory=False,
        )

        device = torch.device(CFG['device'])

        if CFG["model"]=="efficientnet":
            model = XrayImgClassifierEfficientnet(CFG['model_arch'], train.label.nunique(), pretrained=True).to(device)
        elif CFG["model"]=="vit":
            model = XrayImgClassifierVit(CFG['model_arch'], train.label.nunique(), pretrained=True).to(device)

        val_preds = []
        tst_preds = []


        #for epoch in range(CFG['epochs']-3):
        for i, epoch in enumerate(CFG['used_epochs']):
            model.load_state_dict(torch.load(f'save/{config_filename}_{CFG["model_arch"]}_fold_{fold}_{epoch}'))

            with torch.no_grad():
                for _ in range(CFG['tta']):
                    val_preds += [CFG['weights'][i]/sum(CFG['weights'])*inference_one_epoch(model, val_loader, device)]
                    tst_preds += [CFG['weights'][i]/sum(CFG['weights'])*inference_one_epoch(model, tst_loader, device)]

        val_preds = np.sum(val_preds, axis=0)
        tst_preds = np.sum(tst_preds, axis=0)/(CFG['fold_num'])
        val_loss.append(log_loss(valid_.label.values, val_preds))
        val_acc.append((valid_.label.values == np.argmax(val_preds, axis=1)).mean())
        oof_df.loc[val_idx, cols] = val_preds
        y_preds_df.loc[:, cols] += tst_preds #.reshape(len(tst_preds), -1)

    logger.debug('validation loss = {:.5f}'.format(np.mean(val_loss)))
    logger.debug('validation accuracy = {:.5f}'.format(np.mean(val_acc)))


    # 予測値を保存
    oof_df["pred"] = oof_df["1"]
    y_preds_df["pred"] = y_preds_df["1"]
    oof_df = oof_df.drop(cols, axis = 1)
    y_preds_df = y_preds_df.drop(cols, axis=1)

    oof_df.to_csv(f'output/{config_filename}_{CFG["model_arch"]}_oof.csv', index=False)
    y_preds_df.to_csv(f'output/{config_filename}_{CFG["model_arch"]}_test.csv', index=False)

    del model
    torch.cuda.empty_cache()
    return np.argmax(tst_preds, axis=1)


if __name__ == '__main__':
    logger.debug(CFG)
    tst_preds_label_all = infer()
    print(tst_preds_label_all.shape)
    print("0:", (tst_preds_label_all==0).sum())
    print("1:", (tst_preds_label_all==1).sum())
