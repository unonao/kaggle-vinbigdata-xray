import argparse
import json
import os
import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold

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
handler_file = FileHandler(filename=f'./logs/{config_filename}_{CFG["model_arch"]}.log')
handler_file.setLevel(DEBUG)
handler_file.setFormatter(Formatter("%(asctime)s: %(message)s"))
#loggerに2つのハンドラを設定
logger.addHandler(handler_stream)
logger.addHandler(handler_file)

MAIN_PATH = '../../input/vinbigdata-chest-xray-abnormalities-detection/'
TRAIN_PATH = f'../../input/vinbigdata-chest-xray-resized-png-{CFG["dim"]}x{CFG["dim"]}{CFG["way"]}/train'
TRAIN_META = "../../input/dicom-meta/train_meta.csv"

def load_train_df():
    path = os.path.join(MAIN_PATH,"train.csv")
    train_df = pd.read_csv(path)
    is_normal_df = train_df.groupby("image_id")["class_id"].agg(lambda s: (s == 14).sum()).reset_index().rename({"class_id": "num_normal_annotations"}, axis=1)
    is_normal_df["label"] = (is_normal_df["num_normal_annotations"] == 3).astype(int)  # 3人とも異常なしを1とする
    # meta情報を結合
    meta = pd.read_csv(TRAIN_META)
    meta = meta[["FileName", "PixelSpacing0", "PixelSpacing1","PatientSex"]]
    meta["image_id"] = meta["FileName"].str.replace('.dicom', '')
    meta = meta.drop(["FileName"], axis=1)
    is_normal_df = is_normal_df[["image_id", "label"]].merge(meta, how="left", on="image_id")
    """
    is_normal_df["ch1"] = is_normal_df["PixelSpacing0"].fillna(1.)
    is_normal_df["ch2"] = 0
    """
    is_normal_df["PatientSex"] = is_normal_df["PatientSex"].fillna("no")
    is_normal_df["ch1"] = ((is_normal_df["PatientSex"]=="O")|(is_normal_df["PatientSex"]=="no")).astype(int)
    is_normal_df["ch2"] = 0

    print(is_normal_df)
    is_normal_df["label"] = is_normal_df["label"].astype(int)
    return is_normal_df

def main():
    from model.transform import get_train_transforms, get_valid_transforms
    from model.dataloader import prepare_dataloader
    from model.model import XrayImgClassifierEfficientnet, XrayImgClassifierVit
    from model.epoch_api import train_one_epoch, valid_one_epoch
    from model.utils import seed_everything

    logger.debug(CFG)
    train = load_train_df()
    seed_everything(CFG['seed'])

    folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']).split(np.arange(train.shape[0]), train.label.values)

    for fold, (trn_idx, val_idx) in enumerate(folds):
        """
        if fold > 0:
            break
        """
        logger.debug(f'Training with fold {fold} started (train:{len(trn_idx)}, val:{len(val_idx)})')

        train_loader, val_loader = prepare_dataloader(train, (CFG["resize_dim"], CFG["resize_dim"]), trn_idx, val_idx, data_root=os.path.join(TRAIN_PATH), train_bs=CFG["train_bs"], valid_bs=CFG["valid_bs"], num_workers=CFG["num_workers"], do_fmix=False, do_cutmix=False, transform_way=CFG["transform_way"], use_meta = CFG["meta"])

        device = torch.device(CFG['device'])


        if CFG["model"]=="efficientnet":
            model = XrayImgClassifierEfficientnet(CFG['model_arch'], train.label.nunique(), pretrained=True).to(device)
        elif CFG["model"]=="vit":
            model = XrayImgClassifierVit(CFG['model_arch'], train.label.nunique(), pretrained=True).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=CFG['epochs']-1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG['T_0'], T_mult=1, eta_min=CFG['min_lr'], last_epoch=-1)
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=25,
        #                                                max_lr=CFG['lr'], epochs=CFG['epochs'], steps_per_epoch=len(train_loader))

        loss_tr = nn.CrossEntropyLoss().to(device) #MyCrossEntropyLoss().to(device)
        loss_fn = nn.CrossEntropyLoss().to(device)

        for epoch in range(CFG['epochs']):
            train_one_epoch(epoch, model, loss_tr, optimizer, train_loader, device, CFG['accum_iter'], CFG['verbose_step'],scheduler=scheduler, schd_batch_update=False)

            with torch.no_grad():
                valid_one_epoch(epoch, model, loss_fn, val_loader, device, CFG['accum_iter'], CFG['verbose_step'], scheduler=None, schd_loss_update=False)

            torch.save(model.state_dict(),f'save/{config_filename}_{CFG["model_arch"]}_fold_{fold}_{epoch}')

        del model, optimizer, train_loader, val_loader,  scheduler
        torch.cuda.empty_cache()
        logger.debug("\n")

if __name__ == '__main__':
    main()
