import os
import random
import pandas as pd
import cv2
import numpy as np
import torch

def seed_everything(seed):
    "seed値を一括指定"
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_img(path, use_meta=False ,ch1=0.0, ch2=0.0):
    """
    pathからimageの配列を得る
    """
    im_three = cv2.imread(path)
    if use_meta:
        im_three[:,:,1] = int(ch1*255.0)
        im_three[:,:,2] = int(ch2*255.0)
    return im_three

def rand_bbox(size, lam):
    """
    ランダムなボックスを出力する
    lam の大きさに従って、ランダムな位置でカットした正方形を出力する。はみ出たらクリップしてカット
    """
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2
