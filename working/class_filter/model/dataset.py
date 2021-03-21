
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

import os
import sys

from .utils import get_img, rand_bbox

class XrayDataset(Dataset):
    def __init__(self, df, data_root,
                 shape, # 追加
                 transforms=None,
                 output_label=True,
                 one_hot_label=False,
                 do_fmix=False,
                 fmix_params={
                     'alpha': 1.,
                     'decay_power': 3.,
                     'max_soft': True,
                     'reformulate': False
                 },
                 do_cutmix=False,
                 cutmix_params={
                     'alpha': 1,
                 },
                 image_name_col = "image_id",
                 label_col="label",
                 use_meta=False
                ):

        super().__init__()
        self.shape = shape
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.do_cutmix = do_cutmix
        self.cutmix_params = cutmix_params
        self.use_meta = use_meta

        self.output_label = output_label
        self.one_hot_label = one_hot_label
        self.image_name_col = image_name_col
        self.label_col = label_col

        if output_label == True:
            self.labels = self.df[self.label_col].values
            #print(self.labels)

            if one_hot_label is True:
                self.labels = np.eye(self.df[self.label_col].max()+1)[self.labels]
                #print(self.labels)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):

        # get labels
        if self.output_label:
            target = self.labels[index]

        if self.use_meta:
            img  = get_img(f"{self.data_root}/{self.df.loc[index][self.image_name_col]}.png",use_meta=self.use_meta, ch1=self.df.loc[index]["ch1"] , ch2=self.df.loc[index]["ch2"])
        else:
            img  = get_img(f"{self.data_root}/{self.df.loc[index][self.image_name_col]}.png", use_meta=self.use_meta)

        if self.transforms:
            img = self.transforms(image=img)['image']

        if self.do_cutmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
            #print(img.sum(), img.shape)
            with torch.no_grad():
                cmix_ix = np.random.choice(self.df.index, size=1)[0]
                cmix_img  = get_img("{}/{}".format(self.data_root, self.df.iloc[cmix_ix]['Image']))
                if self.transforms:
                    cmix_img = self.transforms(image=cmix_img)['image']

                lam = np.clip(np.random.beta(self.cutmix_params['alpha'], self.cutmix_params['alpha']),0.3,0.4)
                bbx1, bby1, bbx2, bby2 = rand_bbox(self.shape, lam)

                img[:, bbx1:bbx2, bby1:bby2] = cmix_img[:, bbx1:bbx2, bby1:bby2]

                rate = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (self.shape[0]*self.shape[1]))
                target = rate*target + (1.-rate)*self.labels[cmix_ix]

        if self.output_label == True:
            return img, target
        else:
            return img
