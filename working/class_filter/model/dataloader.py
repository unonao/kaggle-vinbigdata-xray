import numpy as np
import pandas as pd
import torch


from .transform import get_train_transforms, get_valid_transforms
from .dataset import XrayDataset


def prepare_dataloader(df, input_shape, trn_idx, val_idx, data_root, train_bs, valid_bs, num_workers, do_fmix=False, do_cutmix=False, transform_way = "resize"):
    """
    データローダーを作成する関数
    """

    train_ = df.loc[trn_idx,:].reset_index(drop=True)
    valid_ = df.loc[val_idx,:].reset_index(drop=True)

    train_ds = XrayDataset(train_, data_root ,input_shape, transforms=get_train_transforms(input_shape,transform_way), output_label=True, one_hot_label=False, do_fmix=do_fmix, do_cutmix=do_cutmix)
    valid_ds = XrayDataset(valid_, data_root ,input_shape, transforms=get_valid_transforms(input_shape,transform_way), output_label=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=train_bs,
        pin_memory=True, # faster and use memory
        drop_last=False,
        shuffle=True,
        num_workers=num_workers,
        #sampler=BalanceClassSampler(labels=train_['label'].values, mode="downsampling")
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=valid_bs,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False,
    )
    return train_loader, val_loader
