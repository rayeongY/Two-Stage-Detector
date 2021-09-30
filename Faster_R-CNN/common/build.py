import os
import numpy as np

from data.dataset import DataSet

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def build_():
    pass


def build_dataloader(dataset_opt, model_opt, optim_opt):

    train_dataset = DataSet(dataset_opt, model_opt, split="train")
    valid_dataset = DataSet(dataset_opt, model_opt, split="valid")

    print(f"Training set: {len(train_dataset)}")
    print(f"Validation set: {len(valid_dataset)}")

    batch_size = optim_opt["OPTIMIZER"]["BATCH_SIZE"]
    train_loader = DataLoader(
        train_dataset,
        batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size,
        shuffle=True,
        collate_fn=valid_dataset.collate_fn,
    )

    return train_loader, valid_loader, len(train_dataset)


def build_optimizer(optim_opt, params):
    
    ## if optim_opt["OPTIMIZER"]["NAME"] == "adam":
    optimizer = torch.optim.Adam(
        params=params,
        lr = optim_opt["OPTIMIZER"]["LR"],
        weight_decay=optim_opt["OPTIMIZER"]["WD"]
    )

    return optimizer