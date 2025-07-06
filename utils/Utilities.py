# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as np

def create_dataset(dataset, features_columns, target_column, p):
    T = dataset.shape[0]
    X = torch.stack([torch.from_numpy(dataset[features_columns].iloc[i-p:i].values).float() for i in range(p, T)])
    Y = torch.stack([torch.from_numpy(dataset[target_column].iloc[i-p+1:i+1].values).float() for i in range(p, T)])
    return X, Y

def mar_dataset(dataset, features_columns, p):
    T = dataset.shape[0]
    X = torch.stack([torch.from_numpy(dataset[features_columns].iloc[i-p:i].values).float() for i in range(p, T)])
    Y = torch.from_numpy(dataset[features_columns].iloc[p:].values).float()
    return X, Y

def mar_dataset_seperate_target(dataset, features_columns, target_column, p):
    T = dataset.shape[0]
    X = torch.stack([torch.from_numpy(dataset[features_columns].iloc[i-p:i].values).float() for i in range(p, T)])
    Y = torch.from_numpy(dataset[target_column].iloc[p:].values).float()
    return X, Y