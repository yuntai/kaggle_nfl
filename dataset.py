import torch
from torch.utils.data import Dataset, DataLoader
from itertools import islice
import gzip
import numpy as np
import pathlib

class RushDataset(Dataset):
    def __init__(self, X, y, mask, aug=True, aug_p=0.5):
        self.X = torch.Tensor(X)
        self.y = torch.Tensor(y)
        self.mask = torch.Tensor(mask)
        self.aug = aug
        self.aug_p = aug_p
        self.feature_ixs = [[0,1,2,3,8,9,10,11,16,17], [4,5,6,7,12,13,14,15,18,19]]

    def __getitem__(self, idx):
        f_ixs = self.feature_ixs[np.random.binomial(1, self.aug_p) if self.aug else 0]
        return self.X[idx, f_ixs, ...], self.y[idx], self.mask[idx]

    def __len__(self):
        return self.X.shape[0]
