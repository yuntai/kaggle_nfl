import torch
from torch.utils.data import Dataset, DataLoader
from itertools import islice
import gzip
import numpy as np
import pathlib

class RushDataset(Dataset):
    def __init__(self, X, X_aug, y, mask, aug=True, aug_p=0.5):
        self.X = torch.Tensor(X)
        self.X_aug = torch.Tensor(X_aug)
        self.y = torch.Tensor(y)
        self.mask = torch.Tensor(mask)
        self.aug = aug
        self.aug_p = aug_p

    def __getitem__(self, idx):
        if self.aug and np.random.binomial(1, self.aug_p):
            return self.X_aug[idx], self.y[idx], self.mask[idx]
        else:
            return self.X[idx], self.y[idx], self.mask[idx]

    def __len__(self):
        return self.X.shape[0]
