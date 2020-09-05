import torch
from torch.utils.data import Dataset, DataLoader
from itertools import islice
import gzip
import numpy as np
import pathlib

class RushDataset(Dataset):
    def __init__(self, X, X_aug, y, mask, aug=True, aug_p=0.5, tta=False):
        D = [X, X_aug, y, mask]
        assert all(d.shape[0] == D[0].shape[0] for d in D)

        self.X = [torch.Tensor(X_aug), torch.Tensor(X)]
        self.y = torch.Tensor(y)
        self.mask = torch.Tensor(mask)
        self.aug = aug
        self.aug_p = aug_p
        self.tta = tta

    def __getitem__(self, idx):
        if not self.tta:
            xix = int(self.aug and np.random.binomial(1, self.aug_p))
            return self.X[xix][idx], self.y[idx], self.mask[idx]
        else:
            return self.X[0][idx], self.X[1][idx], self.y[idx], self.mask[idx]

    def __len__(self):
        return self.X[0].shape[0]
