import pathlib
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from prep import load_data
from dataset import RushDataset
from common import CRPSLoss
from itertools import islice


rootdir = pathlib.Path(".")
name = 'nflrush'
exproot = rootdir/'models'/name

D = load_data(rootdir)
assert all(d.shape[0]==D[0].shape[0] for d in D[:-1])
X, X_aug, y, y_clipped, mask, groups, idx_2017 = D

ixs = list(set(list(range(X.shape[0]))) - set(idx_2017)) # non 2017 data

members = []
dev = 'cuda'
for _dir in islice(exproot.iterdir(), 1):
    with open(_dir/'model.pt', 'rb') as f:
        model = torch.load(f)
    model.to(dev)
    model.eval()
    members.append(model)


with torch.no_grad():
    loss = 0
    for _batch in val_loader:
        X, X_aug, y, _ = [t.to(dev) for t in _batch]
        bsz = X.shape[0]

        y_pred = 0
        for model in members:
            pred = model(X)
            pred_aug = model(X_aug)
            y_pred += (F.softmax(pred, dim=-1) + F.softmax(pred_aug, dim=-1))/2.

        y_pred /= len(members)
        loss += CRPSLoss(y_pred, y).detach().to('cpu').numpy()
avg_loss = loss / len(val_loader)

print(avg_loss)
