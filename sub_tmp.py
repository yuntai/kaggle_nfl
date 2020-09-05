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
dev = 'cuda'

with open(exproot/'0/model.pt', 'rb') as f:
    model = torch.load(f)
    model.eval()
    model.to(dev)

with open(exproot/'0/meta.pt', 'rb') as f:
    meta = torch.load(f)
    test_ix = meta['test_ix']

D = load_data(rootdir)
assert all(d.shape[0]==D[0].shape[0] for d in D[:-1])
X, X_aug, y, y_clipped, mask, groups, idx_2017 = D

ixs = list(range(X.shape[0]))
print(len(set(test_ix) & set(ixs)))
print(len(set(test_ix) & set(idx_2017)))
1/0
ixs = list(set(ixs) - set(idx_2017)) # non 2017 data

ixs = np.random.permutation(ixs)
ixs = ixs[:int(len(ixs) * 0.8)]

val_dataset = RushDataset(X[ixs], X_aug[ixs], y[ixs], mask[ixs], aug=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, drop_last=False)

loss = []
with torch.no_grad():
    for _batch in val_loader:
        X, y, _ = [t.to(dev) for t in _batch]
        y_pred = model(X)
        y_pred = F.softmax(y_pred, dim=-1)
        loss.append(CRPSLoss(y_pred, y).detach().to('cpu').numpy())

loss = np.array(loss)
print(loss.mean(), loss.std())
