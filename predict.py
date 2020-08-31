import torch
import pathlib
import gzip
from torch.utils.data import TensorDataset, DataLoader

def load_dataset(rootdir):
    train_fn = 'train.pkl.gz'
    val_fn = 'val.pkl.gz'
    with gzip.open(rootdir/train_fn, "rb") as f:
        train = torch.load(f)
    with gzip.open(rootdir/val_fn, "rb") as f:
        val = torch.load(f)
    return TensorDataset(*train), TensorDataset(*val)

with open('models/shallow/model.pt', 'rb') as f:
    model = torch.load(f)

_, val_dataset = load_dataset(pathlib.Path('.')/'processed')
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, drop_last=True)
dev = "cuda"

n = torch.arange(-99, 100)[None,:].to(dev)
def CRPSLoss(y_pred, y):
    h = ((n - y[:,None]) >= 0).float()
    return torch.mean((y_pred.cumsum(-1) - h)**2)

model = model.to(dev)
model.eval()
total_loss = 0
with torch.no_grad():
    for _batch in val_loader:
        features, y = [t.to(dev) for t in _batch[1:]]
        y_pred = model(features)
        loss = CRPSLoss(y_pred, y)
        total_loss += loss

print(total_loss / len(val_loader))
