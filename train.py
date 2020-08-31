import os
import json
import time
import gzip
import pathlib
from itertools import islice

from termcolor import colored
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import argparse

from setproctitle import setproctitle

from net import NFLRushNet
from common import YARDS_CLIP

#from utils import radam
#from utils.csv_to_pkl import get_datasets
#from utils.exp_utils import *
#from utils.data_parallel import BalancedDataParallel
#from utils.filters import *

def load_dataset(rootdir):
    train_fn = 'train.pkl.gz'
    val_fn = 'val.pkl.gz'
    with gzip.open(rootdir/'processed'/train_fn, "rb") as f:
        train = torch.load(f)
    with gzip.open(rootdir/'processed'/val_fn, "rb") as f:
        val = torch.load(f)
    return TensorDataset(*train), TensorDataset(*val)

parser = argparse.ArgumentParser(description='NFL yardage prediction')
parser.add_argument('--bsz', type=int, default=64, help='location of the (processed) data')
parser.add_argument('--seed', type=int, default=1111, help="random seed")
parser.add_argument('--nocuda', action='store_true', help="do not run with gpu")
parser.add_argument('--name', type=str, default="nflrush", help="name of the experiment")
parser.add_argument('--fp16', action='store_true', help="use mixed precision from apex to save memory")
parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate (0.0001|5 for adam|sgd)')
parser.add_argument('--max_epochs', type=int, default=50, help='upper epoch limit')
parser.add_argument('-s', '--scheduler', default='onecycle', type=str, choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant',
                                                                        'onecycle']),
parser.add_argument('--eta_min', type=float, default=1e-7, help='min learning rate for cosine scheduler')
parser.add_argument('--warmup_step', type=int, default=5000, help='upper epoch limit')
parser.add_argument('--decay_rate', type=float, default=0.5, help='decay factor when ReduceLROnPlateau is used')
parser.add_argument('--patience', type=int, default=5, help='patience')
parser.add_argument('--lr_min', type=float, default=0.0, help='minimum learning rate during annealing')
parser.add_argument('--lr_max', type=float, default=0.0005, help='maximum learning rate in onecycle scheduler')
parser.add_argument('--grad_clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--log_interval', type=int, default=200, help='report interval')
parser.add_argument('--multi_gpu', type=bool, default=True, help='use multiple GPU')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--noval', action='store_true', help='do not run validation')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if args.nocuda:
        print('WARNING: You have a CUDA device, so you should probably not run with --nocuda')
    else:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.manual_seed_all(args.seed)
device = torch.device('cpu' if args.nocuda else 'cuda')

if args.name:
    setproctitle(args.name)

APEX_AVAILABLE = False
if args.fp16:
    try:
        from apex import amp
        APEX_AVAILABLE = True
    except ModuleNotFoundError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
        APEX_AVAILABLE = False

rootdir = pathlib.Path(".")
train_dataset, val_dataset = load_dataset(rootdir)
train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=args.bsz, shuffle=True, drop_last=True)

# 1. Create a model or load a model
#if args.restart:
#    with open(args.restart_dir/'model.pt', 'rb') as f:
#        model = torch.load(f)
#
#if len(args.load) > 0:
#    model = torch.load(args.load)
#else:

model = NFLRushNet()
n_all_param = sum([p.nelement() for p in model.parameters() if p.requires_grad])

# initilize optimizer and learning rate scheduler
optimizer = optim.Adam(model.parameters(), lr=args.lr)

max_step = args.max_epochs * len(train_loader)
if args.scheduler == 'cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, max_step, eta_min=args.eta_min)
elif args.scheduler == 'inv_sqrt':
    # originally used for Transformer (in Attention is all you need)
    def lr_lambda(step):
        # return a multiplier instead of a learning rate
        if step == 0 and args.warmup_step == 0:
            return 1.
        else:
            return 1. / (step ** 0.5) if step > args.warmup_step else step / (args.warmup_step ** 1.5)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
elif args.scheduler == 'dev_perf':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.decay_rate,
                                                     patience=args.patience, min_lr=args.lr_min)
elif args.scheduler == 'onecycle':
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr_max, epochs=args.max_epochs, steps_per_epoch=len(train_loader))

if args.multi_gpu:
    model = model.to(device)
    para_model = nn.DataParallel(model).to(device)
else:
    para_model = model.to(device)

from utils import Experiment
exp = Experiment(rootdir/"models"/args.name)
logging = exp.get_logger()

logging('=' * 60)
for k, v in args.__dict__.items():
    logging('    - {} : {}'.format(k, v))
logging('=' * 60)
logging(f'#params = {n_all_param/1e6:.2f}M')
logging(f'epoch step size = {len(train_loader)}')

#n = torch.arange(YARDS_CLIP[0], YARDS_CLIP[1]+1)[None]
yards_grid = torch.arange(-99, 100)[None]
def CRPSLoss(y_pred, y):
    h = ((yards_grid - y[:,None]) >= 0).float()
    return torch.mean((y_pred.cumsum(-1) - h)**2)

def run_epoch(loader, model, opt=None, _epoch=-1):
    global train_step
    model.eval() if opt is None else model.train()
    dev = next(model.parameters()).device
    batch_id = 0
    total_loss = 0
    epoch_loss = 0

    with torch.enable_grad() if opt else torch.no_grad():
        for _batch in loader:
            features, _, mask, y = [t.to(dev) for t in _batch[1:]]
            y_pred = para_model(features)
            #y_pred[mask == 0] += -1e8
            #y_pred = F.softmax(y_pred, dim=-1)
            loss = CRPSLoss(y_pred, y)
            batch_loss = loss.detach()
            epoch_loss += batch_loss
            total_loss += batch_loss

            if opt:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                if args.scheduler != 'dev_perf':
                    scheduler.step()

                batch_id += 1
                train_step += 1

                if batch_id % args.log_interval == 0:
                    avg_loss = total_loss / args.log_interval
                    print(f"Epoch {_epoch:2d} | Step {train_step} | LR {opt.param_groups[0]['lr']:.7f} | Loss {avg_loss:.5f}")
                    total_loss = 0

    torch.cuda.empty_cache()
    return epoch_loss / len(loader)

if __name__ == "__main__":
    train_step = 0
    start_epoch = 1
    best_val_loss = float('inf')

    print(optimizer)

    for i in range(start_epoch, args.max_epochs+1):
        start = time.time()
        train_loss = run_epoch(train_loader, model, optimizer, _epoch=i)
        if not args.noval and val_dataset is not None:
            val_loss = run_epoch(val_loader, model)
            end = time.time()
            msg = f"Epoch {i:2d} | {end-start:.2f} sec | LR {optimizer.param_groups[0]['lr']:.7f} | Train Loss {train_loss:.5f} |  Val Loss {val_loss:.5f}"
            if val_loss < best_val_loss:
                exp.save_checkpoint(model, optimizer)
                best_val_loss = val_loss
                msg = f"{colored(msg, 'yellow')}"
        else:
            end = time.time()
            msg = f"Epoch {i:2d} | {end-start:.2f} sec | LR {optimizer.param_groups[0]['lr']:.7f} | Train Loss {train_loss:.5f}"
        print(msg)

        if args.scheduler == 'dev_perf':
            scheduler.step(val_loss if not args.noval else train_loss)
