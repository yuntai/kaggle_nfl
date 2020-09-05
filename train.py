import os
import json
import time
import gzip
import pathlib
from itertools import islice
import pprint
import argparse

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import sklearn
from sklearn.model_selection import GroupKFold
from sklearn.utils import resample

from setproctitle import setproctitle
from termcolor import colored

from dataset import RushDataset
from net import NFLRushNet
from common import CRPSLoss
from prep import load_data


def parse_args():
    parser = argparse.ArgumentParser(description='NFL yardage prediction')
    parser.add_argument('-b', '--bsz', type=int, default=64, help='location of the (processed) data')
    parser.add_argument('--seed', type=int, default=1111, help="random seed")
    parser.add_argument('--nocuda', action='store_true', help="do not run with gpu")
    parser.add_argument('--name', type=str, default="nflrush", help="name of the experiment")
    parser.add_argument('--fp16', action='store_true', help="use mixed precision from apex to save memory")
    parser.add_argument('--max_epochs', type=int, default=50, help='upper epoch limit')

    parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate (0.0001|5 for adam|sgd)')
    parser.add_argument('-s', '--scheduler', default='onecycle', type=str, choices=['cosine', 'inv_sqrt', 'dev_perf','onecycle']),

    # Cosine
    parser.add_argument('--eta_min', type=float, default=1e-7, help='min learning rate for cosine scheduler')

    # dev_perf
    parser.add_argument('--warmup_step', type=int, default=5000, help='upper epoch limit')

    # ReduceLROnPlateau
    parser.add_argument('--decay_rate', type=float, default=0.5, help='decay factor when ReduceLROnPlateau is used')
    parser.add_argument('--patience', type=int, default=5, help='patience')
    parser.add_argument('--lr_min', type=float, default=0.0, help='minimum learning rate during annealing')

    # Onecycle
    parser.add_argument('--lr_max', type=float, default=0.001, help='maximum learning rate in onecycle scheduler')

    parser.add_argument('--grad_clip', type=float, default=0.25, help='gradient clipping')
    parser.add_argument('--log_interval', type=int, default=200, help='report interval')
    parser.add_argument('--multi_gpu', type=bool, default=True, help='use multiple GPU')
    parser.add_argument('--noval', action='store_true', help='do not run validation')

    # model
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')

    # cv & bagging
    parser.add_argument('--n_splits', type=int, default=5, help='do not run validation')
    parser.add_argument('--bagging_p', type=float, default=0.8, help='bagging ratio')
    parser.add_argument('--bagging_size', type=int, default=0, help='bagging ratio')

    return parser.parse_args()

args = parse_args()

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

from utils import Experiment
exp = Experiment(rootdir/"models"/args.name)
logging = exp.get_logger()

logging('=' * 60)
pprint.pprint(args.__dict__)
logging('=' * 60)

# tensorboard stuff
writer = SummaryWriter()

def run_epoch(model, loader, opt=None, scheduler=None, _epoch=-1):
    model.eval() if opt is None else model.train()
    dev = next(model.parameters()).device
    batch_id = 0
    total_loss = 0 # loss for log interval
    epoch_loss = 0

    with torch.enable_grad() if opt else torch.no_grad():
        for _batch in loader:
            X, y, mask = [t.to(dev) for t in _batch]
            y_pred = model(X)
            y_pred = F.softmax(y_pred, dim=-1)
            loss = CRPSLoss(y_pred, y)
            batch_loss = loss.detach().to('cpu').numpy()
            epoch_loss += batch_loss
            total_loss += batch_loss

            if opt:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                if args.scheduler not in ['dev_perf']:
                    scheduler.step()

                #batch_id += 1
                #if batch_id % args.log_interval == 0:
                #    avg_loss = total_loss / args.log_interval
                #    print(f"                    LR {opt.param_groups[0]['lr']:.7f} | Loss {avg_loss:.5f}")
                #    total_loss = 0

    #torch.cuda.empty_cache()
    return epoch_loss / len(loader)

def setup_train(max_steps):
    model = NFLRushNet()
    n_all_param = sum([p.nelement() for p in model.parameters() if p.requires_grad])
    logging(f'#params = {n_all_param/1e6:.2f}M')

    # initilize optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, max_steps, eta_min=args.eta_min)
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
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr_max, total_steps=max_steps)

    if args.multi_gpu:
        model = model.to(device)
        para_model = nn.DataParallel(model).to(device)
    else:
        para_model = model.to(device)

    return para_model, optimizer, scheduler

def run_cv(X, X_aug, y, y_clipped, mask, groups, ixs_2017, n_splits=5):
    cv = GroupKFold(n_splits=n_splits)

    D = [X, X_aug, y, y_clipped, mask, groups]
    D_2017 = [d[ixs_2017] for d in D]

    ixs = list(set(range(X.shape[0])) - set(ixs_2017)) # non 2017 data
    D = [d[ixs] for d in D]

    X, X_aug, y, y_clipped, mask, groups = D
    losses = []

    for fold_ix, (train_ix, test_ix) in enumerate(cv.split(X, y, groups)):
        D_tr = [np.concatenate([x[train_ix], x_2017]) for x, x_2017 in zip(D, D_2017)]
        X_tr, X_aug_tr, y_tr, y_clipped_tr, mask_tr, groups_tr = D_tr

        train_dataset = RushDataset(X_tr, X_aug_tr, y_clipped_tr, mask_tr)
        val_dataset = RushDataset(X[test_ix], X_aug[test_ix], y_clipped[test_ix], mask[test_ix], aug=False)

        #assert len(set(test_ix) & set(ixs_2017)) == 0, "test set should not contain data from 2017"
        train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=args.bsz, shuffle=False, drop_last=True)

        meta = {'train_ix': train_ix, 'test_ix': test_ix}
        val_loss = train_loop(train_loader, val_loader, fold_ix, meta)
        losses.append(val_loss)

    losses = np.array(losses)
    print(f"cv loss avg({losses.mean()}) ±({losses.std()})")

def train_loop(train_loader, val_loader, fold_ix=0, meta={}):

    max_steps = args.max_epochs * len(train_loader)

    logging(f'epoch step size = {len(train_loader)}')
    logging(f'max steps = {max_steps}')

    model, opt, scheduler = setup_train(max_steps)

    train_step = 0
    start_epoch = 1
    best_val_loss = float('inf')

    for i in range(start_epoch, args.max_epochs+1):
        start = time.time()
        train_loss = run_epoch(model, train_loader, opt, scheduler, _epoch=i)
        writer.add_scalar(f'loss/{fold_ix}/train', train_loss, i)
        if not args.noval and val_loader is not None:
            val_loss = run_epoch(model, val_loader)
            end = time.time()
            msg = f"{fold_ix}|{i:2d}|{end-start:.2f}s|lr({opt.param_groups[0]['lr']:.7f})|tr loss({train_loss:.5f})|val loss({val_loss:.5f})"
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                meta['val_loss'] = val_loss
                meta['train_loss'] = train_loss
                exp.save_checkpoint(model, opt, epoch=i, fold_ix=fold_ix, meta=meta)
                msg = f"{colored(msg, 'yellow')}"
            writer.add_scalar(f'loss/{fold_ix}/val', val_loss)
        else:
            end = time.time()
            msg = f"epoch {i:2d}|{end-start:.2f}s|lr({opt.param_groups[0]['lr']:.7f})|tr loss({train_loss:.5f})"
        print(msg)
        lr = opt.param_groups[0]['lr']
        writer.add_scalar(f'learning_rate/{fold_ix}', lr, i)

        if args.scheduler == 'dev_perf':
            scheduler.step(val_loss if not args.noval else train_loss)

    print(f"best val={best_val_loss}")
    return best_val_loss

if __name__ == '__main__':
    X, X_aug, y, y_clipped, mask, groups, idx_2017 = load_data(pathlib.Path('.'))
    run_cv(X, X_aug, y, y_clipped, mask, groups, idx_2017, args.n_splits)
