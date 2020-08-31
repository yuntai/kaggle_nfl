import pathlib
import pandas as pd
import numpy as np
import tqdm
import torch
import gzip
from common import YARDS_CLIP

def preprocess(datadir, val_split=0.8):
    cols_to_load = ['GameId', 'PlayId', 'Team', 'X', 'Y', 'Dir', 'Yards', 'Dis', 'HomeTeamAbbr', 'PossessionTeam',
                    'NflIdRusher', 'NflId', 'PlayDirection', 'FieldPosition', 'YardLine']
    train = pd.read_csv(datadir/'train.csv', low_memory=False)[cols_to_load]

    # fix team abbr
    abbr_corrections = {'BLT': 'BAL', 'CLV': 'CLE', 'ARZ': 'ARI', 'HST': 'HOU'}
    for k, v in abbr_corrections.items():
        train.loc[train.PossessionTeam == k, 'PossessionTeam'] = v
        train.loc[train.FieldPosition == k, 'FieldPosition'] = v

    # find offender/defender and rusher
    train['Offender'] = (train.HomeTeamAbbr == train.PossessionTeam) & (train.Team == 'home') | (train.HomeTeamAbbr != train.PossessionTeam) & (train.Team == 'away')
    train['Rusher'] = train.NflIdRusher == train.NflId
    train['dir']= -(train.Dir*np.pi/180. - np.pi/2.) # adjusted
    train['Y_flipped'] = 53.33 - train['Y'] # y coordinates flipe

    # adjust yardline to x-axis
    __cond = (train.FieldPosition != train.PossessionTeam) & (train.PlayDirection == 'right') | (train.FieldPosition == train.PossessionTeam) & (train.PlayDirection == 'left')
    train.loc[__cond, 'YardLine'] = 100 - train.loc[__cond, 'YardLine']

    # replace speed with Dis field
    train['S'] = train.Dis * 10
    train['Year'] = train.GameId.apply(lambda x: str(x)[:4])

    train['S_dx'] = train.S * np.cos(train.dir)
    train['S_dy'] = train.S * np.sin(train.dir)
    train['S_dy_flipped'] = -train['S_dy']

    # make it always from left to right
    __cond = (train.PlayDirection == 'left')
    train.loc[__cond, 'X'] = 120 - train.loc[__cond, 'X'] # range 0 ~ 120
    train.loc[__cond, 'S_dx'] = -train.loc[__cond, 'S_dx']
    train.loc[__cond, 'YardLine'] = 100 - train.loc[__cond, 'YardLine']

    train.to_csv('train_mod.csv')
    1/0

    features = []
    features2 = []
    cols = ['X', 'Y', 'Y_flipped', 'S_dx', 'S_dy', 'S_dy_flipped']
    for play_id, g in tqdm.tqdm(train.groupby('PlayId')):
        offense = g.loc[g.Offender & ~g.Rusher, cols].values
        diffense = g.loc[~g.Offender, cols].values
        rusher = g.loc[g.Rusher, cols].values

        offense1 = offense[:, [0, 1, 3, 4]]
        diffense1 = diffense[:, [0, 1, 3, 4]]
        rusher1 = rusher[:, [0, 1, 3, 4]]

        f12 = offense1[:,None] - diffense1[None]
        f34 = np.repeat((diffense1 - rusher1)[None], repeats=10, axis=0)
        f5 = np.repeat(diffense1[:,2:][None], repeats=10, axis=0)
        f = torch.Tensor(np.concatenate([f12, f34, f5], axis=-1))
        features.append(f)

        offense2 = offense[:, [0, 2, 3, 5]]
        diffense2 = diffense[:, [0, 2, 3, 5]]
        rusher2 = rusher[:, [0, 2, 3, 5]]

        f12 = offense2[:,None] - diffense2[None]
        f34 = np.repeat((diffense2 - rusher2)[None], repeats=10, axis=0)
        f5 = np.repeat(diffense2[:,2:][None], repeats=10, axis=0)
        f = torch.Tensor(np.concatenate([f12, f34, f5], axis=-1))
        features2.append(f)


    # values fixed within same playid
    vals = train.groupby('PlayId')['PlayId', 'Yards', 'YardLine'].first().values
    play_ids = vals[:, 0]
    yards = vals[:, 1]
    yard_lines = vals[:, 2]
    yards_clipped = np.clip(yards, YARDS_CLIP[0], YARDS_CLIP[1])

    yards_grid = np.arange(-99, 100)[None]
    masks = ((yards_grid <= (100 - yard_lines[:,None])) & (yards_grid >= -yard_lines[:,None]))

    features = torch.stack(features).permute(0, 3, 1, 2).contiguous()
    features2 = torch.stack(features2).permute(0, 3, 1, 2).contiguous()
    play_ids = torch.LongTensor(play_ids)
    yards = torch.IntTensor(yards)
    yards_clipped = torch.IntTensor(yards_clipped)
    masks = torch.IntTensor(masks)

    features = torch.cat([features, features2], dim=0)
    play_ids = play_ids.repeat(2)
    yards = yards.repeat(2)
    yards_clipped = yards_clipped.repeat(2)
    masks = masks.repeat(2, 1)

    D = [play_ids, features, yards, masks, yards_clipped]
    assert all(D[0].size(0) == d.size(0) for d in D)

    np.random.seed(0)
    p = np.random.permutation(D[0].shape[0])

    # Split the training data into train and validation
    split_ix = int(val_split*len(p))

    idx_train = torch.tensor(p[:split_ix])
    idx_val = torch.tensor(p[split_ix:])

    D_train = tuple([d[idx_train] for d in D])
    D_val = tuple([d[idx_val] for d in D])

    return D_train, D_val

if __name__ == '__main__':
    rootdir = pathlib.Path('.')
    D_train, D_val = preprocess(rootdir/'data')

    with gzip.open(rootdir/'processed'/'train.pkl.gz', 'wb') as f:
        torch.save(D_train, f)

    with gzip.open(rootdir/'processed'/'val.pkl.gz', 'wb') as f:
        torch.save(D_val, f)
