import pathlib
import pandas as pd
import numpy as np
import tqdm
import torch
import gzip
from common import YARDS_CLIP

def preprocess(datadir):
    cols_to_load = ['GameId', 'PlayId', 'Team', 'X', 'Y', 'Dir', 'Yards', 'Dis', 'HomeTeamAbbr', 'PossessionTeam',
                    'NflIdRusher', 'NflId', 'PlayDirection', 'FieldPosition', 'YardLine', 'Season']
    train = pd.read_csv(datadir/'train.csv', low_memory=False)[cols_to_load]

    # fix team abbr
    abbr_corrections = {'BLT': 'BAL', 'CLV': 'CLE', 'ARZ': 'ARI', 'HST': 'HOU'}
    for k, v in abbr_corrections.items():
        train.loc[train.PossessionTeam == k, 'PossessionTeam'] = v
        train.loc[train.FieldPosition == k, 'FieldPosition'] = v
        train.loc[train.HomeTeamAbbr == k, 'HomeTeamAbbr'] = v

    # find offender/defender and rusher
    train['Offender'] = (train.HomeTeamAbbr == train.PossessionTeam) & (train.Team == 'home') | (train.HomeTeamAbbr != train.PossessionTeam) & (train.Team == 'away')
    train['Rusher'] = train.NflIdRusher == train.NflId
    train['dir']= -(train.Dir*np.pi/180. - np.pi/2.) # adjusted & radian
    train['Y_aug'] = 53.33 - train['Y'] # y coordinates flipe

    assert train.loc[train.FieldPosition.isna() & (train.YardLine != 50)].shape[0] == 0

    # adjust yardline to x-axis
    __mask = ~train.FieldPosition.isna() & ((train.FieldPosition != train.PossessionTeam) & (train.PlayDirection == 'right') | (train.FieldPosition == train.PossessionTeam) & (train.PlayDirection == 'left'))
    train.loc[__mask, 'YardLine'] = 100 - train.loc[__mask, 'YardLine']

    # fix Speed column for 2017 season
    __2017_season = train.Season == 2017
    train.loc[__2017_season, 'S'] = 10 * train.loc[__2017_season,'Dis']

    train['S_dx'] = train.S * np.cos(train.dir)
    train['S_dy'] = train.S * np.sin(train.dir)
    train['S_dy_aug'] = -train['S_dy']

    # make it always from left to right
    __mask = (train.PlayDirection == 'left')
    train.loc[__mask, 'X'] = 120 - train.loc[__mask, 'X'] # range 0 ~ 120
    train.loc[__mask, 'Y'] = 53.33 - train.loc[__mask, 'Y']
    train.loc[__mask, 'Y_aug'] = 53.33 - train.loc[__mask, 'Y_aug']
    train.loc[__mask, 'S_dx'] = -train.loc[__mask, 'S_dx']
    train.loc[__mask, 'YardLine'] = 100 - train.loc[__mask, 'YardLine']

    # create augmented feature for all rows and select during training
    play_df = train.groupby('PlayId')['Yards', 'YardLine', 'Season', 'GameId'].first()
    play_df.reset_index(inplace=True)
    play_df['YardsClipped'] = play_df.Yards.clip(YARDS_CLIP[0], YARDS_CLIP[1])

    cols = ['X', 'Y', 'S_dx', 'S_dy', 'X', 'Y_aug', 'S_dx', 'S_dy_aug']

    features = []
    for play_id, g in tqdm.tqdm(train.groupby('PlayId')):
        offense = g.loc[g.Offender & ~g.Rusher, cols].values
        diffense = g.loc[~g.Offender, cols].values
        rusher = g.loc[g.Rusher, cols].values

        f12 = diffense[:,None] - offense[None]
        f34 = np.repeat((diffense - rusher)[:,None], repeats=10, axis=1)
        f5 = np.repeat(diffense[:,[2,3,6,7]][:,None], repeats=10, axis=1)
        #f = torch.Tensor(np.concatenate([f12, f34, f5], axis=-1))
        f = np.concatenate([f12, f34, f5], axis=-1)
        f = torch.Tensor(f[..., [0,1,2,3,8,9,10,11,16,17,4,5,6,7,12,13,14,15,18,19]])
        features.append(f)

    features = np.stack(features).transpose((0, 3, 1, 2))
    play_ids = play_df.PlayId.values
    yards = play_df.Yards.values
    yards_clipped = play_df.YardsClipped.values
    yard_lines = play_df.YardLine.values
    yard_grid = np.arange(-99, 100)[None]
    yard_mask = ((yard_grid <= (100 - yard_lines[:,None])) & (yard_grid >= -yard_lines[:,None]))
    seasons = play_df.Season.values
    game_ids = play_df.GameId.values

    D = [game_ids, play_ids, features, yards, yards_clipped, yard_lines, seasons, yard_mask]
    assert all(D[0].shape[0] == d.shape[0] for d in D)

    idxs_2017 = play_df.loc[play_df.Season == 2017].index.tolist()
    idxs = play_df.loc[play_df.Season > 2017].index.tolist()

    #idxs = np.random.permutation(idxs).tolist()
    #split_ix = int(val_split * len(idxs))
    #idxs_train = idxs[:split_ix] + idxs_2017
    #idxs_val = idxs[split_ix:]
    #D_train = tuple([d[idxs_train] for d in D])
    #D_val = tuple([d[idxs_val] for d in D])

    return D, idxs_2017, idxs

#def load_data(rootdir):
#    train_fn = 'train.pkl.gz'
#    val_fn = 'val.pkl.gz'
#    with gzip.open(rootdir/'processed'/train_fn, "rb") as f:
#        train = torch.load(f)
#    with gzip.open(rootdir/'processed'/val_fn, "rb") as f:
#        val = torch.load(f)
#    return train, val

def load_data(rootdir):
    with gzip.open(rootdir/'processed'/'data.pkl.gz', "rb") as f:
        data = torch.load(f)
    return data

if __name__ == '__main__':
    rootdir = pathlib.Path('.')
    data = preprocess(rootdir/'data')

    with gzip.open(rootdir/'processed'/'data.pkl.gz', 'wb') as f:
        torch.save(data, f)

    #with gzip.open(rootdir/'processed'/'val.pkl.gz', 'wb') as f:
    #    torch.save(D_val, f)
