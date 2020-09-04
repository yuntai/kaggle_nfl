import pathlib
import pandas as pd
import numpy as np
import tqdm
import torch
import gzip
YARDS_CLIP = [-15, 50]

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def preprocess(datadir):
    train = pd.read_csv(datadir/'train.csv', low_memory=False)

    # fix team abbr
    abbr_corrections = {'BLT': 'BAL', 'CLV': 'CLE', 'ARZ': 'ARI', 'HST': 'HOU'}
    for k, v in abbr_corrections.items():
        train.loc[train.PossessionTeam == k, 'PossessionTeam'] = v
        train.loc[train.FieldPosition == k, 'FieldPosition'] = v
        train.loc[train.HomeTeamAbbr == k, 'HomeTeamAbbr'] = v

    # find offender/defender and rusher
    train['Offender'] = (train.HomeTeamAbbr == train.PossessionTeam) & (train.Team == 'home') | (train.HomeTeamAbbr != train.PossessionTeam) & (train.Team == 'away')
    train['Rusher'] = train.NflIdRusher == train.NflId

    # some NA in Dir; should be okay as speed = 0
    assert train[train.Dir.isna() & (train.S != 0.0)].size == 0
    train.Dir.fillna(0., inplace=True)

    train['dir']= -(train.Dir*np.pi/180. - np.pi/2.) # adjusted & radian
    train['Y_aug'] = 53.33 - train['Y'] # y coordinates flipe

    # field postion NA when YardLine = 50
    assert train.loc[train.FieldPosition.isna() & (train.YardLine != 50)].shape[0] == 0

    # adjust yardline to x-axis
    __mask = ~train.FieldPosition.isna() & ((train.FieldPosition != train.PossessionTeam) & (train.PlayDirection == 'right') | (train.FieldPosition == train.PossessionTeam) & (train.PlayDirection == 'left'))
    train.loc[__mask, 'YardLine'] = 100 - train.loc[__mask, 'YardLine']

    # fix Speed column for 2017 season
    __2017_season = train.Season == 2017
    train.loc[__2017_season, 'S'] = 10 * train.loc[__2017_season, 'Dis']

    train['dx'] = train.S * np.cos(train.dir)
    train['dy'] = train.S * np.sin(train.dir)
    train['dy_aug'] = -train['dy']

    # make it always from left to right
    __mask = (train.PlayDirection == 'left')
    train.loc[__mask, 'X'] = 120 - train.loc[__mask, 'X'] # range 0 ~ 120
    train.loc[__mask, 'Y'] = 53.33 - train.loc[__mask, 'Y']
    train.loc[__mask, 'Y_aug'] = 53.33 - train.loc[__mask, 'Y_aug']
    train.loc[__mask, 'dx'] = -train.loc[__mask, 'dx']
    train.loc[__mask, 'YardLine'] = 100 - train.loc[__mask, 'YardLine']

    # create augmented feature for all rows and select during training
    play_group = train.groupby('PlayId')
    play_df = play_group[['Yards', 'YardLine', 'Season', 'GameId']].first().reset_index()
    play_df['YardsClipped'] = play_df.Yards.clip(YARDS_CLIP[0], YARDS_CLIP[1])

    cols = ['X', 'Y', 'dx', 'dy', 'X', 'Y_aug', 'dx', 'dy_aug']

    features = []
    features_aug = []
    for _id, g in tqdm.tqdm(play_group):
        offense = g.loc[g.Offender & ~g.Rusher, cols].values
        diffense = g.loc[~g.Offender, cols].values
        rusher = g.loc[g.Rusher, cols].values

        f12 = diffense[:,None] - offense[None]
        f34 = np.repeat((diffense - rusher)[:,None], repeats=10, axis=1)
        f5 = np.repeat(diffense[:,[2,3,6,7]][:,None], repeats=10, axis=1)

        f     = np.concatenate([f12[...,:4], f34[...,:4], f5[...,:2]], axis=-1)
        f_aug = np.concatenate([f12[...,4:], f34[...,4:], f5[...,2:]], axis=-1)

        features.append(f)
        features_aug.append(f_aug)

    YARD_GRID = np.arange(-99, 100)[None]
    EYE = np.eye(199)

    features = np.stack(features).transpose((0, 3, 1, 2)) # channel first
    features_aug = np.stack(features_aug).transpose((0, 3, 1, 2)) # channel first

    assert np.isnan(features).sum() == 0, f"nan found in features"
    assert np.isnan(features_aug).sum() == 0, f"nan found in features"

    play_ids = play_df.PlayId.values
    yards = EYE[play_df.Yards.values + 99]
    yards_clipped = EYE[play_df.YardsClipped.values + 99]
    yard_lines = play_df.YardLine.values
    yard_mask = ((YARD_GRID <= (100 - yard_lines[:,None])) & (YARD_GRID >= -yard_lines[:,None])).astype(np.int)
    game_ids = play_df.GameId.values
    idxs_2017 = play_df.loc[play_df.Season == 2017].index.tolist()

    D = [features, features_aug, yards, yards_clipped, yard_mask, game_ids]
    assert all(D[0].shape[0] == d.shape[0] for d in D)
    D += [idxs_2017]

    return D

def load_data(rootdir):
    with gzip.open(rootdir/'processed/data.pkl.gz', "rb") as f:
        return torch.load(f)

if __name__ == '__main__':
    rootdir = pathlib.Path('.')
    data = preprocess(rootdir/'data')
    with gzip.open(rootdir/'processed/data.pkl.gz', 'wb') as f:
        torch.save(data, f)

