# based on https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/119314
# CV
- Always include 2017 data for training, 3 group folds by week for 2018 data, use only 2018 data for evaluation.

# Features
total 36 features, ['IsRusher','IsRusherTeam','X','Y','Dir_X','Dir_Y',
'Orientation_X','Orientation_Y','S','DistanceToBall',
'BallDistanceX','BallDistanceY','BallAngleX','BallAngleY',
'related_horizontal_v','related_vertical_v',
'related_horizontal_A','related_vertical_A',
'TeamDistance','EnermyTeamDistance',
'TeamXstd','EnermyXstd',
'EnermyYstd','TeamYstd',
'DistanceToBallRank','DistanceToBallRank_AttTeam','DistanceToBallRank_DefTeam',
'YardLine','NextX','NextY',
'NextDistanceToBall',
'BallNextAngleX','BallNextAngleY',
'BallNextDistanceX','BallNextDistanceY','A']
