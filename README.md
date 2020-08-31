
- competition home page
  https://www.kaggle.com/c/nfl-big-data-bowl-2020/data

- No. 1 solution
  https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/119400

- No. 3 solution
  https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/119400

The followings is nothing but a summarization of what is explained in 
https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/119400; no credit is claimed.


# Data processing
  - always be from left to right
  - clip target to -30 and 50
  - replace S with Dis * 10 (time frame different issue, how to find it?)
  - A = (Dis/S)/0.1 (not used in training)
  - no standardization as only relative features being relied upon

# CV
  - 5-fold GroupKFold on GameId, but in validation folds we only consider data from 2018
  - correlation btw/ CV & LB score

# autmentadion & tta
What worked really well for us is to add augmentation and TTA for Y coordinates. 
We assume that in a mirrored world the runs would have had the same outcomes.
- 50% augmentation 
- 50-50 blend of flipped and non-flipped inference

# comments
We have a 10x11 tensor, with 10 offensive players (excluding rusher) and 11 defenders. 
Then you do a 1x1 convolution (order does not matter) and then you pool over the defender dimension leaving you with 1x11 and then you repeat it with a 1D CNN, pool again, and in the end you are left with 1x1. So in rough words you find the strength for all offensive players for each defensive player and then you find the strength of all defensive players with respect to the rusher.

In the first step you pool all offensive players for each defensive players and in the second step you pool over all defensive players.

# Links
- https://www.kaggle.com/philippsinger/nfl-playing-surface-analytics-the-zoo
- Graph Transformer
  https://www.kaggle.com/cpmpml/graph-transfomer
