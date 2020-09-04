import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.utils import weight_norm

class NFLRushNet(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        target_size = 199
        n_channels = 10
        h, w = 11, 10 # (offense, defense)
        lambda_ = 0.7

        self.net1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(n_channels, 128, kernel_size=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(128, 160, kernel_size=1),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(160, 128, kernel_size=1),
                nn.ReLU()
            )
        ])

        # pool over offense
        self.max_pool1 = nn.MaxPool2d((h, 1))
        self.avg_pool1 = nn.AvgPool2d((h, 1))

        self.bn1 = nn.BatchNorm1d(128)

        self.net2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(128, 160, kernel_size=1, bias=False),
                nn.ReLU(),
                nn.BatchNorm1d(160)
            ),
            nn.Sequential(
                nn.Conv1d(160, 96, kernel_size=1, bias=False),
                nn.ReLU(),
                nn.BatchNorm1d(96)
            ),
            nn.Sequential(
                nn.Conv1d(96, 96, kernel_size=1, bias=False),
                nn.ReLU(),
                nn.BatchNorm1d(96)
            )
        ])

        self.max_pool2 = nn.MaxPool1d(w)
        self.avg_pool2 = nn.AvgPool1d(w)


        self.net3 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(96, 96, bias=False),
                nn.ReLU(),
                nn.BatchNorm1d(96)
            ),
            nn.Sequential(
                nn.Linear(96, 256, bias=False),
                nn.ReLU(),
                nn.LayerNorm(256)
            ),
        ])

        self.do = nn.Dropout(dropout)
        self.out = nn.Linear(256, target_size)

        self.apply(self.weights_init)

    def forward(self, inp):

        x = inp
        for i in range(len(self.net1)):
            x = self.net1[i](x)

        x = (0.3*self.max_pool1(x) + 0.7*self.avg_pool1(x)).squeeze()

        x = self.bn1(x)

        for i in range(len(self.net2)):
            x = self.net2[i](x)

        x = (0.3*self.max_pool2(x) + 0.7*self.avg_pool2(x)).squeeze()

        for i in range(len(self.net3)):
            x = self.net3[i](x)

        x = self.do(x)

        x = self.out(x)

        return x
        #return F.softmax(x, dim=-1)

    def wnorm(self):
        self.conv = weight_norm(self.conv, name="weight")
        for i in range(len(self.pre)):
            self.pre[i] = weight_norm(self.pre[i], name="weight")

    @staticmethod
    def init_weight(weight):
        nn.init.uniform_(weight, -0.1, 0.1)

    @staticmethod
    def init_bias(bias):
        nn.init.constant_(bias, 0.0)

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1 or classname.find('Conv2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                NFLRushNet.init_weight(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    NFLRushNet.init_bias(m.bias)

if __name__ == '__main__':
    net = NFLRushNet()
    bsz=5;n_channels=10;h=10;w=11
    print(net)
    x = torch.rand((bsz, n_channels, w , h))
    x = net(x)
    print(x.shape)

