import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.utils import weight_norm

class NFLNet(nn.Module):
    def __init__(self, n_channels=10, h=11, w=10, lam=.6, wnorm=True):
        super().__init__()

        self.bn0 = nn.BatchNorm2d(n_channels)

        self.interaction_net = nn.ModuleList([
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

        self.max_pool = nn.MaxPool2d((h, w))
        self.avg_pool = nn.AvgPool2d((h, w))

        self.lin1 = nn.Linear(256, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.lin2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.out = nn.Linear(256, 199)

        #self.conv = nn.Conv2d(n_channels, 1, kernel_size=1)
        #if wnorm:
        #    self.wnorm()
        self.apply(self.weights_init)

    def forward(self, inp):

        x = self.bn0(inp)
        for i in range(len(self.interaction_net)):
            x = self.interaction_net[i](x)

        x = torch.cat([self.max_pool(x).squeeze(), self.avg_pool(x).squeeze()], dim=-1)

        x = F.relu(self.bn1(self.lin1(x)))
        x = F.relu(self.bn2(self.lin2(x)))
        x = self.out(x)
        x = F.softmax(x)

        return x
    #x = self.conv(x)
    #x0 = self.max_pool(x)
    #x1 = self.avg_pool(x)

        #return (x0 * self.lam + x1 * (1.-self.lam)).squeeze()

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
                NFLNet.init_weight(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    NFLNet.init_bias(m.bias)
