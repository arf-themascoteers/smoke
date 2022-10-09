import torch.nn as nn


class MyMachine(nn.Module):
    def __init__(self):
        super().__init__()
        self.band_net = nn.Sequential(
            nn.Linear(300,1)
        )

    def forward(self, x):
        x = self.band_net(x)
        return x

