from torch import nn
from ..utils import make_divisible, build_activation


class SEModule(nn.Module):
    def __init__(self, channels, rd_ratio=1/16, rd_channels=None, rd_divisor=8, bias=True, act_config=None):
        super(SEModule, self).__init__()
        if not rd_channels:
            rd_channels = make_divisible(v=channels * rd_ratio, min_value=rd_divisor, round_limit=0.)
        if act_config is None:
            act_config = dict(type='ReLU', inplace=True)

        self.channels = channels

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Conv2d(channels, rd_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=bias)
        self.act = build_activation(config=act_config)
        self.fc2 = nn.Conv2d(rd_channels, channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=bias)
        self.gate = nn.Sigmoid()
    
    def forward(self, x):
        x_se = self.pool(x)

        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)
        