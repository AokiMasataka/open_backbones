from torch import nn, Tensor


class SEModule(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super(SEModule, self).__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x: Tensor) -> Tensor:
        return x * self.cSE(x) + x * self.sSE(x)
        