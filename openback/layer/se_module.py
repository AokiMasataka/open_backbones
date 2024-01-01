from torch import nn, Tensor
from .act_norm import build_activation


class SEModule(nn.Module):
    def __init__(
        self,
        channels: int,
        rd_ratio: float = 1/16,
        rd_channels: int = None,
        rd_divisor: int = 8,
        bias: bool = True,
        act_config: dict = None
    ) -> None:
        super(SEModule, self).__init__()
        if rd_channels is None:
            assert channels % 8 == 0
            rd_channels = max(int(channels * rd_ratio), rd_divisor)
        if act_config is None:
            act_config = dict(type='ReLU', inplace=True)

        self.channels = channels

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Conv2d(channels, rd_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=bias)
        self.act = build_activation(config=act_config)
        self.fc2 = nn.Conv2d(rd_channels, channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=bias)
        self.gate = nn.Sigmoid()
    
    def forward(self, x: Tensor) -> Tensor:
        x_se = self.pool(x)

        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)
        