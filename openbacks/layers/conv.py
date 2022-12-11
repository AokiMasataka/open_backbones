from torch import nn
from ..utils import build_activation, build_norm


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, act_config=None, norm_config=None):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            bias=bias
        )
        
        if act_config is not None:
            self.act = build_activation(config=act_config)
        else:
            self.act = False
        
        if norm_config is not None:
            self.norm = build_norm(config=norm_config)
        else:
            self.act = False
    
    def forward(self, input):
        input = self.conv(input)
        
        if self.act:
            input = self.act(input)
        
        if self.norm:
            input = self.norm(input)
        return input
