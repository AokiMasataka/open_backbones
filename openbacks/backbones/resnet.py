from torch import nn
from ..layers import SEModule
from ..utils import BaseModule, build_activation
from ..builder import BACKBONES


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, act_config=None, eps=1e-5, hidden_dim_ratio=None, avg_down=False):
        super(BasicBlock, self).__init__()
        if act_config is None:
            act_config = dict(type='ReLU', inaplce=True)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 3), stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=in_channels, eps=eps)
        self.act1 = build_activation(config=act_config)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels, eps=eps)
        self.act2 = build_activation(config=act_config)

        self.se = SEModule(channels=out_channels)
        self.downsample = False

        if stride == 2:
            if avg_down:
                self.downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True, count_include_pad=False),
                    nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
                    nn.BatchNorm2d(num_features=out_channels, eps=eps)
                )
            else:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0), bias=False),
                    nn.BatchNorm2d(num_features=out_channels, eps=eps)
                )
    
    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.se(x)

        if self.downsample:
            shortcut = self.downsample(shortcut)

        x += shortcut
        x = self.act2(x)
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, act_config=None, eps=1e-5, hidden_dim_ratio=1, avg_down=False):
        super(Bottleneck, self).__init__()
        if act_config is None:
            act_config = dict(type='ReLU', inaplce=True)
        hidden_dim = in_channels // hidden_dim_ratio

        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=hidden_dim, eps=eps)
        self.act1 = build_activation(config=act_config)

        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(3, 3), stride=(stride, stride), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=hidden_dim, eps=eps)
        self.act2 = build_activation(config=act_config)

        self.conv3 = nn.Conv2d(hidden_dim, out_channels, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels, eps=eps)
        self.act3 = build_activation(config=act_config)

        self.se = SEModule(channels=out_channels)

        self.downsample = False

        if stride == 2:
            if avg_down:
                self.downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), ceil_mode=True, count_include_pad=False),
                    nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
                    nn.BatchNorm2d(num_features=out_channels, eps=eps)
                )
            else:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0), bias=False),
                    nn.BatchNorm2d(num_features=out_channels, eps=eps)
                )
    
    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = self.se(x)

        if self.downsample:
            shortcut = self.downsample(shortcut)
        
        x += shortcut
        x = self.act3(x)
        return x


@BACKBONES.register_module
class ResNet(BaseModule):
    def __init__(
        self,
        block_type,
        layers,
        in_channels=3,
        stem_width=64,
        deep_stem=False,
        channels=(64, 128, 256, 512),
        avg_down=False,
        out_indices=(0, 1, 2, 3, 4),
        init_config=None,
        act_config=None,
        norm_config=None,
    ):
        super(ResNet, self).__init__(init_confg=init_config)
        assert isinstance(layers, (list, tuple))

        block_dict = {'BasicBlock': BasicBlock, 'Bottleneck': Bottleneck}
        if act_config is None:
            act_config = dict(type='ReLU', inaplce=True)
        
        if norm_config is None:
            norm_config = dict(type='BatchNorm2d')
        
        self.block_type = block_type
        self.in_channels = in_channels
        self.channels = channels
        self.stem_width = stem_width
        self.out_indices = out_indices
        self.act_config = act_config
        self.nrom_config = norm_config

        if deep_stem:
            self.conv1 = nn.Sequential(*[
                nn.Conv2d(in_channels, stem_width, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(stem_width),
                build_activation(config=act_config),
                nn.Conv2d(stem_width, stem_width, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(stem_width),
                build_activation(config=act_config),
                nn.Conv2d(stem_width, stem_width, 3, stride=1, padding=1, bias=False)
            ])
        else:
            self.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=stem_width,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=3,
                bias=False
            )

        self.bn1 = nn.BatchNorm2d(num_features=stem_width)
        self.act1 = build_activation(config=act_config)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        current_channle = stem_width
        for layer, (num_layer, channel) in enumerate(zip(layers, channels)):
            block = list()
            for block_index in range(num_layer):
                if block_index == 0:
                    stride = 2
                    hidden_dim_ratio = 2
                else:
                    current_channle = channel
                    stride = 1
                    hidden_dim_ratio = 4
                
                if block_index == 0 and layer == 0:
                    hidden_dim_ratio = 1

                block.append(block_dict[block_type](
                    in_channels=current_channle,
                    out_channels=channel,
                    stride=stride,
                    act_config=act_config,
                    eps=1e-5,
                    hidden_dim_ratio=hidden_dim_ratio,
                    avg_down=avg_down
                ))
            
            self.add_module(f'layer{layer + 1}', nn.Sequential(*block))
        
        self.blocks = (self.layer1, self.layer2, self.layer3, self.layer4)
        self.init()
    
    def forward(self, x):
        x = self.stem(x)
        
        feats = [x] if 0 in self.out_indices else list()

        for index, block in enumerate(self.blocks, 1):
            x = block(x)
            if index in self.out_indices:
                feats.append(x)
        return feats

    def stem(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.max_pool(x)
        return x
