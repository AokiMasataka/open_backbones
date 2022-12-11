import re
import torch
from torch import nn
from ..layers import Mlp, DropPath, LayerNorm2d
from ..utils import BaseModule
from ..builder import BACKBONES


class ConvNeXtBlock(nn.Module):
    """ ConvNeXt Block
    There are two equivalent implementations:
      (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
      (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    Unlike the official impl, this one allows choice of 1 or 2, 1x1 conv can be faster with appropriate
    choice of LayerNorm impl, however as model size increases the tradeoffs appear to change and nn.Linear
    is a better choice. This was observed with PyTorch 1.10 on 3090 GPU, it could change over time & w/ different HW.
    Args:
        in_chs (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(
            self,
            in_chs,
            out_chs=None,
            kernel_size=7,
            stride=1,
            dilation=1,
            mlp_ratio=4,
            conv_bias=True,
            ls_init_value=1e-6,
            act_config=None,
            drop_path=0.,
    ):
        super(ConvNeXtBlock, self).__init__()
        out_chs = out_chs or in_chs
        if act_config is None:
            act_config = dict(type='GELU')
        
        if dilation == 2:
            padding = 6
        else:
            padding = 3

        self.conv_dw = nn.Conv2d(
            in_channels=in_chs,
            out_channels=out_chs,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            bias=conv_bias
        )
        self.norm = nn.LayerNorm(out_chs)
        self.mlp = Mlp(out_chs, int(mlp_ratio * out_chs), act_config=act_config)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(out_chs)) if ls_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2)

        if self.gamma is not None:
            x = x.mul(self.gamma.view(1, -1, 1, 1))

        x = self.drop_path(x) + shortcut
        return x


class ConvNeXtStage(nn.Module):
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size=7,
            stride=2,
            depth=2,
            dilation=(1, 1),
            drop_path_rates=None,
            ls_init_value=1.0,
            conv_bias=True,
            act_config=None,
    ):
        super(ConvNeXtStage, self).__init__()
        self.grad_checkpointing = False

        if in_chs != out_chs or stride > 1 or dilation[0] != dilation[1]:
            downsamlpe_kernelsize = 2 if stride > 1 or dilation[0] != dilation[1] else 1
            pad = 'same' if dilation[1] > 1 else 0  # same padding needed if dilation used
            self.downsample = nn.Sequential(
                LayerNorm2d(in_chs),
                nn.Conv2d(
                    in_channels=in_chs,
                    out_channels=out_chs,
                    kernel_size=(2, 2),
                    stride=(2, 2),
                    dilation=dilation[0],
                    padding=0,
                    bias=conv_bias
                ),
            )
            in_chs = out_chs
        else:
            self.downsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.] * depth
        stage_blocks = []
        for i in range(depth):
            stage_blocks.append(ConvNeXtBlock(
                in_chs=in_chs,
                out_chs=out_chs,
                kernel_size=kernel_size,
                dilation=dilation[1],
                drop_path=drop_path_rates[i],
                ls_init_value=ls_init_value,
                conv_bias=conv_bias,
                act_config=act_config
            ))
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


@BACKBONES.register_module
class ConvNeXt(BaseModule):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  - https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (tuple(int)): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_rate (float): Head dropout rate
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
            self,
            in_chans=3,
            output_stride=32,
            depths=(3, 3, 9, 3),
            dims=(96, 192, 384, 768),
            out_indices=(0, 1, 2, 3),
            kernel_sizes=7,
            ls_init_value=1e-6,
            patch_size=4,
            conv_bias=True,
            drop_rate=0.0,
            drop_path_rate=0.,
            act_config=None,
            init_config=None,
    ):
        super(ConvNeXt, self).__init__(init_confg=init_config)
        assert output_stride in (8, 16, 32)

        self.out_indices = out_indices
        self.drop_rate = drop_rate
        self.feature_info = []

        # NOTE: this stem is a minimal form of ViT PatchEmbed, as used in SwinTransformer w/ patch_size = 4
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=patch_size, stride=patch_size, bias=conv_bias),
            LayerNorm2d(dims[0])
        )
        stem_stride = patch_size

        self.stages = nn.Sequential()
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages = []
        prev_chs = dims[0]
        curr_stride = stem_stride
        dilation = 1
        # 4 feature resolution stages, each consisting of multiple residual blocks
        for i in range(4):
            stride = 2 if curr_stride == 2 or i > 0 else 1
            if curr_stride >= output_stride and stride > 1:
                dilation *= stride
                stride = 1

            curr_stride *= stride
            first_dilation = 1 if dilation in (1, 2) else 2
            out_chs = dims[i]
            stages.append(ConvNeXtStage(
                prev_chs,
                out_chs,
                kernel_size=kernel_sizes,
                stride=stride,
                dilation=(first_dilation, dilation),
                depth=depths[i],
                drop_path_rates=dp_rates[i],
                ls_init_value=ls_init_value,
                conv_bias=conv_bias,
                act_config=act_config,
            ))
            prev_chs = out_chs
            # NOTE feature_info use currently assumes stage 0 == stride 1, rest are stride 2
            self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, module=f'stages.{i}')]
        self.stages = nn.Sequential(*stages)
        self.num_features = prev_chs

        self.init(checkpoint_filter_fn=checkpoint_filter_fn)
    
    def forward(self, x):
        x = self.stem(x)

        feats = list()
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index in self.out_indices:
                feats.append(x)
        return feats


def checkpoint_filter_fn(key, value, state_dict):
    key = key.replace('downsample_layers.0.', 'stem.')
    key = re.sub(r'stages.([0-9]+).([0-9]+)', r'stages.\1.blocks.\2', key)
    key = re.sub(r'downsample_layers.([0-9]+).([0-9]+)', r'stages.\1.downsample.\2', key)
    key = key.replace('dwconv', 'conv_dw')
    key = key.replace('pwconv', 'mlp.fc')

    if value.ndim == 2 and 'head' not in key:
        model_shape = state_dict()[key].shape
        value = value.reshape(model_shape)
    return key, value