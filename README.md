# Introduction
This library provides an easy to use backbone for diverse computer vision

# Installation

```
git clone https://github.com/AokiMasataka/open_backbones.git
cd ./open_backbones
pip install .
```

# config file
The config file is written in python, see ./configs/ for a template.

# model config
Basically, it is written as follows

```python
backbone=dict(
    type='ResNet',
    block_type='BasicBlock',
    layers=(2, 2, 2, 2),
    in_channels=3,
    stem_width=64,
    channels=(64, 128, 256, 512),
    act_config=dict(type='ReLU', inplace=True)
)
```

# usage

config file must be populated with dict type
```python
from openbacks import build_backbone

config = dict(
        type='MixVisionTransformer',
        img_size=224,
        embed_dims=(32, 64, 160, 256),
        num_heads=(1, 2, 5, 8),
        mlp_ratios=(4, 4, 4, 4),
        qkv_bias=True,
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        drop_rate=0.0,
        drop_path_rate=0.1,
        init_config=dict(pretrained='https://github.com/qubvel/segmentation_models.pytorch/releases/download/v0.0.2/mit_b0.pth')
    )
backbone = build_backbone(config=config)
```