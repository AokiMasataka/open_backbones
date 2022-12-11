backbone=dict(
    type='ResNet',
    block_type='Bottleneck',
    layers=(3, 8, 36, 3),
    in_channels=3,
    stem_width=64,
    deep_stem=True,
    channels=(256, 512, 1024, 2048),
    act_config=dict(type='ReLU', inplace=True),
    init_config=dict(
        pretrained='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet200d_ra2-bdba9bf9.pth',
    )
)