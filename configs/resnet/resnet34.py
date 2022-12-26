backbone=dict(
    type='ResNet',
    block_type='BasicBlock',
    layers=(2, 2, 2, 2),
    channels=(64, 128, 256, 512),
    in_channels=3,
    stem_width=64,
    act_config=dict(type='ReLU', inplace=True)
)