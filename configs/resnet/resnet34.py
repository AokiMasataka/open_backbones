backbone=dict(
    type='ResNet',
    block_type='BasicBlock',
    layers=(2, 2, 2, 2),
    in_channels=3,
    stem_width=64,
    channels=(64, 128, 256, 512),
    act_config=dict(type='ReLU', inplace=True)
)