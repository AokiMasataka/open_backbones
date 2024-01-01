backbone=dict(
    type='ResNet',
    block_type='BasicBlock',
    layers=(2, 2, 2, 2),
    channels=(64, 128, 256, 512),
    in_channels=3,
    stem_width=64,
    act_config=dict(type='ReLU', inplace=True),
	norm_config=dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), div=255.0)
)