backbone=dict(
    type='ResNet',
    block_type='Bottleneck',
    layers=(3, 8, 36, 3),
    channels=(256, 512, 1024, 2048),
    in_channels=3,
    stem_width=64,
    deep_stem=True,
    act_config=dict(type='ReLU', inplace=True),
    init_config=dict(
        pretrained='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet200d_ra2-bdba9bf9.pth',
    ),
	norm_config=dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), div=255.0)
)