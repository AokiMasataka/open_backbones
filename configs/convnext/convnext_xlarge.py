backbone=dict(
    type='ConvNeXt',
    depth=(3, 3, 27, 3),
    dims=(256, 512, 1024, 2048),
    out_indices=(0, 1, 2, 3),
    drop_path_rate=0.3,
    init_config=dict(pretrained='https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_384_ema.pth')
)