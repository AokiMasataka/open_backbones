backbone=dict(
    type='ConvNeXt',
    depth=(3, 3, 27, 3),
    dims=(128, 256, 512, 1024),
    out_indices=(0, 1, 2, 3),
    drop_path_rate=0.3,
    init_config=dict(pretrained='https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_384.pth')
)