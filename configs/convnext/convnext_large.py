backbone=dict(
    type='ConvNeXt',
    depth=(3, 3, 27, 3),
    dims=(192, 384, 768, 1536),
    out_indices=(0, 1, 2, 3),
    drop_path_rate=0.3,
    init_config=dict(pretrained='ttps://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_384.pth')
)