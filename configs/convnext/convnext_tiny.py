backbone = dict(
    type='ConvNeXt',
    depth=(3, 3, 9, 3),
    dims=(96, 192, 284, 768),
    out_indices=(0, 1, 2, 3),
    drop_path_rate=0.2,
    init_config=dict(pretrained='https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_384.pth')
)