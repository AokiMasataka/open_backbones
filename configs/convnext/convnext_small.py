backbone = dict(
    type='ConvNeXt',
    depths=(3, 3, 27, 3),
    dims=(96, 192, 384, 768),
    out_indices=(0, 1, 2, 3),
    drop_path_rate=0.25,
    init_config=dict(pretrained='https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_1k_384.pth'),
	norm_config=dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), div=255.0)
)