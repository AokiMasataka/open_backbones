backbone=dict(
    type='ConvNeXt',
    depths=(3, 3, 27, 3),
    dims=(256, 512, 1024, 2048),
    out_indices=(0, 1, 2, 3),
    drop_path_rate=0.3,
    init_config=dict(pretrained='https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_384_ema.pth'),
	norm_config=dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), div=255.0)
)