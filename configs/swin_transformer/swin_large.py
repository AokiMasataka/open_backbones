backbone=dict(
    type='SwinTransformer',
    patch_size=4,
    window_size=12,
    embed_dim=192,
    depths=(2, 2, 18, 2),
    num_heads=(6, 12, 24, 48),
    init_config=dict(
        ptrtrained='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'
    )
)