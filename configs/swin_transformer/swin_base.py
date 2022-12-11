backbone=dict(
    type='SwinTransformer',
    patch_size=4,
    window_size=12,
    embed_dim=128,
    depths=(2, 2, 18, 2),
    num_heads=(4, 8, 16, 32),
    init_config=dict(
        pretrained='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth'
    )
)