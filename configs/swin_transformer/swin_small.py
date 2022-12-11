backbone=dict(
    type='SwinTransformer',
    patch_size=4,
    window_size=7,
    embed_dim=96,
    depths=(2, 2, 18, 2),
    num_heads=(3, 6, 12, 24),
    init_config=dict(
        pretrained='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'
    )
)