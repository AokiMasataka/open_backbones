backbone=dict(
    type='MixVisionTransformer',
    embed_dims=(32, 64, 160, 256),
    num_heads=(1, 2, 5, 8),
    mlp_ratios=(4, 4, 4, 4),
    qkv_bias=True,
    depths=[2, 2, 2, 2],
    sr_ratios=[8, 4, 2, 1],
    drop_rate=0.0,
    drop_path_rate=0.1,
    init_config=dict(pretrained='https://github.com/qubvel/segmentation_models.pytorch/releases/download/v0.0.2/mit_b0.pth')
),