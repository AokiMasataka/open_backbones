import torch
from openbacks.layers.attention import SelfAttention, CrossAttention



def test_selfattn():
    x = torch.rand(2, 32, 64)
    self_attn = SelfAttention(
        embed_dim=64, num_heads=2, qkv_bias=False, relative_pos_bias=False,
    )

    with torch.no_grad():
        y = self_attn(x)
    
    print(x.shape == y.shape)

    self_attn = SelfAttention(
        embed_dim=64, num_heads=2, qkv_bias=False, relative_pos_bias=True,
    )

    with torch.no_grad():
        y = self_attn(x)
    
    print(x.shape == y.shape)


def test_crossattn():
    x = torch.rand(2, 32, 64)
    
    cross_attn = CrossAttention(
        embed_dim=64, num_heads=2, qkv_bias=False, relative_pos_bias=False,
    )

    with torch.no_grad():
        y = cross_attn(x, context=x)
    
    print(x.shape == y.shape)

    context = torch.rand(2, 128, 32)
    cross_attn = CrossAttention(
        embed_dim=64, cross_dim=32, num_heads=2, qkv_bias=False, relative_pos_bias=True,
    )

    with torch.no_grad():
        y = cross_attn(x, context=context)
    
    print(x.shape == y.shape)


if __name__ == '__main__':
    test_selfattn()
    test_crossattn()
