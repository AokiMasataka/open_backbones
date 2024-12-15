import math
import numpy
import torch
from torch import nn, Tensor
from torch.nn import functional


def relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
    relative_buckets = 0

    num_buckets //= 2
    relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
    relative_position = torch.abs(relative_position)
    # now relative_position is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    relative_position_if_large = torch.log(relative_position.float() / max_exact) / math.log(max_distance / max_exact)
    relative_position_if_large = max_exact + (relative_position_if_large * (num_buckets - max_exact)).to(torch.long)
    relative_position_if_large = torch.min(relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1))
    relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
    return relative_buckets


class BaseAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        relative_pos_bias: bool = False,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
    ):
        super(BaseAttention, self).__init__()
        self.relative_pos_bias = relative_pos_bias
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance

        if relative_pos_bias:
            self.relative_attention_bias = nn.Embedding(relative_attention_num_buckets, num_heads)

    def compute_bias(self, query_length, key_length, device=None):
        context_position = torch.arange(query_length, dtype=torch.long, device=device).unsqueeze(1)
        memory_position = torch.arange(key_length, dtype=torch.long, device=device).unsqueeze(0)
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position = relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values


class SelfAttention(BaseAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        relative_pos_bias: bool = True, 
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super(SelfAttention, self).__init__(
            num_heads=num_heads,
            relative_pos_bias=relative_pos_bias,
            relative_attention_num_buckets=relative_attention_num_buckets,
            relative_attention_max_distance=relative_attention_max_distance,
        )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(in_features=embed_dim, out_features=embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)

    def forward(self, hidden_states: Tensor, attn_mask: Tensor = None) -> Tensor:
        B, N, C = hidden_states.shape
        qkv = self.qkv(hidden_states).view(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        x = functional.scaled_dot_product_attention(
            query=q, key=k, value=v, attn_mask=attn_mask, dropout_p=self.dropout, is_causal=False
        )

        if self.relative_pos_bias:
            position_bias = self.compute_bias(N, N, device=x.device)
            x = x + position_bias

        x = x.transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj(x)
        return x


class CrossAttention(BaseAttention):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        cross_dim: int = None,
        qkv_bias: bool = False,
        relative_pos_bias: bool = True, 
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super(CrossAttention, self).__init__(
            num_heads=num_heads,
            relative_pos_bias=relative_pos_bias,
            relative_attention_num_buckets=relative_attention_num_buckets,
            relative_attention_max_distance=relative_attention_max_distance,
        )
        self.embed_dim = embed_dim
        self.cross_dim = embed_dim if cross_dim is None else cross_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        self.q = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=qkv_bias)
        self.kv = nn.Linear(in_features=self.cross_dim, out_features=embed_dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
    
    def forward(self, hidden_states: Tensor, context: Tensor, attn_mask: Tensor = None) -> Tensor:
        B, N, C = hidden_states.shape
        _, context_N, _ = context.shape

        q = self.q(hidden_states).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        kv = self.kv(context).view(B, context_N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        x = functional.scaled_dot_product_attention(
            query=q, key=k, value=v, attn_mask=attn_mask, dropout_p=self.dropout, is_causal=False
        )

        if self.relative_pos_bias:
            position_bias = self.compute_bias(N, N, device=x.device)
            x = x + position_bias

        x = x.transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj(x)
        return x



class FeedForward(nn.Sequential):
    def __init__(self, input_dim: int, output_dim: int = None, ffn_dim: int = None, dropout: float = 0.0, activation_fn: str = 'gelu'):
        acts = {'gelu': nn.GELU, 'relu': nn.ReLU}
        assert activation_fn in acts.keys()
        output_dim = input_dim if output_dim is None else output_dim
        ffn_dim = output_dim if ffn_dim is None else ffn_dim
        self.output_dim = output_dim
        super(FeedForward, self).__init__(
            nn.Linear(in_features=input_dim, out_features=ffn_dim),
            acts[activation_fn](),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=ffn_dim, out_features=output_dim),
        )
