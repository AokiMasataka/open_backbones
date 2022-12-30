import math
import torch
from torch import nn


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


class SelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        relative_pos_bias: bool = True, 
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        dropout: float = 0.0,
    ):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.relative_pos_bias = relative_pos_bias 
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout = dropout
        self.key_value_proj_dim = embed_dim // num_heads

        self.qkv = nn.Linear(in_features=embed_dim, out_features=embed_dim * 3, bias=False)
        self.out = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)

        self.relative_attention_bias = nn.Embedding(relative_attention_num_buckets, num_heads)
    
    def compute_bias(self, query_length, key_length, device=None):
        context_position = torch.arange(query_length, dtype=torch.long, device=device).unsqueeze(1)
        memory_position = torch.arange(key_length, dtype=torch.long, device=device).unsqueeze(0)
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values
    
    def forward(self, hidden_states):
        batch_size, seq_length, _ = hidden_states.shape

        query_key_value_states = self.kv(hidden_states).view(batch_size, seq_length, 3, self.num_heads, self.key_value_proj_dim)
        query_states, key_states, value_states = query_key_value_states.transpose(1, 3).unbind(2)

        scores = torch.matmul(query_states, key_states.transpose(3, 2))

        if self.relative_pos_bias:
            position_bias = self.compute_bias(seq_length, seq_length, device=scores.device)
            scores = scores + position_bias
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)  # (batch_size, seq_length, dim)
        attn_output = self.out(attn_output)
        return attn_output


class CrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        relative_pos_bias: bool = True, 
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        dropout: float = 0.0,
    ):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.relative_pos_bias = relative_pos_bias
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout = dropout
        self.key_value_proj_dim = embed_dim // num_heads

        self.q = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)
        self.kv = nn.Linear(in_features=embed_dim, out_features=embed_dim * 2, bias=False)
        self.out = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=False)

        self.relative_attention_bias = nn.Embedding(relative_attention_num_buckets, num_heads)
    
    def compute_bias(self, query_length, key_length, device=None):
        context_position = torch.arange(query_length, dtype=torch.long, device=device).unsqueeze(1)
        memory_position = torch.arange(key_length, dtype=torch.long, device=device).unsqueeze(0)
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values
    
    def forward(self, hidden_states, key_value_states):
        batch_size, seq_length, _ = hidden_states.shape
        key_length =  key_value_states.shape[1]

        query_states = self.q(hidden_states).view(batch_size, seq_length, self.num_heads, self.key_value_proj_dim).transpose(1, 2)
        key_value_states = self.kv(key_value_states).view(batch_size, key_length, 2, self.num_heads, self.key_value_proj_dim)
        key_states, value_states = key_value_states.transpose(1, 3).unbind(2)

        scores = torch.matmul(query_states, key_states.transpose(3, 2))

        if self.relative_pos_bias:
            position_bias = self.compute_bias(seq_length, key_length, device=scores.device)
            scores = scores + position_bias
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)  # (batch_size, seq_length, dim)
        attn_output = self.out(attn_output)
        return attn_output



class FeedForward(nn.Sequential):
    def __init__(self, input_dim: int, output_dim: int = None, ffn_dim: int = None, dropout: float = 0.0, act: str = 'gelu'):
        acts = {'gelu': nn.GELU, 'relu': nn.ReLU}
        assert act in acts.keys()
        output_dim = input_dim if output_dim is None else output_dim
        ffn_dim = output_dim if ffn_dim is None else ffn_dim
        self.output_dim = output_dim
        super(FeedForward, self).__init__(
            nn.Linear(in_features=input_dim, out_features=ffn_dim),
            acts[act](),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=ffn_dim, out_features=output_dim),
        )