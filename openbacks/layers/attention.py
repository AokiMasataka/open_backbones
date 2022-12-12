import torch
from torch import nn
from torch.nn import functional


class AttentionBase(nn.Module):
    def __init__(self):
        super(AttentionBase, self).__init__()
        pass
    
    def _attention(self, query, key, value):
        if self.upcast_attention:
            query = query.float()
            key = key.float()
        
        B, N, C = query.shape
        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )

        attention_probs = attention_scores.softmax(dim=2).to(value.dtype)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = hidden_states.view(B // self.num_heads, self.num_heads, N, C)
        hidden_states = hidden_states.permute(0, 2, 1, 3).reshape(B // self.num_heads, N, C * self.num_heads)
        return hidden_states


class SelfAttention(AttentionBase):
    def __init__(self, dim_query, num_heads=8, dropout=0.0, bias=False, upcast_attention=False):
        super(SelfAttention, self).__init__()
        self.dim_query = dim_query
        self.dim_head = dim_query // num_heads
        self.num_heads = num_heads
        self.scale = self.dim_head ** -0.5
        self.upcast_attention = upcast_attention
        
        self.qkv = nn.Linear(dim_query, dim_query * 3, bias=bias)

        self.out_proj = nn.Linear(dim_query, dim_query)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden_states):
        batch, query_n, query_dim = hidden_states.shape

        query = self.qkv(hidden_states).view(batch, query_n, 3, self.num_heads, self.dim_head)
        query = query.permute(2, 0, 3, 1, 4).reshape(3, batch * self.num_heads, query_n, self.dim_head)
        query, key, value = query.unbind(0)

        hidden_states = self._attention(query=query, key=key, value=value)
        return self.dropout(self.out_proj(hidden_states))


class CrossAttention(AttentionBase):
    def __init__(self, dim_query, dim_cross=None, num_heads=8, dropout=0.0, bias=False, upcast_attention=False):
        super(CrossAttention, self).__init__()
        self.dim_query = dim_query
        self.dim_head = dim_query // num_heads
        self.num_heads = num_heads
        self.scale = self.dim_head ** -0.5
        self.upcast_attention = upcast_attention
        
        self.q = nn.Linear(dim_query, dim_query, bias=bias)
        self.kv = nn.Linear(dim_cross, dim_query * 2, bias=bias)

        self.out_proj = nn.Linear(dim_query, dim_query)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, hidden_states, context=None):
        context = hidden_states if context is None else context
        batch, query_n, query_dim = hidden_states.shape
        _, context_n, context_dim = context.shape

        query = self.q(hidden_states).view(batch, query_n, self.num_heads, self.dim_head)
        query = query.permute(0, 2, 1, 3).reshape(batch * self.num_heads, query_n, self.dim_head)

        kv = self.kv(context).view(batch, context_n, 2, self.num_heads, self.dim_head)
        kv = kv.permute(2, 0, 3, 1, 4).reshape(2, batch * self.num_heads, context_n, self.dim_head)
        key, value = kv.unbind(0)
        
        hidden_states = self._attention(query=query, key=key, value=value)
        return self.dropout(self.out_proj(hidden_states))


class SelfAttention2D(SelfAttention):
    def __init__(self, dim_query, dim_inner, num_heads=8, dropout=0.0, bias=False, upcast_attention=False):
        super(SelfAttention2D, self).__init__(
            dim_query=dim_query,
            dim_inner=dim_inner,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            upcast_attention=upcast_attention
        )
    
    def forward(self, hidden_states):
        batch, query_dim, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch, height * width, query_dim).transpose(1, 2)
        query_n = height * width

        query = self.qkv(hidden_states).view(batch, query_n, 3, self.num_heads, self.dim_head)
        query = query.permute(2, 0, 3, 1, 4).reshape(3, batch * self.num_heads, query_n, self.dim_head)
        query, key, value = query.unbind(0)

        hidden_states = self._attention(query=query, key=key, value=value)
        hidden_states = self.dropout(self.out_proj(hidden_states))
        return hidden_states.transpose(1, 2).view(batch, query_dim, height, width)


class CrossAttention2D(CrossAttention):
    def __init__(self, dim_query, dim_inner, dim_cross=None, num_heads=8, dropout=0.0, bias=False, upcast_attention=False):
        super(CrossAttention2D, self).__init__(
            dim_query=dim_query,
            dim_inner=dim_inner,
            dim_cross=dim_cross,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            upcast_attention=upcast_attention
        )
    
    def forward(self, hidden_states, context=None):
        context = hidden_states if context is None else context
        batch, query_dim, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch, height * width, query_dim).transpose(1, 2)
        query_n = height * width

        _, context_n, context_dim = context.shape

        query = self.q(hidden_states).view(batch, query_n, self.num_heads, self.dim_head)
        query = query.permute(0, 2, 1, 3).reshape(batch * self.num_heads, self.dim_head)

        kv = self.kv(context).view(batch, context_n, 2, self.num_heads, self.dim_head)
        kv = kv.permute(2, 0, 3, 1, 4).reshape(2, batch * self.num_heads, query_n, self.dim_head)
        key, value = kv.unbind(0)
        
        hidden_states = self._attention(query=query, key=key, value=value)
        hidden_states = self.dropout(self.out_proj(hidden_states))
        return hidden_states.transpose(1, 2).view(batch, query_dim, height, width)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = 'geglu',
    ):
        super(FeedForward, self).__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        act_fns = {'gelu': GELU, 'geglu': GEGLU}

        self.body = nn.Sequential(
            act_fns[activation_fn](dim, inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, hidden_states):
        return self.body(hidden_states)


class GELU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, hidden_states):
        return functional.gelu(self.proj(hidden_states))


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, hidden_states):
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * functional.gelu(gate)
