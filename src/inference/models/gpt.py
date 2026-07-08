"""GPT-2 style causal language model.

The attention module here only owns the q/k/v/out projections; the cache and
softmax machinery live in the model-agnostic :class:`inference.attention.Attention`.
"""

import torch
import torch.nn as nn

from inference.attention import Attention
from inference.models.layers import FeedForward, LayerNorm

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "num_heads": 12,
    "num_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


class GptAttention(nn.Module):
    """Dense multi-head attention: q/k/v/out projections + shared attention core."""

    def __init__(self, d_in, d_out, num_heads, dropout=0.0, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.attn = Attention(num_heads, self.head_dim, dropout=dropout)

    def forward(self, x, kv_cache=None, layer_id=None, input_lens=None, cache_batch_indices=None):
        q = self.W_query(x)
        k = self.W_key(x)
        v = self.W_value(x)
        context = self.attn(
            q,
            k,
            v,
            kv_cache=kv_cache,
            layer_id=layer_id,
            input_lens=input_lens,
            cache_batch_indices=cache_batch_indices,
        )
        return self.out_proj(context)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GptAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["num_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, kv_cache=None, layer_id=None, input_lens=None, cache_batch_indices=None):
        shortcut = x
        x = self.norm1(x)
        x = self.att(
            x,
            kv_cache=kv_cache,
            layer_id=layer_id,
            input_lens=input_lens,
            cache_batch_indices=cache_batch_indices,
        )
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GptModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.position_embedding = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_embedding = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["num_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx, kv_cache=None, input_lens=None, cache_batch_indices=None):
        batch_size, seq_len = in_idx.shape

        if kv_cache is None:
            positions = torch.arange(seq_len, device=in_idx.device)
        elif seq_len == 1:
            if cache_batch_indices is not None:
                indices = torch.tensor(cache_batch_indices, device=in_idx.device, dtype=torch.long)
                positions = kv_cache.pos[indices].unsqueeze(1)
            else:
                positions = kv_cache.pos[:batch_size].unsqueeze(1)
        else:
            positions = torch.zeros(batch_size, seq_len, dtype=torch.long, device=in_idx.device)
            if input_lens is None:
                raise ValueError("input_lens is required for batched prefill with kv_cache")
            for i in range(batch_size):
                valid_len = input_lens[i].item()
                if cache_batch_indices is not None:
                    start_pos = kv_cache.pos[cache_batch_indices[i]].item()
                else:
                    start_pos = kv_cache.pos[i].item()
                positions[i, seq_len - valid_len :] = torch.arange(
                    start_pos, start_pos + valid_len, device=in_idx.device
                )

        x = self.token_embedding(in_idx) + self.position_embedding(positions)
        x = self.drop_embedding(x)

        for layer_id, block in enumerate(self.trf_blocks):
            x = block(
                x,
                kv_cache=kv_cache,
                layer_id=layer_id,
                input_lens=input_lens,
                cache_batch_indices=cache_batch_indices,
            )

        x = self.final_norm(x)
        return self.out_head(x)
