"""Model-agnostic attention over a KV cache.

This layer holds no projection weights. Callers pass already-projected
``q``/``k``/``v`` tensors of shape ``(batch, seq, num_heads * head_dim)`` and it
computes causal attention, reading/writing the KV cache when one is supplied.

Splitting the cache + softmax machinery out of any specific model lets every
backend reuse the same paged/dense KV cache path — the projections (and any
model-specific step like RoPE) stay in the model definition.
"""

import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, layer_id=None, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_out = num_heads * head_dim
        self.layer_id = layer_id
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q,
        k,
        v,
        kv_cache=None,
        layer_id=None,
        input_lens=None,
        cache_batch_indices=None,
    ):
        """Causal attention over projected q/k/v.

        Training (no cache): full causal attention over all tokens.
        Inference (with cache): appends per-row K/V into ``kv_cache`` and attends
        over the accumulated context. For left-padded prefill, pass ``input_lens``
        so only non-pad tokens are used.
        """
        resolved_layer_id = layer_id if layer_id is not None else self.layer_id
        if kv_cache is None:
            return self._forward_full_causal(q, k, v)
        if resolved_layer_id is None:
            raise ValueError("layer_id is required when kv_cache is set")
        return self._forward_with_cache(
            q, k, v, kv_cache, resolved_layer_id, input_lens, cache_batch_indices
        )

    def _shape_heads(self, tensor):
        b, num_tokens, _ = tensor.shape
        return tensor.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, context, b, num_tokens):
        context = context.transpose(1, 2).contiguous()
        return context.view(b, num_tokens, self.d_out)

    def _forward_full_causal(self, q, k, v):
        b, num_tokens, _ = q.shape
        queries = self._shape_heads(q)
        keys = self._shape_heads(k)
        values = self._shape_heads(v)

        attn_scores = queries @ keys.transpose(2, 3)
        mask = torch.triu(
            torch.ones(num_tokens, num_tokens, device=q.device, dtype=torch.bool),
            diagonal=1,
        )
        attn_scores.masked_fill_(mask, -torch.inf)

        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = attn_weights @ values
        return self._merge_heads(context, b, num_tokens)

    def _forward_with_cache(self, q, k, v, kv_cache, layer_id, input_lens, cache_batch_indices):
        batch_size, num_new_tokens, _ = q.shape
        out = torch.zeros_like(q)

        for i in range(batch_size):
            cache_slot = cache_batch_indices[i] if cache_batch_indices is not None else i
            if input_lens is not None and num_new_tokens > 1:
                valid_len = input_lens[i].item()
                out_row = self._forward_with_cache_row(
                    q[i : i + 1, -valid_len:],
                    k[i : i + 1, -valid_len:],
                    v[i : i + 1, -valid_len:],
                    kv_cache,
                    cache_slot,
                    layer_id,
                )
                out[i, -valid_len:] = out_row.squeeze(0)
            else:
                out[i : i + 1] = self._forward_with_cache_row(
                    q[i : i + 1],
                    k[i : i + 1],
                    v[i : i + 1],
                    kv_cache,
                    cache_slot,
                    layer_id,
                )

        return out

    def _forward_with_cache_row(self, q, k, v, kv_cache, batch_idx, layer_id):
        b, num_new_tokens, _ = q.shape
        assert b == 1

        past_keys, past_values = kv_cache.get(batch_idx, layer_id)
        past_len = past_keys.shape[0]

        queries = self._shape_heads(q)
        new_keys = self._shape_heads(k)
        new_values = self._shape_heads(v)

        keys_to_store = new_keys.squeeze(0).transpose(0, 1).contiguous()
        values_to_store = new_values.squeeze(0).transpose(0, 1).contiguous()
        kv_cache.append(batch_idx, layer_id, keys_to_store, values_to_store)

        if past_len > 0:
            past_keys = past_keys.to(dtype=queries.dtype, device=queries.device)
            past_values = past_values.to(dtype=queries.dtype, device=queries.device)
            past_keys = past_keys.transpose(0, 1).unsqueeze(0)
            past_values = past_values.transpose(0, 1).unsqueeze(0)
            keys = torch.cat([past_keys, new_keys], dim=2)
            values = torch.cat([past_values, new_values], dim=2)
        else:
            keys = new_keys
            values = new_values

        total_keys = keys.shape[2]
        attn_scores = queries @ keys.transpose(2, 3)

        if num_new_tokens > 1:
            mask = self._causal_mask(num_new_tokens, total_keys, past_len, attn_scores.device)
            attn_scores.masked_fill_(mask, -torch.inf)

        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = attn_weights @ values
        return self._merge_heads(context, b, num_new_tokens)

    @staticmethod
    def _causal_mask(num_queries, total_keys, past_len, device):
        """Mask future keys for prefill chunks.

        Query i (at absolute position past_len + i) may attend to keys 0..past_len + i.
        """
        key_idx = torch.arange(total_keys, device=device)
        query_abs_pos = torch.arange(num_queries, device=device) + past_len
        return key_idx.unsqueeze(0) > query_abs_pos.unsqueeze(1)
