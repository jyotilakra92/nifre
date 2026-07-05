import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.context_length = context_length
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x, kv_cache=None, layer_id=None, input_lens=None, cache_batch_indices=None):
        """Run multi-head attention, optionally reading/writing a KV cache.

        Training (no cache): pass only ``x``. Full causal attention over all tokens.

        Inference (with cache): also pass ``kv_cache`` and ``layer_id``.
        Processes every batch row, appending K/V per row into the cache. For left-padded
        prefill, pass ``input_lens`` (shape ``(batch,)``) so only non-pad tokens are used.
        """
        if kv_cache is None:
            return self._forward_training(x)
        if layer_id is None:
            raise ValueError("layer_id is required when kv_cache is set")
        return self._forward_with_cache(x, kv_cache, layer_id, input_lens, cache_batch_indices)

    def _forward_training(self, x):
        b, num_tokens, _ = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)

    def _forward_with_cache(self, x, kv_cache, layer_id, input_lens=None, cache_batch_indices=None):
        batch_size, num_new_tokens, _ = x.shape
        out = torch.zeros_like(x)

        for i in range(batch_size):
            cache_slot = cache_batch_indices[i] if cache_batch_indices is not None else i
            if input_lens is not None and num_new_tokens > 1:
                valid_len = input_lens[i].item()
                x_row = x[i : i + 1, -valid_len:]
                out_row = self._forward_with_cache_row(
                    x_row, kv_cache, cache_slot, layer_id
                )
                out[i, -valid_len:] = out_row.squeeze(0)
            else:
                x_row = x[i : i + 1]
                out[i : i + 1] = self._forward_with_cache_row(
                    x_row, kv_cache, cache_slot, layer_id
                )

        return out

    def _forward_with_cache_row(self, x, kv_cache, batch_idx, layer_id):
        b, num_new_tokens, _ = x.shape
        assert b == 1

        past_keys, past_values = kv_cache.get(batch_idx, layer_id)
        past_len = past_keys.shape[0]

        queries = self.W_query(x)
        new_keys = self.W_key(x)
        new_values = self.W_value(x)

        queries = queries.view(b, num_new_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        new_keys = new_keys.view(b, num_new_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        new_values = new_values.view(b, num_new_tokens, self.num_heads, self.head_dim).transpose(1, 2)

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

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_new_tokens, self.d_out)
        return self.out_proj(context_vec)

    @staticmethod
    def _causal_mask(num_queries, total_keys, past_len, device):
        """Mask future keys for prefill chunks.

        Query i (at absolute position past_len + i) may attend to keys 0..past_len + i.
        """
        key_idx = torch.arange(total_keys, device=device)
        query_abs_pos = torch.arange(num_queries, device=device) + past_len
        return key_idx.unsqueeze(0) > query_abs_pos.unsqueeze(1)
