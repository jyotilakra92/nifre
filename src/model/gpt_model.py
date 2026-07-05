import torch
import torch.nn as nn

from layer_norm import LayerNorm
from transformer_block import TransformerBlock

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "num_heads": 12,
    "num_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}


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
                positions[i, seq_len - valid_len :] = torch.arange(
                    valid_len, device=in_idx.device
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
