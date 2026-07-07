import torch

from inference.data_model import ModelConfig
from inference.kv_cache import KVCache


def batch_token_ids(token_lists, device, pad_id):
    """Left-pad token lists to a batch tensor and return per-row real lengths."""
    max_len = max(len(tokens) for tokens in token_lists)
    batch = []
    lengths = []
    for tokens in token_lists:
        lengths.append(len(tokens))
        padding = [pad_id] * (max_len - len(tokens))
        batch.append(padding + tokens)
    token_ids = torch.tensor(batch, device=device)
    input_lens = torch.tensor(lengths, device=device, dtype=torch.long)
    return token_ids, input_lens


def make_kv_cache(model_config: ModelConfig, device, dtype=None):
    if dtype is None:
        dtype = torch.float16
    return KVCache(
        num_layers=model_config.num_layers,
        max_seq_len=model_config.max_seq_len,
        n_heads=model_config.n_heads,
        head_dim=model_config.head_dim,
        device=device,
        dtype=dtype,
    )
