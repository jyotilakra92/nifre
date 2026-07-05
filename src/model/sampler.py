import torch


def sample_greedy(logits):
    """logits: (batch, vocab_size) -> (batch, 1)"""
    return torch.argmax(logits, dim=-1, keepdim=True)
