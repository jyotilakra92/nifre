"""KV cache backed by Hugging Face ``past_key_values`` tuples."""

from __future__ import annotations

from typing import Any

import torch


def past_key_values_length(past_key_values: Any | None) -> int:
    if past_key_values is None:
        return 0
    get_seq_length = getattr(past_key_values, "get_seq_length", None)
    if get_seq_length is not None:
        return int(get_seq_length())
    return past_key_values[0][0].shape[-2]


class HFKVCache:
    """Per-slot storage for HF ``past_key_values`` during continuous batching."""

    def __init__(self, max_seq_len: int, device: torch.device | str = "cpu") -> None:
        self.max_seq_len = max_seq_len
        self.device = device
        self.batch_size = 0
        self.pos: torch.Tensor | None = None
        self._past: list[Any | None] = []

    def init_batch(self, batch_size: int) -> None:
        self.batch_size = batch_size
        self.pos = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self._past = [None] * batch_size

    def get_past(self, batch_idx: int) -> Any | None:
        return self._past[batch_idx]

    def set_past(self, batch_idx: int, past_key_values: Any | None) -> None:
        length = past_key_values_length(past_key_values)
        if length > self.max_seq_len:
            raise ValueError(
                f"sequence length {length} exceeds max_seq_len {self.max_seq_len}"
            )
        self._past[batch_idx] = past_key_values
        self.pos[batch_idx] = length

    def length(self, batch_idx: int) -> int:
        return self.pos[batch_idx].item()

    def lengths(self) -> torch.Tensor:
        return self.pos.clone()

    def reset_slot(self, batch_idx: int) -> None:
        self._past[batch_idx] = None
        self.pos[batch_idx] = 0

    def free(self) -> None:
        self.batch_size = 0
        self.pos = None
        self._past = []
