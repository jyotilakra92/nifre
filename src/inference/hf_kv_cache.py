"""KV cache backed by Hugging Face ``past_key_values``.

Hugging Face models run their own (dense) attention kernels, so they cannot use
nifre's block-paged KV cache directly. They *can*, however, benefit from prefix
caching: when a new prompt shares a leading span of tokens with a previously
seen prompt, the cached key/value tensors for that span are reused and the
matching tokens are skipped during prefill.

To align with nifre's native ``PrefixCache``, reuse is tracked at **block
granularity** using block-chained hashing: each full block of ``block_size``
tokens is keyed by ``(parent_hash, block_tokens)``. Blocks are stored once and
shared across every prompt whose prefix hashes to the same chain, so common
prefixes are not duplicated in memory. Lookup walks the chain and stops at the
first missing block, exactly like the paged path.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any

import torch

CacheKey = tuple[int, tuple[int, ...]]


def past_key_values_length(past_key_values: Any | None) -> int:
    if past_key_values is None:
        return 0
    get_seq_length = getattr(past_key_values, "get_seq_length", None)
    if get_seq_length is not None:
        return int(get_seq_length())
    return past_key_values[0][0].shape[-2]


def _past_layers(past: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return per-layer ``(keys, values)`` tensors for any HF cache flavour."""
    layers = getattr(past, "layers", None)
    if layers is not None:  # transformers >= 4.5x DynamicCache
        return [(layer.keys, layer.values) for layer in layers]
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):  # older DynamicCache
        return list(zip(past.key_cache, past.value_cache))
    return [(k, v) for (k, v) in past]  # legacy tuple


def _build_past(layer_tensors: list[tuple[torch.Tensor, torch.Tensor]]) -> Any:
    """Build a fresh ``DynamicCache`` from per-layer ``(keys, values)`` tensors."""
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    for layer_idx, (keys, values) in enumerate(layer_tensors):
        cache.update(keys, values, layer_idx)
    return cache


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.element_size() * tensor.nelement()


class _BlockPrefixCache:
    """Block-chained prefix cache holding per-block HF key/value slices.

    Each entry maps a chained block hash to that block's per-layer
    ``(keys, values)`` slices (sequence length == ``block_size``). Shared prefix
    blocks map to the same key, so they are stored only once.
    """

    ROOT_HASH = 0

    def __init__(self, block_size: int, max_entries: int = 1024) -> None:
        self.block_size = block_size
        self.max_entries = max_entries
        self.entries: OrderedDict[CacheKey, list[tuple[torch.Tensor, torch.Tensor]]] = (
            OrderedDict()
        )
        self.lookups = 0
        self.hits = 0
        self.misses = 0
        self.tokens_reused = 0
        self.memory_bytes = 0
        self._lookup_time_sec = 0.0

    @staticmethod
    def _chain(parent_hash: int, block_tokens: tuple[int, ...]) -> int:
        return hash((parent_hash, block_tokens))

    def insert(
        self,
        token_ids: list[int],
        layers: list[tuple[torch.Tensor, torch.Tensor]],
        cached_len: int,
    ) -> None:
        parent_hash = self.ROOT_HASH
        num_full_blocks = min(len(token_ids), cached_len) // self.block_size

        for block_idx in range(num_full_blocks):
            start = block_idx * self.block_size
            block_tokens = tuple(token_ids[start : start + self.block_size])
            key = (parent_hash, block_tokens)

            if key not in self.entries:
                while len(self.entries) >= self.max_entries:
                    self._evict_oldest()
                stop = start + self.block_size
                block_slices = [
                    (
                        keys[:, :, start:stop, :].detach().clone(),
                        values[:, :, start:stop, :].detach().clone(),
                    )
                    for (keys, values) in layers
                ]
                self.entries[key] = block_slices
                self.memory_bytes += sum(
                    _tensor_bytes(k) + _tensor_bytes(v) for (k, v) in block_slices
                )

            self.entries.move_to_end(key)
            parent_hash = self._chain(parent_hash, block_tokens)

    def lookup(self, token_ids: list[int]) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
        """Return concatenated per-layer ``(keys, values)`` for the longest cached prefix."""
        start_time = time.perf_counter()
        self.lookups += 1
        parent_hash = self.ROOT_HASH
        matched: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        # Leave at least one token for the prefill forward pass.
        max_full_blocks = (len(token_ids) - 1) // self.block_size

        for block_idx in range(max_full_blocks):
            start = block_idx * self.block_size
            block_tokens = tuple(token_ids[start : start + self.block_size])
            key = (parent_hash, block_tokens)
            node = self.entries.get(key)
            if node is None:
                break
            matched.append(node)
            self.entries.move_to_end(key)
            parent_hash = self._chain(parent_hash, block_tokens)

        self._lookup_time_sec += time.perf_counter() - start_time

        if not matched:
            self.misses += 1
            return None

        self.hits += 1
        self.tokens_reused += len(matched) * self.block_size

        num_layers = len(matched[0])
        layer_tensors: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(num_layers):
            keys = torch.cat([block[layer_idx][0] for block in matched], dim=2)
            values = torch.cat([block[layer_idx][1] for block in matched], dim=2)
            layer_tensors.append((keys, values))
        return layer_tensors

    def _evict_oldest(self) -> None:
        _key, block_slices = self.entries.popitem(last=False)
        self.memory_bytes -= sum(
            _tensor_bytes(k) + _tensor_bytes(v) for (k, v) in block_slices
        )

    def clear(self) -> None:
        self.entries.clear()
        self.memory_bytes = 0

    def snapshot(self) -> dict[str, float | int]:
        hit_rate = self.hits / self.lookups if self.lookups else 0.0
        avg_lookup_ms = (
            (self._lookup_time_sec / self.lookups) * 1000 if self.lookups else 0.0
        )
        return {
            "block_size": self.block_size,
            "lookups": self.lookups,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 4),
            "tokens_reused": self.tokens_reused,
            "entries": len(self.entries),
            "max_entries": self.max_entries,
            "memory_mb": round(self.memory_bytes / (1024**2), 4),
            "avg_lookup_ms": round(avg_lookup_ms, 4),
        }


class HFKVCache:
    """Per-slot storage for HF ``past_key_values`` during continuous batching.

    When ``enable_prefix_cache`` is set, completed prompts register their blocks
    in a block-chained prefix cache and later prompts reuse matching blocks.
    """

    def __init__(
        self,
        max_seq_len: int,
        device: torch.device | str = "cpu",
        *,
        enable_prefix_cache: bool = False,
        block_size: int = 16,
        prefix_cache_max_entries: int = 1024,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.device = device
        self.enable_prefix_cache = enable_prefix_cache
        self.block_size = block_size
        self.batch_size = 0
        self.pos: torch.Tensor | None = None
        self._past: list[Any | None] = []
        self.prefix_cache = (
            _BlockPrefixCache(block_size, max_entries=prefix_cache_max_entries)
            if enable_prefix_cache
            else None
        )

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
        if self.prefix_cache is not None:
            self.prefix_cache.clear()

    # -- prefix cache -----------------------------------------------------

    def register_prefix(self, batch_idx: int, token_ids: list[int]) -> None:
        if self.prefix_cache is None:
            return
        past = self._past[batch_idx]
        if past is None:
            return
        cached_len = past_key_values_length(past)
        self.prefix_cache.insert(token_ids, _past_layers(past), cached_len)

    def try_load_prefix(self, batch_idx: int, token_ids: list[int]) -> int:
        if self.prefix_cache is None:
            return 0
        layer_tensors = self.prefix_cache.lookup(token_ids)
        if layer_tensors is None:
            return 0
        matched_len = layer_tensors[0][0].shape[2]
        self._past[batch_idx] = _build_past(layer_tensors)
        self.pos[batch_idx] = matched_len
        return matched_len

    def prefix_stats(self) -> dict[str, float | int] | None:
        if self.prefix_cache is None:
            return None
        return self.prefix_cache.snapshot()
