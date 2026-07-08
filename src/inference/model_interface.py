from typing import List, Protocol, Union

import torch

from inference.data_model import ModelConfig
from inference.hf_kv_cache import HFKVCache
from inference.kv_cache import KVCache
from inference.paged_kv_cache import PagedKVCache

KVCacheType = Union[KVCache, PagedKVCache, HFKVCache]


class Tokenizer(Protocol):
    def encode(self, text: str) -> List[int]:
        ...

    def decode(self, token_ids: List[int]) -> str:
        ...

    @property
    def pad_token_id(self) -> int:
        ...


class InferenceModel(Protocol):
    """Any causal LM the engine can run with a KV cache."""

    config: ModelConfig

    def eval(self) -> None:
        ...

    def __call__(
        self,
        token_ids: torch.Tensor,
        kv_cache: KVCacheType = None,
        input_lens: torch.Tensor = None,
        cache_batch_indices: List[int] = None,
    ) -> torch.Tensor:
        ...
