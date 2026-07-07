from typing import List, Protocol

import torch

from inference.data_model import ModelConfig
from inference.kv_cache import KVCache


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
        kv_cache: KVCache = None,
        input_lens: torch.Tensor = None,
        cache_batch_indices: List[int] = None,
    ) -> torch.Tensor:
        ...
