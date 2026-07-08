from pathlib import Path
from typing import Optional, Tuple

import tiktoken
import torch

from inference.data_model import ModelConfig
from inference.model_interface import InferenceModel, Tokenizer
from inference.models.gpt import GPT_CONFIG_124M, GptModel

GPT2_PAD_TOKEN_ID = 50256


class TiktokenTokenizer:
    def __init__(self, encoding_name: str = "gpt2", pad_token_id: int = GPT2_PAD_TOKEN_ID):
        self._encoding = tiktoken.get_encoding(encoding_name)
        self._pad_token_id = pad_token_id

    def encode(self, text: str):
        return self._encoding.encode(text)

    def decode(self, token_ids):
        return self._encoding.decode(token_ids)

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id


class GptInferenceModel:
    """Adapter that wraps the local GPT implementation behind InferenceModel."""

    def __init__(self, model: GptModel, pad_token_id: int = GPT2_PAD_TOKEN_ID):
        self._model = model
        cfg = model.cfg
        self.config = ModelConfig(
            num_layers=cfg["num_layers"],
            max_seq_len=cfg["context_length"],
            n_heads=cfg["num_heads"],
            head_dim=cfg["emb_dim"] // cfg["num_heads"],
            vocab_size=cfg["vocab_size"],
            pad_token_id=pad_token_id,
            block_size=cfg.get("block_size", 16),
        )

    @property
    def dtype(self):
        return next(self._model.parameters()).dtype

    def eval(self) -> None:
        self._model.eval()

    def __call__(self, token_ids, kv_cache=None, input_lens=None, cache_batch_indices=None):
        return self._model(
            token_ids,
            kv_cache=kv_cache,
            input_lens=input_lens,
            cache_batch_indices=cache_batch_indices,
        )


def load_gpt_model(checkpoint: Optional[Path], device: torch.device) -> GptModel:
    if checkpoint is not None and checkpoint.exists():
        checkpoint_data = torch.load(checkpoint, map_location=device)
        model = GptModel(checkpoint_data["config"])
        model.load_state_dict(checkpoint_data["model_state_dict"])
    else:
        model = GptModel(GPT_CONFIG_124M)
    model.to(device)
    model.eval()
    return model


def load_gpt_backend(
    checkpoint: Optional[Path],
    device: torch.device,
) -> Tuple[GptInferenceModel, TiktokenTokenizer]:
    model = load_gpt_model(checkpoint, device)
    wrapper = GptInferenceModel(model)
    tokenizer = TiktokenTokenizer(pad_token_id=wrapper.config.pad_token_id)
    return wrapper, tokenizer
