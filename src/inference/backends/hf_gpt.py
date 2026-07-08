"""Hugging Face GPT-2 backend (uses ``transformers.GPT2LMHeadModel`` directly)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch

from inference.data_model import ModelConfig
from inference.hf_kv_cache import HFKVCache
from inference.model_interface import InferenceModel, Tokenizer

GPT2_PAD_TOKEN_ID = 50256


def _require_transformers():
    try:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for the hf-gpt backend: pip install transformers"
        ) from exc
    return GPT2LMHeadModel, GPT2Tokenizer


def _truncate_position_embeddings(model, context_length: int) -> None:
    if context_length >= model.config.n_positions:
        return
    old_wpe = model.transformer.wpe
    new_wpe = torch.nn.Embedding(context_length, old_wpe.embedding_dim)
    new_wpe.weight.data.copy_(old_wpe.weight.data[:context_length])
    model.transformer.wpe = new_wpe
    model.config.n_positions = context_length


class HfGptTokenizer:
    def __init__(self, tokenizer, pad_token_id: int = GPT2_PAD_TOKEN_ID):
        self._tokenizer = tokenizer
        self._pad_token_id = pad_token_id

    def encode(self, text: str) -> List[int]:
        return self._tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        return self._tokenizer.decode(token_ids)

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id


class HfGptInferenceModel:
    """Adapter around Hugging Face GPT-2 for the nifre inference engine."""

    supports_paged_kv_cache = False
    supports_prefix_cache = False

    def __init__(
        self,
        model,
        *,
        context_length: int,
        pad_token_id: int = GPT2_PAD_TOKEN_ID,
    ):
        self._model = model
        hf_cfg = model.config
        self.config = ModelConfig(
            num_layers=hf_cfg.n_layer,
            max_seq_len=context_length,
            n_heads=hf_cfg.n_head,
            head_dim=hf_cfg.n_embd // hf_cfg.n_head,
            vocab_size=hf_cfg.vocab_size,
            pad_token_id=pad_token_id,
            block_size=16,
        )

    @property
    def dtype(self) -> torch.dtype:
        return next(self._model.parameters()).dtype

    def eval(self) -> None:
        self._model.eval()

    def make_kv_cache(self, device: torch.device) -> HFKVCache:
        return HFKVCache(self.config.max_seq_len, device=device)

    def _forward_row(
        self,
        token_ids: torch.Tensor,
        *,
        kv_cache: HFKVCache,
        slot: int,
        input_len: Optional[int],
    ) -> torch.Tensor:
        if input_len is not None:
            seq_len = token_ids.shape[0]
            start = seq_len - input_len
            input_ids = token_ids[start:].unsqueeze(0)
        else:
            input_ids = token_ids.unsqueeze(0)

        past = kv_cache.get_past(slot)
        if past is None:
            attention_mask = torch.ones(
                1,
                input_ids.shape[1],
                device=input_ids.device,
                dtype=torch.long,
            )
            outputs = self._model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=True,
            )
        else:
            outputs = self._model(
                input_ids,
                past_key_values=past,
                use_cache=True,
            )

        kv_cache.set_past(slot, outputs.past_key_values)
        return outputs.logits

    def __call__(
        self,
        token_ids,
        kv_cache=None,
        input_lens=None,
        cache_batch_indices=None,
    ):
        if kv_cache is None:
            attention_mask = torch.ones(
                token_ids.shape[0],
                token_ids.shape[1],
                device=token_ids.device,
                dtype=torch.long,
            )
            return self._model(token_ids, attention_mask=attention_mask).logits

        batch_size = token_ids.shape[0]
        logits_rows = []
        for i in range(batch_size):
            slot = cache_batch_indices[i] if cache_batch_indices is not None else i
            input_len = input_lens[i].item() if input_lens is not None else None
            logits_rows.append(
                self._forward_row(
                    token_ids[i],
                    kv_cache=kv_cache,
                    slot=slot,
                    input_len=input_len,
                )
            )
        return torch.cat(logits_rows, dim=0)


def load_hf_gpt_model(
    *,
    model_name: str,
    device: torch.device,
    context_length: int = 256,
):
    GPT2LMHeadModel, _ = _require_transformers()
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    model = GPT2LMHeadModel.from_pretrained(model_name, torch_dtype=dtype)
    _truncate_position_embeddings(model, context_length)
    model.to(device)
    model.eval()
    return model


def load_hf_gpt_backend(
    checkpoint: Optional[Path],
    device: torch.device,
    *,
    hf_model: str = "gpt2",
    context_length: int = 256,
) -> Tuple[HfGptInferenceModel, HfGptTokenizer]:
    del checkpoint  # weights come from Hugging Face Hub / cache, not nifre checkpoints
    GPT2LMHeadModel, GPT2Tokenizer = _require_transformers()

    hf_model_obj = load_hf_gpt_model(
        model_name=hf_model,
        device=device,
        context_length=context_length,
    )
    pad_token_id = GPT2_PAD_TOKEN_ID
    wrapper = HfGptInferenceModel(
        hf_model_obj,
        context_length=context_length,
        pad_token_id=pad_token_id,
    )
    tokenizer = GPT2Tokenizer.from_pretrained(hf_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    hf_tokenizer = HfGptTokenizer(tokenizer, pad_token_id=pad_token_id)
    return wrapper, hf_tokenizer
