"""Generic Hugging Face causal-LM backend.

Runs almost any ``AutoModelForCausalLM`` (Llama, Qwen, Mistral, Phi, GPT-2, ...)
behind the nifre engine, using Hugging Face's own ``past_key_values`` cache.

Architecture-specific details (RoPE, GQA, learned vs. rotary positions) are
handled inside the HF model itself — this adapter only bridges nifre's per-slot
batching to HF's cache interface, so it is model-agnostic.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch

from inference.data_model import ModelConfig
from inference.hf_kv_cache import HFKVCache


def _require_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for the hf backend: pip install transformers"
        ) from exc
    return AutoModelForCausalLM, AutoTokenizer


def _config_value(cfg, *names, default=None):
    for name in names:
        value = getattr(cfg, name, None)
        if value is not None:
            return value
    return default


class HfTokenizer:
    def __init__(self, tokenizer, pad_token_id: int):
        self._tokenizer = tokenizer
        self._pad_token_id = pad_token_id

    def encode(self, text: str) -> List[int]:
        return self._tokenizer.encode(text)

    def decode(self, token_ids: List[int]) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id


class HfCausalLM:
    """Adapter around any Hugging Face causal LM for the nifre inference engine."""

    # HF runs its own dense attention, so nifre's block-paged KV cache does not
    # apply. Prefix caching is supported at the ``past_key_values`` level.
    supports_paged_kv_cache = False
    supports_prefix_cache = True

    def __init__(self, model, *, context_length: int, pad_token_id: int):
        self._model = model
        cfg = model.config

        n_layers = _config_value(cfg, "num_hidden_layers", "n_layer")
        n_heads = _config_value(cfg, "num_attention_heads", "n_head")
        hidden = _config_value(cfg, "hidden_size", "n_embd")
        head_dim = _config_value(cfg, "head_dim", default=hidden // n_heads)
        vocab = _config_value(cfg, "vocab_size")
        max_pos = _config_value(
            cfg, "max_position_embeddings", "n_positions", default=context_length
        )

        self.config = ModelConfig(
            num_layers=n_layers,
            max_seq_len=min(context_length, max_pos) if max_pos else context_length,
            n_heads=n_heads,
            head_dim=head_dim,
            vocab_size=vocab,
            pad_token_id=pad_token_id,
            block_size=16,
        )

    @property
    def dtype(self) -> torch.dtype:
        return next(self._model.parameters()).dtype

    def eval(self) -> None:
        self._model.eval()

    def make_kv_cache(
        self, device: torch.device, *, enable_prefix_cache: bool = False
    ) -> HFKVCache:
        return HFKVCache(
            self.config.max_seq_len,
            device=device,
            enable_prefix_cache=enable_prefix_cache,
            block_size=self.config.block_size,
        )

    def _forward_row(self, token_ids, *, kv_cache, slot, input_len):
        if input_len is not None:
            seq_len = token_ids.shape[0]
            start = seq_len - input_len
            input_ids = token_ids[start:].unsqueeze(0)
        else:
            input_ids = token_ids.unsqueeze(0)

        past = kv_cache.get_past(slot)
        if past is None:
            attention_mask = torch.ones(
                1, input_ids.shape[1], device=input_ids.device, dtype=torch.long
            )
            outputs = self._model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=True,
            )
        else:
            outputs = self._model(input_ids, past_key_values=past, use_cache=True)

        kv_cache.set_past(slot, outputs.past_key_values)
        # Only the last position is ever sampled, and concurrent rows may have
        # different chunk lengths (varying prompts, prefix-cache hits), so return
        # a fixed (1, 1, vocab) slice to keep the batch concatenable.
        return outputs.logits[:, -1:, :]

    def __call__(self, token_ids, kv_cache=None, input_lens=None, cache_batch_indices=None):
        if kv_cache is None:
            attention_mask = torch.ones(
                token_ids.shape[0], token_ids.shape[1], device=token_ids.device, dtype=torch.long
            )
            return self._model(token_ids, attention_mask=attention_mask).logits

        batch_size = token_ids.shape[0]
        logits_rows = []
        for i in range(batch_size):
            slot = cache_batch_indices[i] if cache_batch_indices is not None else i
            input_len = input_lens[i].item() if input_lens is not None else None
            logits_rows.append(
                self._forward_row(token_ids[i], kv_cache=kv_cache, slot=slot, input_len=input_len)
            )
        return torch.cat(logits_rows, dim=0)


def load_hf_model(*, model_name: str, device: torch.device, context_length: int):
    AutoModelForCausalLM, _ = _require_transformers()
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    model.eval()
    return model


def load_hf_backend(
    checkpoint: Optional[Path],
    device: torch.device,
    *,
    hf_model: str = "gpt2",
    context_length: int = 2048,
) -> Tuple[HfCausalLM, HfTokenizer]:
    del checkpoint  # weights come from the Hugging Face Hub / cache, not nifre checkpoints
    _, AutoTokenizer = _require_transformers()

    model = load_hf_model(model_name=hf_model, device=device, context_length=context_length)
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = 0

    wrapper = HfCausalLM(model, context_length=context_length, pad_token_id=pad_token_id)
    hf_tokenizer = HfTokenizer(tokenizer, pad_token_id=pad_token_id)
    return wrapper, hf_tokenizer
