from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch

from inference.backends.gpt import load_gpt_backend
from inference.backends.hf_auto import load_hf_backend
from inference.model_interface import InferenceModel, Tokenizer

BackendLoader = Callable[..., Tuple[InferenceModel, Tokenizer]]

BACKENDS: Dict[str, BackendLoader] = {
    "gpt": load_gpt_backend,
    "hf": load_hf_backend,
}

# Backends whose weights come from the Hugging Face Hub (need --hf-model, not --checkpoint).
HF_BACKENDS = {"hf"}


def is_hf_backend(name: str) -> bool:
    return name in HF_BACKENDS


def list_backends():
    return sorted(BACKENDS.keys())


def load_backend(
    name: str,
    checkpoint: Optional[Path],
    device: torch.device,
    **loader_kwargs,
) -> Tuple[InferenceModel, Tokenizer]:
    if name not in BACKENDS:
        supported = ", ".join(list_backends())
        raise ValueError(f"unknown model backend {name!r}; supported: {supported}")
    return BACKENDS[name](checkpoint, device, **loader_kwargs)
