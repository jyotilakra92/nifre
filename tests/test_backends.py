import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference.backends.registry import load_backend


def test_load_gpt_backend():
    device = torch.device("cpu")
    model, tokenizer = load_backend("gpt", checkpoint=None, device=device)

    assert model.config.num_layers == 12
    assert model.config.pad_token_id == 50256
    assert tokenizer.pad_token_id == model.config.pad_token_id
    assert tokenizer.encode("hello")


def test_load_hf_backend():
    pytest = __import__("pytest")
    pytest.importorskip("transformers")

    device = torch.device("cpu")
    model, tokenizer = load_backend(
        "hf",
        checkpoint=None,
        device=device,
        hf_model="gpt2",
        context_length=64,
    )

    assert model.config.num_layers == 12
    assert model.config.max_seq_len == 64
    assert getattr(model, "supports_paged_kv_cache", True) is False
    assert tokenizer.encode("hello")
