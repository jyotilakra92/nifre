import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parent.parent / "src"
MODEL = SRC / "model"
for path in (SRC, MODEL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from inference.backends.registry import load_backend


def test_load_gpt_backend():
    device = torch.device("cpu")
    model, tokenizer = load_backend("gpt", checkpoint=None, device=device)

    assert model.config.num_layers == 12
    assert model.config.pad_token_id == 50256
    assert tokenizer.pad_token_id == model.config.pad_token_id
    assert tokenizer.encode("hello")
