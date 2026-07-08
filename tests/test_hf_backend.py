import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("transformers")

SRC = Path(__file__).resolve().parent.parent / "src"
MODEL = SRC / "model"
for path in (SRC, MODEL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from inference.backends.registry import load_backend
from inference.engine import Engine


def test_hf_gpt_engine_generate():
    device = torch.device("cpu")
    model, tokenizer = load_backend(
        "hf-gpt",
        checkpoint=None,
        device=device,
        context_length=64,
    )
    engine = Engine(model, max_concurrent_requests=2, device=device)
    prompt = tokenizer.encode("Hello")
    result = engine.generate(prompt, max_new_tokens=3)
    assert result.state.value == "finished"
    assert len(result.output_token_ids) == 3
