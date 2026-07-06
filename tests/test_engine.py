import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parent.parent / "src"
MODEL = SRC / "model"
for path in (SRC, MODEL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from inference.backends.gpt import GptInferenceModel
from inference.engine import Engine
from model.gpt_model import GPT_CONFIG_124M, GptModel


def _tiny_gpt_model(device):
    cfg = dict(GPT_CONFIG_124M)
    cfg["num_layers"] = 2
    cfg["emb_dim"] = 32
    cfg["num_heads"] = 4
    cfg["context_length"] = 64
    return GptInferenceModel(GptModel(cfg).to(device).eval())


def test_engine_smoke():
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = _tiny_gpt_model(device)

    engine = Engine(model, max_concurrent_requests=2, device=device)
    engine.add_request([1, 2, 3], max_new_tokens=2)
    engine.add_request([10, 20, 30, 40], max_new_tokens=2)
    engine.add_request([5, 6], max_new_tokens=2)
    engine.run_until_done()

    assert len(engine.get_completed()) == 3
    for request in engine.get_completed().values():
        assert request.num_generated == 2
