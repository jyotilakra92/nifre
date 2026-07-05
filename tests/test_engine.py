import torch

from model.gpt_model import GPT_CONFIG_124M, GptModel
from inference.engine import Engine


def test_engine_smoke():
    torch.manual_seed(0)
    device = torch.device("cpu")
    cfg = dict(GPT_CONFIG_124M)
    cfg["num_layers"] = 2
    cfg["emb_dim"] = 32
    cfg["num_heads"] = 4
    cfg["context_length"] = 64
    model = GptModel(cfg).to(device).eval()

    engine = Engine(model, max_concurrent_requests=2, device=device)
    engine.add_request([1, 2, 3], max_new_tokens=2)
    engine.add_request([10, 20, 30, 40], max_new_tokens=2)
    engine.add_request([5, 6], max_new_tokens=2)
    engine.run_until_done()

    assert len(engine.get_completed()) == 3
    for request in engine.get_completed().values():
        assert request.num_generated == 2
