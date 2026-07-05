import torch

from model.generate import GPT2_PAD_TOKEN_ID, batch_token_ids, generate
from model.gpt_model import GPT_CONFIG_124M, GptModel
from inference.engine import Engine


def _strip_left_pad(token_row, pad_id=GPT2_PAD_TOKEN_ID):
    tokens = token_row.tolist()
    while tokens and tokens[0] == pad_id:
        tokens.pop(0)
    return tokens


def _tiny_model(device):
    cfg = dict(GPT_CONFIG_124M)
    cfg["num_layers"] = 2
    cfg["emb_dim"] = 32
    cfg["num_heads"] = 4
    cfg["context_length"] = 64
    return GptModel(cfg).to(device).eval()


def test_static_batch_matches_single():
    torch.manual_seed(42)
    device = torch.device("cpu")
    model = _tiny_model(device)

    prompts = [[1, 2, 3], [10, 20, 30, 40]]
    token_ids, input_lens = batch_token_ids(prompts, device)
    batched = generate(model, token_ids, max_new_tokens=2, input_lens=input_lens)

    for i, tokens in enumerate(prompts):
        single = generate(model, torch.tensor([tokens], device=device), max_new_tokens=2)
        assert _strip_left_pad(batched[i]) == single[0].tolist()


def test_model_runner_via_engine():
    torch.manual_seed(42)
    device = torch.device("cpu")
    model = _tiny_model(device)

    engine = Engine(model, max_concurrent_requests=2, device=device)
    req_a = engine.generate([1, 2, 3], max_new_tokens=2)
    req_b = engine.generate([10, 20, 30, 40], max_new_tokens=2)

    assert req_a.num_generated == 2
    assert req_b.num_generated == 2
