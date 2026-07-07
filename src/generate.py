import argparse
from pathlib import Path

import tiktoken
import torch

from model.gpt_model import GptModel, GPT_CONFIG_124M
from inference.kv_cache import KVCache
from sampler import sample_greedy

GPT2_PAD_TOKEN_ID = 50256


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_kv_cache(model, device, dtype=None):
    cfg = model.cfg
    if dtype is None:
        dtype = next(model.parameters()).dtype
    return KVCache(
        num_layers=cfg["num_layers"],
        max_seq_len=cfg["context_length"],
        n_heads=cfg["num_heads"],
        head_dim=cfg["emb_dim"] // cfg["num_heads"],
        device=device,
        dtype=dtype,
    )


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = GptModel(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def batch_token_ids(token_lists, device, pad_id=GPT2_PAD_TOKEN_ID):
    """Left-pad token lists to a batch tensor and return per-row real lengths."""
    max_len = max(len(tokens) for tokens in token_lists)
    batch = []
    lengths = []
    for tokens in token_lists:
        lengths.append(len(tokens))
        padding = [pad_id] * (max_len - len(tokens))
        batch.append(padding + tokens)
    token_ids = torch.tensor(batch, device=device)
    input_lens = torch.tensor(lengths, device=device, dtype=torch.long)
    return token_ids, input_lens


def strip_left_pad(token_row, pad_id=GPT2_PAD_TOKEN_ID):
    tokens = token_row.tolist()
    while tokens and tokens[0] == pad_id:
        tokens.pop(0)
    return tokens


@torch.no_grad()
def generate(model, token_ids, max_new_tokens, input_lens=None):
    """Run static batched generation. ``token_ids`` has shape ``(batch, prompt_len)``."""
    model.eval()
    device = token_ids.device
    batch_size = token_ids.shape[0]
    cache = make_kv_cache(model, device)
    cache.init_batch(batch_size)

    if input_lens is None:
        input_lens = torch.full(
            (batch_size,),
            token_ids.shape[1],
            device=device,
            dtype=torch.long,
        )

    output = token_ids.clone()
    logits = model(output, kv_cache=cache, input_lens=input_lens)

    for _ in range(max_new_tokens):
        next_token = sample_greedy(logits[:, -1, :])
        output = torch.cat([output, next_token], dim=1)

        if _ == max_new_tokens - 1:
            break

        logits = model(next_token, kv_cache=cache)

    cache.free()
    return output


@torch.no_grad()
def generate_single(model, token_ids, max_new_tokens):
    """Generate one sequence. ``token_ids`` has shape ``(1, prompt_len)``."""
    return generate(model, token_ids, max_new_tokens)


def main():
    parser = argparse.ArgumentParser(description="Generate text with static batched KV-cache inference")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).parent / "model" / "checkpoints" / "gpt_model_checkpoint.pt",
        help="Path to a checkpoint saved as {config, model_state_dict}",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Prompt text; pass multiple times for static batching",
    )
    parser.add_argument("--max-new-tokens", type=int, default=50)
    args = parser.parse_args()

    prompts = args.prompt if args.prompt else ["Every effort moves you"]

    device = get_device()
    print(f"Device: {device}")
    print(f"Batch size: {len(prompts)}")

    if args.checkpoint.exists():
        print(f"Loading checkpoint: {args.checkpoint}")
        model = load_model(args.checkpoint, device)
    else:
        print(f"No checkpoint at {args.checkpoint} — using random weights")
        model = GptModel(GPT_CONFIG_124M).to(device)
        model.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    token_lists = [tokenizer.encode(prompt) for prompt in prompts]
    token_ids, input_lens = batch_token_ids(token_lists, device)
    output_ids = generate(model, token_ids, max_new_tokens=args.max_new_tokens, input_lens=input_lens)

    print("\n--- Generated ---")
    for i, prompt in enumerate(prompts):
        print(f"\n[{i}] prompt: {prompt!r}")
        print(tokenizer.decode(strip_left_pad(output_ids[i])))


if __name__ == "__main__":
    main()
