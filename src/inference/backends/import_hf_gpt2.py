"""Import Hugging Face GPT-2 weights into a nifre checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from inference.models.gpt import GptModel


def hf_config_for_nifre(*, context_length: int = 256) -> dict:
    return {
        "vocab_size": 50257,
        "context_length": context_length,
        "emb_dim": 768,
        "num_heads": 12,
        "num_layers": 12,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }


def _map_hf_state_dict(hf_model, context_length: int) -> dict[str, torch.Tensor]:
    hf = hf_model.state_dict()
    mapped: dict[str, torch.Tensor] = {}

    mapped["token_embedding.weight"] = hf["transformer.wte.weight"]
    mapped["position_embedding.weight"] = hf["transformer.wpe.weight"][:context_length]
    mapped["final_norm.scale"] = hf["transformer.ln_f.weight"]
    mapped["final_norm.shift"] = hf["transformer.ln_f.bias"]
    mapped["out_head.weight"] = hf["transformer.wte.weight"]

    for layer in range(12):
        prefix = f"transformer.h.{layer}"
        block = f"trf_blocks.{layer}"

        mapped[f"{block}.norm1.scale"] = hf[f"{prefix}.ln_1.weight"]
        mapped[f"{block}.norm1.shift"] = hf[f"{prefix}.ln_1.bias"]
        mapped[f"{block}.norm2.scale"] = hf[f"{prefix}.ln_2.weight"]
        mapped[f"{block}.norm2.shift"] = hf[f"{prefix}.ln_2.bias"]

        c_attn_w = hf[f"{prefix}.attn.c_attn.weight"]
        c_attn_b = hf[f"{prefix}.attn.c_attn.bias"]
        q_w, k_w, v_w = c_attn_w.split(768, dim=1)
        q_b, k_b, v_b = c_attn_b.split(768, dim=0)

        mapped[f"{block}.att.W_query.weight"] = q_w.t()
        mapped[f"{block}.att.W_key.weight"] = k_w.t()
        mapped[f"{block}.att.W_value.weight"] = v_w.t()
        mapped[f"{block}.att.W_query.bias"] = q_b
        mapped[f"{block}.att.W_key.bias"] = k_b
        mapped[f"{block}.att.W_value.bias"] = v_b

        mapped[f"{block}.att.out_proj.weight"] = hf[f"{prefix}.attn.c_proj.weight"].t()
        mapped[f"{block}.att.out_proj.bias"] = hf[f"{prefix}.attn.c_proj.bias"]

        mapped[f"{block}.ff.layers.0.weight"] = hf[f"{prefix}.mlp.c_fc.weight"].t()
        mapped[f"{block}.ff.layers.0.bias"] = hf[f"{prefix}.mlp.c_fc.bias"]
        mapped[f"{block}.ff.layers.2.weight"] = hf[f"{prefix}.mlp.c_proj.weight"].t()
        mapped[f"{block}.ff.layers.2.bias"] = hf[f"{prefix}.mlp.c_proj.bias"]

    return mapped


def import_hf_gpt2_checkpoint(
    output_path: Path,
    *,
    model_name: str = "gpt2",
    context_length: int = 256,
) -> Path:
    try:
        from transformers import GPT2LMHeadModel
    except ImportError as exc:
        raise SystemExit(
            "transformers is required: pip install transformers"
        ) from exc

    config = hf_config_for_nifre(context_length=context_length)
    hf_model = GPT2LMHeadModel.from_pretrained(model_name)
    hf_model.eval()

    model = GptModel(config)
    model.load_state_dict(_map_hf_state_dict(hf_model, context_length), strict=False)
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config,
            "model_state_dict": model.state_dict(),
            "source": model_name,
        },
        output_path,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face GPT-2 into a nifre checkpoint.pt"
    )
    parser.add_argument(
        "--model",
        default="gpt2",
        help="Hugging Face model id (default: gpt2)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=256,
        help="Truncate positional embeddings to this length (match vLLM --max-model-len)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/checkpoints/gpt2_hf_checkpoint.pt"),
        help="Output checkpoint path",
    )
    args = parser.parse_args()

    path = import_hf_gpt2_checkpoint(
        args.output,
        model_name=args.model,
        context_length=args.context_length,
    )
    print(f"Wrote {path} from {args.model} (context_length={args.context_length})")


if __name__ == "__main__":
    main()
