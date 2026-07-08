"""GPU, KV-cache, and runtime metadata probes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import torch

if TYPE_CHECKING:
    from inference.engine import Engine


RUNTIME_CHOICES = ("custom", "vLLM", "SGLang", "TRT-LLM")
PRECISION_CHOICES = ("fp16", "bf16", "int8", "fp8")


def dtype_to_precision(dtype: torch.dtype) -> str:
    mapping = {
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        torch.int8: "int8",
    }
    for attr in ("float8_e4m3fn", "float8_e5m2"):
        fp8 = getattr(torch, attr, None)
        if fp8 is not None:
            mapping[fp8] = "fp8"
    return mapping.get(dtype, "fp16")


def _is_paged_cache(cache) -> bool:
    return hasattr(cache, "allocated_blocks") and hasattr(cache, "block_size")


def kv_cache_bytes(cache) -> int:
    # HF's dense cache (HFKVCache) exposes no contiguous K/V tensor to size, so
    # nifre cannot account its memory here.
    if cache is None or getattr(cache, "k", None) is None:
        return 0
    elem_size = torch.tensor([], dtype=cache.dtype).element_size()
    if _is_paged_cache(cache):
        per_tensor = (
            cache.num_blocks * cache.block_size * cache.n_heads * cache.head_dim * elem_size
        )
    else:
        per_tensor = (
            cache.batch_size * cache.max_seq_len * cache.n_heads * cache.head_dim * elem_size
        )
    return int(per_tensor * 2 * cache.num_layers)


def kv_cache_used_bytes(cache) -> int:
    if cache is None or getattr(cache, "pos", None) is None:
        return 0
    if not hasattr(cache, "n_heads"):  # e.g. HFKVCache
        return 0
    elem_size = torch.tensor([], dtype=cache.dtype).element_size()
    tokens_used = int(cache.pos.sum().item())
    per_token = cache.n_heads * cache.head_dim * elem_size * 2 * cache.num_layers
    return tokens_used * per_token


def kv_cache_utilization(cache) -> float:
    if cache is None:
        return 0.0
    if _is_paged_cache(cache):
        if cache.num_blocks == 0:
            return 0.0
        return float(cache.allocated_blocks) / cache.num_blocks
    if cache.pos is None or cache.batch_size == 0:
        return 0.0
    capacity = cache.batch_size * cache.max_seq_len
    if capacity == 0:
        return 0.0
    return float(cache.pos.sum().item()) / capacity


class RuntimeProbe:
    """Collects device and model metadata for the dashboard."""

    def __init__(
        self,
        *,
        runtime: str = "custom",
        model_name: str = "unknown",
        precision: Optional[str] = None,
    ) -> None:
        if runtime not in RUNTIME_CHOICES:
            raise ValueError(f"runtime must be one of {RUNTIME_CHOICES}")
        self.runtime = runtime
        self.model_name = model_name
        self._precision_override = precision
        self._engine: Optional["Engine"] = None

    def attach(self, engine: "Engine") -> None:
        self._engine = engine

    def _gpu_stats(self) -> Dict[str, float]:
        if not torch.cuda.is_available():
            return {"gpu_utilization": 0.0, "gpu_memory_gb": 0.0}

        device = self._engine.device if self._engine else torch.device("cuda")
        if device.type != "cuda":
            return {"gpu_utilization": 0.0, "gpu_memory_gb": 0.0}

        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        total = torch.cuda.get_device_properties(device).total_memory
        memory_gb = allocated / (1024**3)
        utilization = reserved / total if total else 0.0
        return {
            "gpu_utilization": round(utilization * 100, 2),
            "gpu_memory_gb": round(memory_gb, 3),
        }

    def _engine_config(self) -> Dict[str, object]:
        engine = self._engine
        if engine is None:
            return {}

        cache = engine.cache
        use_paged = engine.use_paged_kv_cache
        config: Dict[str, object] = {
            "use_paged_kv_cache": use_paged,
            "use_prefix_cache": getattr(engine, "use_prefix_cache", False),
            "prefill_chunk_size": engine.prefill_chunk_size,
            "max_tokens_per_step": engine.max_tokens_per_step,
            "max_concurrent_requests": engine.max_concurrent_requests,
            "cache_type": "paged" if use_paged else "dense",
        }

        if cache is not None and _is_paged_cache(cache):
            prefix_info = (
                cache.prefix_cache.snapshot()
                if getattr(cache, "prefix_cache", None) is not None
                else None
            )
            config.update(
                {
                    "block_size": cache.block_size,
                    "num_blocks": cache.num_blocks,
                    "allocated_blocks": cache.allocated_blocks,
                    "prefix_cache": prefix_info,
                }
            )
        elif cache is not None:
            prefix_info = (
                cache.prefix_stats() if hasattr(cache, "prefix_stats") else None
            )
            config.update(
                {
                    "block_size": getattr(cache, "block_size", None),
                    "num_blocks": None,
                    "allocated_blocks": None,
                    "max_seq_len": cache.max_seq_len,
                    "batch_size": cache.batch_size,
                    "prefix_cache": prefix_info,
                }
            )
        else:
            config["block_size"] = engine.model.config.block_size if use_paged else None
            config["num_blocks"] = None
            config["allocated_blocks"] = 0

        return config

    def snapshot(self) -> Dict[str, object]:
        engine = self._engine
        cache = engine.cache if engine else None
        gpu = self._gpu_stats()

        precision = self._precision_override
        if precision is None and engine is not None:
            dtype = getattr(engine.model, "dtype", torch.float16)
            precision = dtype_to_precision(dtype)

        total_bytes = kv_cache_bytes(cache)
        used_bytes = kv_cache_used_bytes(cache)
        util = kv_cache_utilization(cache)

        return {
            "gpu_utilization_pct": gpu["gpu_utilization"],
            "gpu_memory_used_gb": gpu["gpu_memory_gb"],
            "kv_cache_memory_gb": round(total_bytes / (1024**3), 4),
            "kv_cache_used_gb": round(used_bytes / (1024**3), 4),
            "kv_cache_utilization_pct": round(util * 100, 2),
            "runtime": self.runtime,
            "model_name": self.model_name,
            "precision": precision or "fp16",
            "device": str(engine.device) if engine else "cpu",
            "engine_config": self._engine_config(),
        }

    def sample_to_store(self, store) -> None:
        info = self.snapshot()
        store.record_runtime_sample(
            gpu_utilization=float(info["gpu_utilization_pct"]),
            gpu_memory_gb=float(info["gpu_memory_used_gb"]),
            kv_cache_utilization=float(info["kv_cache_utilization_pct"]) / 100.0,
        )
