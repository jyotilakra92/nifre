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


def kv_cache_bytes(cache) -> int:
    if cache is None or cache.k is None:
        return 0
    elem_size = torch.tensor([], dtype=cache.dtype).element_size()
    per_tensor = (
        cache.batch_size * cache.max_seq_len * cache.n_heads * cache.head_dim * elem_size
    )
    return int(per_tensor * 2 * cache.num_layers)


def kv_cache_used_bytes(cache) -> int:
    if cache is None or cache.pos is None:
        return 0
    elem_size = torch.tensor([], dtype=cache.dtype).element_size()
    tokens_used = int(cache.pos.sum().item())
    per_token = cache.n_heads * cache.head_dim * elem_size * 2 * cache.num_layers
    return tokens_used * per_token


def kv_cache_utilization(cache) -> float:
    if cache is None or cache.pos is None or cache.batch_size == 0:
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
        }

    def sample_to_store(self, store) -> None:
        info = self.snapshot()
        store.record_runtime_sample(
            gpu_utilization=float(info["gpu_utilization_pct"]),
            gpu_memory_gb=float(info["gpu_memory_used_gb"]),
            kv_cache_utilization=float(info["kv_cache_utilization_pct"]) / 100.0,
        )
