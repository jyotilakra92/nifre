# Inference Engine

LLM inference engine with KV-cache, static batching, continuous batching, and a model-agnostic backend interface. Includes a FastAPI server and a reference GPT backend.

## Features

- **KV cache** — prefill + decode without recomputing past attention
- **Static batching** — run multiple prompts in one forward pass (fixed batch)
- **Continuous batching** — requests join and leave between decode steps
- **Chunked prefill** — long prompts are cached in fixed-size chunks so decode can interleave
- **Paged KV cache** — block-pooled K/V storage with per-sequence block tables (engine default)
- **Prefix caching** — reuse cached K/V blocks across requests that share prompt prefixes
- **Model-agnostic API** — plug in backends via `InferenceModel` + `Tokenizer`
- **FastAPI server** — HTTP completions with blocking JSON or SSE streaming (`stream: true`)
- **Observability dashboard** — request health, latency, throughput, GPU/runtime, optimization history

## Project layout

```text
nifre/
├── src/
│   ├── inference/          # Engine, scheduler, server, backends
│   │   ├── engine.py
│   │   ├── scheduler.py
│   │   ├── kv_cache.py
│   │   ├── paged_kv_cache.py
│   │   ├── prefix_cache.py
│   │   ├── block_allocator.py
│   │   ├── block_table.py
│   │   ├── model_runner.py
│   │   ├── server.py
│   │   └── backends/       # Model adapters (hf-gpt, gpt)
│   ├── generate.py         # Static-batched CLI
│   ├── bench.py            # Synthetic load generator for auto-tune / perf testing
│   ├── autotune/           # Classifier, policy, controller, admin API
│   ├── sampler.py          # Greedy sampling helper
│   └── observability/      # Metrics, dashboard, optimization tracking
│       ├── metrics_store.py
│       ├── collector.py
│       ├── runtime_probe.py
│       ├── optimization.py
│       └── dashboard/      # FastAPI dashboard UI
│   └── model/              # Reference GPT implementation
│       ├── gpt_model.py
│       └── attention.py
├── tests/                  # Smoke tests
└── requirements.txt
```

Set `PYTHONPATH` so both packages resolve:

```bash
export PYTHONPATH=src:src/model
```

## Setup

```bash
cd nifre
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Checkpoint (optional)

For sensible text output, place a trained checkpoint at:

```text
src/model/checkpoints/gpt_model_checkpoint.pt
```

Expected format:

```python
{
    "config": {...},           # GPT config dict
    "model_state_dict": {...}
}
```

Without a checkpoint, the server falls back to **random weights** (output will be gibberish, but the pipeline still runs).

## Run tests

```bash
PYTHONPATH=src:src/model python3 -m tests
```

## FastAPI server

Start the server (loads Hugging Face `gpt2` by default):

```bash
PYTHONPATH=src:src/model python3 -m inference.server \
  --model hf-gpt \
  --hf-model gpt2 \
  --context-length 256 \
  --port 8000 \
  --max-concurrent 2
```

The custom PyTorch GPT backend is still available with `--model gpt` and an optional checkpoint.

Or with uvicorn:

```bash
PYTHONPATH=src:src/model uvicorn inference.server:app --host 127.0.0.1 --port 8000
```

Interactive API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

The server starts a background **EngineWorker** on launch. HTTP handlers submit requests via `generate` / `generate_stream` on the worker; a single thread owns `engine.step()` so concurrent clients batch safely.

### Endpoints

**Health**

```bash
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
```

**Completions (non-streaming, default)**

Omit `stream` or set `"stream": false` to receive one JSON response when generation finishes. The response follows the OpenAI **text_completion** shape:

```bash
curl -s -X POST http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Every effort moves you", "max_new_tokens": 20}' \
  | python3 -m json.tool
```

Example response:

```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1700000000,
  "model": "gpt",
  "choices": [
    {
      "text": " ...generated text only...",
      "index": 0,
      "logprobs": null,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 20,
    "total_tokens": 24
  }
}
```

**Completions (streaming SSE, OpenAI-compatible)**

Set `"stream": true` to receive Server-Sent Events in OpenAI streaming format:

```bash
curl -N -X POST http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Every effort moves you", "max_new_tokens": 20, "stream": true}'
```

Example events:

```text
data: {"id":"cmpl-abc123","object":"text_completion","created":1700000000,"model":"gpt","choices":[{"text":" hello","index":0,"logprobs":null,"finish_reason":null}]}

data: {"id":"cmpl-abc123","object":"text_completion","created":1700000000,"model":"gpt","choices":[{"text":" world","index":0,"logprobs":null,"finish_reason":"length"}]}

data: [DONE]
```

Optional request field `"model"` overrides the model name echoed in responses (defaults to the loaded backend).

### Server options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `hf-gpt` | Registered backend name (`hf-gpt` or `gpt`) |
| `--hf-model` | `gpt2` | Hugging Face model id (for `hf-gpt` backend) |
| `--context-length` | `256` | Max context (truncates HF position embeddings) |
| `--checkpoint` | `src/model/checkpoints/gpt_model_checkpoint.pt` | Weights path (custom `gpt` backend only) |
| `--max-concurrent` | `2` | Max concurrent requests (cache slots) |
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8000` | Port |

## Chunked prefill

Long prompts are no longer processed in a single prefill forward. Each request caches its prompt in chunks (default **128 tokens** per step), staying in `PREFILL` until `prefill_offset` reaches the prompt length. That lets other requests decode between chunks.

```text
Request A (long prompt):  PREFILL chunk → PREFILL chunk → DECODE…
Request B (short prompt):           PREFILL → DECODE…  (interleaved with A)
```

### Configuration

Set chunk size when constructing the engine:

```python
engine = Engine(
    model,
    max_concurrent_requests=2,
    device=device,
    prefill_chunk_size=512,
    max_tokens_per_step=1024,
)
```

`add_request()` copies `Engine.prefill_chunk_size` onto each `InferenceRequest`. Short prompts still complete in one step when `len(prompt) <= prefill_chunk_size`.

### Token budget (`max_tokens_per_step`)

Each `engine.step()` caps total tokens processed in that step (default **2048**). The scheduler uses **decode-first** ordering:

1. Add decode requests (1 token each) until budget is exhausted
2. Add prefill chunks (`min(prefill_chunk_size, prompt remaining)`) for remaining budget

Requests that do not fit are deferred to the next step. This smooths latency under load when many prefills and decodes are active.

### Lifecycle

| Step | What happens |
|------|----------------|
| `model_runner.prefill` | Processes `prompt[offset : offset + chunk_size]`, advances `prefill_offset` |
| Intermediate chunk | Returns `None`; request stays in `PREFILL` |
| Final chunk | Returns first sampled token; engine calls `mark_prefill_done` |
| `scheduler.mark_prefill_done` | Requires `prefill_complete`; transitions to `DECODE` |

### Tests

Chunking is covered in `tests/test_model_runner.py` and `tests/test_scheduler.py`.

## Paged KV cache

The engine uses **PagedKVCache** by default (`use_paged_kv_cache=True`). Physical K/V memory is split into fixed-size **blocks** managed by a shared `BlockAllocator`. Each sequence has a `BlockTable` that maps logical token positions to physical block IDs.

```text
BlockAllocator (shared pool)
  ├── Block 0  →  K/V tensors at index 0, all layers
  ├── Block 1  →  K/V tensors at index 1, all layers
  └── ...

Slot 0 BlockTable:  [physical 3] [physical 7]     →  tokens 0–7, 8–15
Slot 1 BlockTable:  [physical 1]                  →  tokens 0–7
```

Blocks use **reference counting** (`retain` / `release`) so the same physical block can be shared safely (e.g. by the prefix cache and multiple sequences).

Disable paging to use the dense per-slot `KVCache`:

```python
engine = Engine(model, max_concurrent_requests=2, device=device, use_paged_kv_cache=False)
```

`ModelConfig.block_size` (default **16**) controls how many tokens fit in each block. GPT backends read `block_size` from the model config dict when present.

## Prefix caching

When many requests share the same prompt prefix (system prompts, RAG context, few-shot examples), prefix caching skips redundant prefill work by reusing already-computed K/V blocks.

```text
Request 1: [system prompt 16 tok] + [user A]
  → full prefill → register_prefix() stores blocks in PrefixCache

Request 2: [same system prompt 16 tok] + [user B]
  → try_load_prefix() hits 16 tokens → prefill only [user B]
```

### How it works

1. **Block-chained hashing** — each full block of `block_size` tokens is keyed by `(parent_hash, block_tokens)`. Lookup walks the chain and returns the longest cached prefix.
2. **On prefill complete** — `PagedKVCache.register_prefix()` inserts the prompt's full blocks into the cache and retains them.
3. **On new request** — `ModelRunner` calls `try_load_prefix()` before resetting the slot. On a hit, `prefill_offset` starts after the cached prefix.
4. **On request finish** — the slot's block table is cleared; cached blocks stay alive via the prefix cache's references.

Only **full blocks** are cached (`len(prompt) // block_size`). A 20-token prompt with `block_size=16` caches one block (16 tokens).

### Configuration

```python
engine = Engine(
    model,
    max_concurrent_requests=4,
    device=device,
    use_paged_kv_cache=True,   # required for prefix caching
    use_prefix_cache=True,     # default
)
```

Prefix cache size defaults to **1024 entries** (LRU eviction). Disable with `use_prefix_cache=False`.

### Example: shared system prompt

```python
system = tokenizer.encode("You are a helpful assistant. " * 10)
q1 = system + tokenizer.encode("What is Python?")
q2 = system + tokenizer.encode("What is Rust?")

engine.generate(q1, max_new_tokens=20)
engine.generate(q2, max_new_tokens=20)
# Second request skips prefill for the shared system tokens.
```

### Tests

Prefix caching is covered in `tests/test_prefix_cache.py`, `tests/test_paged_kv_cache.py`, and `tests/test_block_allocator.py`.

## Observability dashboard

The inference server ships with a built-in observability dashboard. Start the server as usual, then open:

**Dashboard:** [http://127.0.0.1:8000/observability](http://127.0.0.1:8000/observability)

**Metrics API:**

```bash
curl -s http://127.0.0.1:8000/observability/api/metrics | python3 -m json.tool
```

### Dashboard sections

| Section | Metrics |
|---------|---------|
| **Request health** | requests/sec, active, queued, completed, error rate, timeout rate |
| **Latency** | TTFT, total latency, prefill/decode step latency, inter-token latency (P50/P95/P99) |
| **Throughput** | tokens/sec, input/output tokens/sec, tokens/request, prefill tokens/step, prefix cache hits & tokens saved |
| **GPU / runtime** | GPU util & memory, KV cache memory & utilization, cache type, block pool, prefix cache hit rate |
| **Optimization history** | baseline vs current latency/throughput, cost improvement, attempted/promoted/rolled back |

### Standalone dashboard (optional)

Poll metrics from a running inference server on a separate port:

```bash
PYTHONPATH=src:src/model python3 -m observability.dashboard.server \
  --port 9090 \
  --metrics-url http://127.0.0.1:8000/observability/api/metrics
```

### Record optimization events (Python)

```python
from observability import Observability

obs = Observability(model_name="gpt", runtime="custom")
obs.attach(engine)  # auto-records promotions for paged-kv-cache, prefix-cache, chunked-prefill, etc.

obs.optimization.record_attempt("continuous-batching", details="enabled scheduler v2")
obs.optimization.record_promotion("continuous-batching")
obs.optimization.record_rollback("fp8-kv-cache", details="accuracy regression")
```

## Static batching CLI

The reference GPT model also supports static batched generation (no continuous scheduler):

```bash
PYTHONPATH=src:src/model python3 -m generate \
  --prompt "Every effort moves you" \
  --prompt "The cat sat on the mat" \
  --max-new-tokens 20
```

## Use the engine in Python

```python
import torch
from pathlib import Path

from inference.backends.registry import load_backend
from inference.engine import Engine
from generate import get_device

device = get_device()
checkpoint = Path("src/model/checkpoints/gpt_model_checkpoint.pt")

model, tokenizer = load_backend(
    "gpt",
    checkpoint if checkpoint.exists() else None,
    device,
)

engine = Engine(
    model,
    max_concurrent_requests=2,
    device=device,
    use_paged_kv_cache=True,
    use_prefix_cache=True,
)

# Single blocking request
token_ids = tokenizer.encode("Every effort moves you")
result = engine.generate(token_ids, max_new_tokens=20)
print(tokenizer.decode(result.prompt_token_ids + result.output_token_ids))

# Streaming request (token-by-token)
for token_id in engine.generate_stream(token_ids, max_new_tokens=20):
    print(tokenizer.decode([token_id]), end="", flush=True)
print()

# Or queue multiple requests and step manually
engine.add_request(tokenizer.encode("Hello"), max_new_tokens=10)
engine.add_request(tokenizer.encode("The quick brown fox"), max_new_tokens=10)
engine.run_until_done()

for req in engine.get_completed().values():
    print(tokenizer.decode(req.prompt_token_ids + req.output_token_ids))
```

## Architecture

```text
Client
  → FastAPI server (server.py)
    → EngineWorker (background step loop)
      → Engine (scheduler + KV cache lifecycle)
        → ModelRunner (prefill / decode forwards, prefix cache lookup)
          → InferenceModel backend (e.g. GPT)
            → Attention reads/writes PagedKVCache or KVCache
```

| Component | Role |
|-----------|------|
| **EngineWorker** | Single thread owns ``engine.step()``; HTTP handlers submit via ``generate`` / ``generate_stream`` |
| **Scheduler** | Queue, batch slots, token budget per step (decode-first), enforces `prefill_complete` |
| **Engine** | Owns cache, calls scheduler + model runner each step; configures chunking and prefix cache |
| **ModelRunner** | Batched prefill chunks + decode forwards; `try_load_prefix` on slot prep |
| **KVCache** | Dense per-slot storage (optional via `use_paged_kv_cache=False`) |
| **PagedKVCache** | Block-pooled K/V storage with `BlockAllocator` + `BlockTable` per slot (engine default) |
| **PrefixCache** | Block-chained hash map from token prefixes to shared physical blocks |
| **BlockAllocator** | Physical block pool with reference counting for shared blocks |
| **Backend** | Weights, tokenizer, cache-aware forward |

## Adding a new model backend

1. Implement `InferenceModel` — expose `.config` and a cache-aware `__call__` returning logits.
2. Implement `Tokenizer` — `encode`, `decode`, `pad_token_id`.
3. Add `load_my_backend()` in `src/inference/backends/my_model.py`.
4. Register it in `src/inference/backends/registry.py`:

```python
BACKENDS = {
    "gpt": load_gpt_backend,
    "my_model": load_my_backend,
}
```

5. Run: `python3 -m inference.server --model my_model`

See `src/inference/model_interface.py` and `src/inference/backends/gpt.py` for the reference adapter.

## Auto-tuning

The embedded auto-tuner observes metrics, classifies workload, proposes config changes, and promotes or rolls back after an evaluation window.

**Start with auto-tune enabled:**

```bash
PYTHONPATH=src:src/model python3 -m inference.server \
  --auto-tune \
  --tuning-goal balanced \
  --tuning-interval 30 \
  --tuning-evaluation 60
```

**Admin API** (also available when observability is enabled):

```bash
curl http://127.0.0.1:8000/v1/admin/tuning
curl -X POST http://127.0.0.1:8000/v1/admin/tuning \
  -H 'Content-Type: application/json' \
  -d '{"enabled": true, "goal": "latency"}'
```

The observability dashboard includes an **Auto-Tuning** panel (`/observability`) showing goal, pending attempt, last action, and live engine config.

## Benchmark workloads

Generate synthetic traffic against a running server (useful before enabling auto-tune):

```bash
# Terminal 1 — server
PYTHONPATH=src:src/model python3 -m inference.server --max-concurrent 4

# Terminal 2 — benchmark
PYTHONPATH=src:src/model python3 -m bench --profile chat --duration 30 --concurrency 4
PYTHONPATH=src:src/model python3 -m bench --profile rag --duration 30
PYTHONPATH=src:src/model python3 -m bench --profile batch --duration 30
```

Profiles:

| Profile | Simulates |
|---------|-----------|
| `chat` | Short prompts, moderate concurrency |
| `rag` | Long shared prefix + short question suffixes (prefix-cache friendly) |
| `batch` | Many unique prompts |

## Comparing nifre vs vLLM (same weights)

Both engines should use **Hugging Face `gpt2`** (124M). nifre loads it natively via the `hf-gpt` backend (same `transformers` model vLLM uses):

**nifre** (port 8000):

```bash
PYTHONPATH=src:src/model python3 -m inference.server \
  --model hf-gpt \
  --hf-model gpt2 \
  --context-length 256 \
  --max-concurrent 4
```

**vLLM** (port 8001) — same model id and context cap:

```bash
vllm serve gpt2 --host 127.0.0.1 --port 8001 --max-model-len 256
```

Use the same prompts. Greedy decode on a fixed prompt should match between engines before benchmarking throughput.

| Setting | nifre | vLLM |
|---------|-------|------|
| Weights | `gpt2` via `--model hf-gpt` | `gpt2` |
| Context | `--context-length 256` | `--max-model-len 256` |
| Tokenizer | HF GPT-2 (`transformers`) | HF GPT-2 tokenizer |

### Custom GPT backend + weight import (optional)

If you want to benchmark nifre's **custom** PyTorch GPT implementation with the same HF weights, import once:

```bash
PYTHONPATH=src:src/model python3 -m inference.backends.import_hf_gpt2 \
  --model gpt2 \
  --context-length 256 \
  --output src/model/checkpoints/gpt2_hf_checkpoint.pt
```

Then run with `--model gpt --checkpoint src/model/checkpoints/gpt2_hf_checkpoint.pt`.

## What is not included yet

- Global block pool oversubscription (admission control when pool is full)
- Fused paged-attention GPU kernels
- Production auth, rate limits, or multi-GPU serving

These are natural next steps after the core engine is solid.
