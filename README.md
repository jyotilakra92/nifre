# Inference Engine

A small, educational LLM inference engine with KV-cache, static batching, continuous batching, and a model-agnostic backend interface. Includes a FastAPI server and a reference GPT backend.

## Features

- **KV cache** — prefill + decode without recomputing past attention
- **Static batching** — run multiple prompts in one forward pass (fixed batch)
- **Continuous batching** — requests join and leave between decode steps
- **Chunked prefill** — long prompts are cached in fixed-size chunks so decode can interleave
- **Model-agnostic API** — plug in backends via `InferenceModel` + `Tokenizer`
- **FastAPI server** — HTTP completions endpoint with OpenAPI docs
- **Observability dashboard** — request health, latency, throughput, GPU/runtime, optimization history

## Project layout

```text
nifre/
├── src/
│   ├── inference/          # Engine, scheduler, server, backends
│   │   ├── engine.py
│   │   ├── scheduler.py
│   │   ├── model_runner.py
│   │   ├── server.py
│   │   └── backends/       # Model adapters (gpt today)
│   └── observability/      # Metrics, dashboard, optimization tracking
│       ├── metrics_store.py
│       ├── collector.py
│       ├── runtime_probe.py
│       ├── optimization.py
│       └── dashboard/      # FastAPI dashboard UI
│   └── model/              # Reference GPT implementation
│       ├── gpt_model.py
│       ├── attention.py
│       ├── kv_cache.py
│       └── generate.py     # Static-batched CLI
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

Start the server:

```bash
PYTHONPATH=src:src/model python3 -m inference.server \
  --model gpt \
  --port 8000 \
  --max-concurrent 2
```

Or with uvicorn:

```bash
PYTHONPATH=src:src/model uvicorn inference.server:app --host 127.0.0.1 --port 8000
```

Interactive API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Endpoints

**Health**

```bash
curl -s http://127.0.0.1:8000/health | python3 -m json.tool
```

**Completions**

```bash
curl -s -X POST http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Every effort moves you", "max_new_tokens": 20}' \
  | python3 -m json.tool
```

Example response:

```json
{
  "request_id": "9810ebc3",
  "prompt": "Every effort moves you",
  "text": "Every effort moves you ...",
  "output_token_ids": [20625, 10325, ...],
  "model": "gpt"
}
```

### Server options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `gpt` | Registered backend name |
| `--checkpoint` | `src/model/checkpoints/gpt_model_checkpoint.pt` | Weights path |
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
)
```

`add_request()` copies `Engine.prefill_chunk_size` onto each `InferenceRequest`. Short prompts still complete in one step when `len(prompt) <= prefill_chunk_size`.

### Lifecycle

| Step | What happens |
|------|----------------|
| `model_runner.prefill` | Processes `prompt[offset : offset + chunk_size]`, advances `prefill_offset` |
| Intermediate chunk | Returns `None`; request stays in `PREFILL` |
| Final chunk | Returns first sampled token; engine calls `mark_prefill_done` |
| `scheduler.mark_prefill_done` | Requires `prefill_complete`; transitions to `DECODE` |

### Tests

Chunking is covered in `tests/test_model_runner.py` and `tests/test_scheduler.py`:

```bash
PYTHONPATH=src:src/model python3 -m tests
```

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
| **Throughput** | tokens/sec, input/output tokens/sec, tokens/request, batch size, decode iterations/sec |
| **GPU / runtime** | GPU util & memory, KV cache memory & utilization, runtime, model, precision |
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
obs.attach(engine)

obs.optimization.record_attempt("continuous-batching", details="enabled scheduler v2")
obs.optimization.record_promotion("continuous-batching")
obs.optimization.record_rollback("fp8-kv-cache", details="accuracy regression")
```

## Static batching CLI

The reference GPT model also supports static batched generation (no continuous scheduler):

```bash
PYTHONPATH=src:src/model python3 -m model.generate \
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
from model.generate import get_device

device = get_device()
checkpoint = Path("src/model/checkpoints/gpt_model_checkpoint.pt")

model, tokenizer = load_backend(
    "gpt",
    checkpoint if checkpoint.exists() else None,
    device,
)

engine = Engine(model, max_concurrent_requests=2, device=device)

# Single blocking request
token_ids = tokenizer.encode("Every effort moves you")
result = engine.generate(token_ids, max_new_tokens=20)
print(tokenizer.decode(result.prompt_token_ids + result.output_token_ids))

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
    → Engine (scheduler + KV cache lifecycle)
      → ModelRunner (prefill / decode forwards)
        → InferenceModel backend (e.g. GPT)
          → Attention reads/writes KVCache
```

| Component | Role |
|-----------|------|
| **Scheduler** | Queue, batch slots, prefill vs decode groups; enforces `prefill_complete` before decode |
| **Engine** | Owns cache, calls scheduler + model runner each step; configures `prefill_chunk_size` |
| **ModelRunner** | Batched prefill chunks + decode forwards + greedy sampling |
| **KVCache** | Per-slot K/V storage (engine allocates, model uses) |
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

## What is not included yet

- Scheduler token budget (cap tokens per step)
- PagedAttention
- Token streaming (SSE)
- OpenAI-compatible API shape
- Production auth, rate limits, or multi-GPU serving

These are natural next steps after the core engine is solid.
