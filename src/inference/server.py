"""FastAPI server exposing the continuous-batching inference engine."""

import argparse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse, StreamingResponse

from inference.backends.registry import list_backends, load_backend
from inference.engine import Engine
from inference.engine_worker import EngineWorker
from inference.openai_api import (
    build_completion_response,
    new_completion_id,
    stream_openai_completion_events,
)
from generate import get_device
from observability import Observability
from observability.dashboard.server import register_observability_routes
from autotune.admin import register_tuning_routes

DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent.parent / "model" / "checkpoints" / "gpt_model_checkpoint.pt"
)


class CompletionRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=20, ge=1)
    stream: bool = Field(default=False)
    model: Optional[str] = Field(default=None)


def _completions_blocking(
    worker: EngineWorker,
    tokenizer,
    prompt_token_ids: List[int],
    max_new_tokens: int,
    model_backend: str,
    requested_model: Optional[str],
) -> JSONResponse:
    result = worker.generate(prompt_token_ids, max_new_tokens=max_new_tokens)
    completion_text = tokenizer.decode(result.output_token_ids)
    payload = build_completion_response(
        completion_id=new_completion_id(),
        model=requested_model or model_backend,
        completion_text=completion_text,
        prompt_tokens=len(result.prompt_token_ids),
        completion_tokens=len(result.output_token_ids),
    )
    return JSONResponse(payload)


def create_engine(
    model_backend: str,
    checkpoint: Path,
    max_concurrent: int,
    observability: Optional[Observability] = None,
    *,
    hf_model: str = "gpt2",
    context_length: int = 256,
) -> Engine:
    device = get_device()
    loader_kwargs = {}
    if model_backend == "hf-gpt":
        loader_kwargs = {"hf_model": hf_model, "context_length": context_length}
    else:
        checkpoint_path = checkpoint if checkpoint.exists() else None
        if checkpoint_path is None:
            print(f"No checkpoint at {checkpoint} — using random weights")
        else:
            print(f"Loading checkpoint: {checkpoint}")

    if model_backend == "hf-gpt":
        print(f"Loading Hugging Face model: {hf_model} (context_length={context_length})")
        model, _tokenizer = load_backend(model_backend, None, device, **loader_kwargs)
    else:
        checkpoint_path = checkpoint if checkpoint.exists() else None
        model, _tokenizer = load_backend(model_backend, checkpoint_path, device, **loader_kwargs)
    metrics = observability.collector if observability else None
    engine = Engine(
        model,
        max_concurrent_requests=max_concurrent,
        device=device,
        metrics_collector=metrics,
    )
    if observability:
        observability.attach(engine)
    return engine


def create_app(
    model_backend: str = "hf-gpt",
    checkpoint: Path = DEFAULT_CHECKPOINT,
    max_concurrent: int = 2,
    engine: Optional[Engine] = None,
    tokenizer=None,
    backend_name: Optional[str] = None,
    observability: Optional[Observability] = None,
    enable_observability: bool = True,
    auto_tune: bool = False,
    tuning_goal: str = "balanced",
    tuning_interval_sec: float = 30.0,
    tuning_evaluation_sec: float = 60.0,
    hf_model: str = "gpt2",
    context_length: int = 256,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        worker: Optional[EngineWorker] = None
        tuning_controller = None
        if engine is not None:
            app.state.engine = engine
            app.state.tokenizer = tokenizer
            app.state.model_backend = backend_name or model_backend
            if enable_observability:
                obs = observability or Observability(
                    model_name=backend_name or model_backend,
                    runtime="custom",
                )
                obs.attach(app.state.engine)
                app.state.observability = obs
            else:
                app.state.observability = observability
        else:
            device = get_device()
            runtime = "huggingface" if model_backend == "hf-gpt" else "custom"
            obs = None
            if enable_observability:
                obs = observability or Observability(model_name=model_backend, runtime=runtime)
            loader_kwargs = {}
            if model_backend == "hf-gpt":
                loader_kwargs = {"hf_model": hf_model, "context_length": context_length}
                print(f"Loading Hugging Face model: {hf_model} (context_length={context_length})")
                model, app.state.tokenizer = load_backend(
                    model_backend, None, device, **loader_kwargs
                )
            else:
                checkpoint_path = checkpoint if checkpoint.exists() else None
                if checkpoint_path is None:
                    print(f"No checkpoint at {checkpoint} — using random weights")
                else:
                    print(f"Loading checkpoint: {checkpoint}")
                model, app.state.tokenizer = load_backend(
                    model_backend, checkpoint_path, device, **loader_kwargs
                )
            app.state.engine = Engine(
                model,
                max_concurrent_requests=max_concurrent,
                device=device,
                metrics_collector=obs.collector if obs else None,
            )
            if obs:
                obs.attach(app.state.engine)
            app.state.observability = obs
            app.state.model_backend = model_backend

        worker = EngineWorker(app.state.engine)
        worker.start()
        app.state.worker = worker

        if app.state.observability is not None:
            from autotune import ControllerConfig, TuningController

            tuning_controller = TuningController(
                engine=app.state.engine,
                observability=app.state.observability,
                goal=tuning_goal,
                worker=worker,
                config=ControllerConfig(
                    interval_sec=tuning_interval_sec,
                    evaluation_sec=tuning_evaluation_sec,
                ),
            )
            app.state.tuning_controller = tuning_controller
            if auto_tune:
                tuning_controller.start()
        else:
            app.state.tuning_controller = None

        yield
        if tuning_controller is not None:
            tuning_controller.stop()
        worker.stop()

    app = FastAPI(
        title="Inference Engine",
        description="Model-agnostic continuous-batching LLM inference server",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(request: Request):
        payload = {
            "status": "ok",
            "model": request.app.state.model_backend,
            "backends": list_backends(),
        }
        if getattr(request.app.state, "observability", None):
            payload["observability"] = "/observability"
        if getattr(request.app.state, "tuning_controller", None) is not None:
            payload["auto_tune"] = "/v1/admin/tuning"
        return payload

    @app.post("/v1/completions")
    def completions(body: CompletionRequest, request: Request):
        """Return full JSON (default) or SSE token stream when ``stream=true``."""
        worker = request.app.state.worker
        tokenizer = request.app.state.tokenizer
        token_ids = tokenizer.encode(body.prompt)

        if body.stream:
            return StreamingResponse(
                stream_openai_completion_events(
                    worker,
                    tokenizer,
                    token_ids,
                    body.max_new_tokens,
                    body.model or request.app.state.model_backend,
                ),
                media_type="text/event-stream",
            )

        return _completions_blocking(
            worker,
            tokenizer,
            token_ids,
            body.max_new_tokens,
            request.app.state.model_backend,
            body.model,
        )

    if enable_observability:
        register_observability_routes(app)
        register_tuning_routes(app)

    return app


app = create_app()


def main():
    parser = argparse.ArgumentParser(description="Inference engine FastAPI server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model",
        default="hf-gpt",
        choices=list_backends(),
        help="Registered model backend to load (default: hf-gpt = Hugging Face GPT-2)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint path for the custom gpt backend only",
    )
    parser.add_argument(
        "--hf-model",
        default="gpt2",
        help="Hugging Face model id when using --model hf-gpt",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=256,
        help="Max context length (truncates HF position embeddings)",
    )
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument(
        "--auto-tune",
        action="store_true",
        help="Enable embedded auto-tuning controller",
    )
    parser.add_argument(
        "--tuning-goal",
        default="balanced",
        choices=["latency", "throughput", "balanced"],
        help="Auto-tuning optimization goal",
    )
    parser.add_argument(
        "--tuning-interval",
        type=float,
        default=30.0,
        help="Seconds between auto-tune observation cycles",
    )
    parser.add_argument(
        "--tuning-evaluation",
        type=float,
        default=60.0,
        help="Seconds to evaluate a config change before promote/rollback",
    )
    args = parser.parse_args()

    server_app = create_app(
        args.model,
        args.checkpoint,
        args.max_concurrent,
        auto_tune=args.auto_tune,
        tuning_goal=args.tuning_goal,
        tuning_interval_sec=args.tuning_interval,
        tuning_evaluation_sec=args.tuning_evaluation,
        hf_model=args.hf_model,
        context_length=args.context_length,
    )
    print(f"Serving on http://{args.host}:{args.port}")
    print(f"Model backend: {args.model}")
    if args.model == "hf-gpt":
        print(f"HF model: {args.hf_model}  context_length={args.context_length}")
    print("Docs:  http://{host}:{port}/docs".format(host=args.host, port=args.port))
    print('POST /v1/completions  {"prompt": "...", "max_new_tokens": 20}  # blocking JSON')
    print('POST /v1/completions  {"prompt": "...", "max_new_tokens": 20, "stream": true}  # SSE')
    print("GET  /health")
    print("GET  /observability  (metrics dashboard)")
    print("GET  /v1/admin/tuning  (auto-tune status)")
    if args.auto_tune:
        print(f"Auto-tune: enabled  goal={args.tuning_goal}")
    uvicorn.run(server_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
