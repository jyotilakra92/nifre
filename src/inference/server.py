"""FastAPI server exposing the continuous-batching inference engine."""

import argparse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from inference.backends.registry import list_backends, load_backend
from inference.engine import Engine
from model.generate import get_device
from observability import Observability
from observability.dashboard.server import register_observability_routes

DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent.parent / "model" / "checkpoints" / "gpt_model_checkpoint.pt"
)


class CompletionRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=20, ge=1)


class CompletionResponse(BaseModel):
    request_id: str
    prompt: str
    text: str
    output_token_ids: List[int]
    model: str


def create_engine(
    model_backend: str,
    checkpoint: Path,
    max_concurrent: int,
    observability: Optional[Observability] = None,
) -> Engine:
    device = get_device()
    checkpoint_path = checkpoint if checkpoint.exists() else None
    if checkpoint_path is None:
        print(f"No checkpoint at {checkpoint} — using random weights")
    else:
        print(f"Loading checkpoint: {checkpoint}")

    model, _tokenizer = load_backend(model_backend, checkpoint_path, device)
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
    model_backend: str = "gpt",
    checkpoint: Path = DEFAULT_CHECKPOINT,
    max_concurrent: int = 2,
    engine: Optional[Engine] = None,
    tokenizer=None,
    backend_name: Optional[str] = None,
    observability: Optional[Observability] = None,
    enable_observability: bool = True,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
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
            checkpoint_path = checkpoint if checkpoint.exists() else None
            obs = None
            if enable_observability:
                obs = observability or Observability(model_name=model_backend, runtime="custom")
            model, app.state.tokenizer = load_backend(
                model_backend, checkpoint_path, device
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
        yield

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
        return payload

    @app.post("/v1/completions", response_model=CompletionResponse)
    def completions(body: CompletionRequest, request: Request):
        engine = request.app.state.engine
        tokenizer = request.app.state.tokenizer

        token_ids = tokenizer.encode(body.prompt)
        result = engine.generate(token_ids, max_new_tokens=body.max_new_tokens)
        text = tokenizer.decode(result.prompt_token_ids + result.output_token_ids)

        return CompletionResponse(
            request_id=result.request_id,
            prompt=body.prompt,
            text=text,
            output_token_ids=result.output_token_ids,
            model=request.app.state.model_backend,
        )

    if enable_observability:
        register_observability_routes(app)

    return app


app = create_app()


def main():
    parser = argparse.ArgumentParser(description="Inference engine FastAPI server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model",
        default="gpt",
        choices=list_backends(),
        help="Registered model backend to load",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-concurrent", type=int, default=2)
    args = parser.parse_args()

    server_app = create_app(args.model, args.checkpoint, args.max_concurrent)
    print(f"Serving on http://{args.host}:{args.port}")
    print(f"Model backend: {args.model}")
    print("Docs:  http://{host}:{port}/docs".format(host=args.host, port=args.port))
    print('POST /v1/completions  {"prompt": "...", "max_new_tokens": 20}')
    print("GET  /health")
    print("GET  /observability  (metrics dashboard)")
    uvicorn.run(server_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
