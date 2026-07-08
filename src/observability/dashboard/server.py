"""Observability dashboard server and route registration."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from observability import Observability

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_INDEX_HTML = (_STATIC_DIR / "index.html").read_text()


def create_dashboard_app(metrics_provider: Callable[[], dict]) -> FastAPI:
    app = FastAPI(title="Inference Observability Dashboard")

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return _INDEX_HTML

    @app.get("/api/metrics")
    def metrics():
        return JSONResponse(metrics_provider())

    return app


def register_observability_routes(app: FastAPI) -> None:
    """Mount dashboard and metrics API on an existing FastAPI app."""

    @app.get("/observability", response_class=HTMLResponse, include_in_schema=False)
    def observability_dashboard():
        html = _INDEX_HTML.replace('fetch("/api/metrics")', 'fetch("/observability/api/metrics")')
        html = html.replace("<body>", '<body data-metrics-path="/observability/api/metrics" data-tuning-path="/observability/api/tuning">')
        return html

    @app.get("/observability/api/metrics", include_in_schema=False)
    def observability_metrics(request: Request):
        obs = getattr(request.app.state, "observability", None)
        if obs is None:
            return JSONResponse({"error": "observability not enabled"}, status_code=503)
        return obs.snapshot()


def _remote_metrics_provider(url: str) -> Callable[[], dict]:
    def provider() -> dict:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                return json.loads(response.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return {"timestamp": time.time(), "error": "unable to reach inference server"}

    return provider


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone observability dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument(
        "--metrics-url",
        default="http://127.0.0.1:8000/observability/api/metrics",
        help="Inference server metrics endpoint to poll",
    )
    args = parser.parse_args()

    provider = _remote_metrics_provider(args.metrics_url)
    app = create_dashboard_app(provider)
    print(f"Dashboard: http://{args.host}:{args.port}")
    print(f"Polling metrics from: {args.metrics_url}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
