"""Minimal HTTP server exposing the continuous-batching engine."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import tiktoken

from inference.engine import Engine
from model.generate import get_device, load_model
from model.gpt_model import GPT_CONFIG_124M, GptModel

DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent.parent / "model" / "checkpoints" / "gpt_model_checkpoint.pt"
)


class InferenceHandler(BaseHTTPRequestHandler):
    engine: Engine = None
    tokenizer = None

    def do_GET(self):
        if self.path == "/health":
            self._json_response(200, {"status": "ok"})
            return
        self._json_response(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/completions":
            self._json_response(404, {"error": "not found"})
            return

        body = self._read_json()
        prompt = body.get("prompt")
        max_new_tokens = int(body.get("max_new_tokens", 20))

        if not prompt:
            self._json_response(400, {"error": "prompt is required"})
            return

        token_ids = self.tokenizer.encode(prompt)
        request = self.engine.generate(token_ids, max_new_tokens=max_new_tokens)
        text = self.tokenizer.decode(request.prompt_token_ids + request.output_token_ids)

        self._json_response(
            200,
            {
                "request_id": request.request_id,
                "prompt": prompt,
                "text": text,
                "output_token_ids": request.output_token_ids,
            },
        )

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _json_response(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(f"[server] {self.address_string()} {format % args}")


def create_engine(checkpoint: Path, max_concurrent: int) -> Engine:
    device = get_device()
    if checkpoint.exists():
        print(f"Loading checkpoint: {checkpoint}")
        model = load_model(checkpoint, device)
    else:
        print(f"No checkpoint at {checkpoint} — using random weights")
        model = GptModel(GPT_CONFIG_124M).to(device)
        model.eval()
    return Engine(model, max_concurrent_requests=max_concurrent, device=device)


def main():
    parser = argparse.ArgumentParser(description="Inference engine HTTP server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-concurrent", type=int, default=2)
    args = parser.parse_args()

    InferenceHandler.engine = create_engine(args.checkpoint, args.max_concurrent)
    InferenceHandler.tokenizer = tiktoken.get_encoding("gpt2")

    server = HTTPServer((args.host, args.port), InferenceHandler)
    print(f"Serving on http://{args.host}:{args.port}")
    print("POST /v1/completions  {\"prompt\": \"...\", \"max_new_tokens\": 20}")
    print("GET  /health")
    server.serve_forever()


if __name__ == "__main__":
    main()
