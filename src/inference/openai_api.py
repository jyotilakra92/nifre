"""OpenAI-compatible response shapes for the completions API."""

from __future__ import annotations

import json
import time
import uuid
from typing import Iterator, List, Optional

from inference.engine_worker import EngineWorker


def new_completion_id() -> str:
    return f"cmpl-{uuid.uuid4().hex}"


def completion_chunk_sse(
    *,
    completion_id: str,
    model: str,
    text: str,
    finish_reason: Optional[str] = None,
    created: Optional[int] = None,
) -> str:
    payload = {
        "id": completion_id,
        "object": "text_completion",
        "created": created or int(time.time()),
        "model": model,
        "choices": [
            {
                "text": text,
                "index": 0,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\n"


def stream_openai_completion_events(
    worker: EngineWorker,
    tokenizer,
    prompt_token_ids: List[int],
    max_new_tokens: int,
    model: str,
) -> Iterator[str]:
    """Yield SSE events in OpenAI ``text_completion`` streaming format."""
    completion_id = new_completion_id()
    created = int(time.time())

    token_stream = worker.generate_stream(prompt_token_ids, max_new_tokens)
    pending_token_id: Optional[int] = None

    for token_id in token_stream:
        if pending_token_id is not None:
            text = tokenizer.decode([pending_token_id])
            yield completion_chunk_sse(
                completion_id=completion_id,
                model=model,
                text=text,
                finish_reason=None,
                created=created,
            )
        pending_token_id = token_id

    if pending_token_id is not None:
        text = tokenizer.decode([pending_token_id])
        yield completion_chunk_sse(
            completion_id=completion_id,
            model=model,
            text=text,
            finish_reason="length",
            created=created,
        )

    yield "data: [DONE]\n\n"


def build_completion_response(
    *,
    completion_id: str,
    model: str,
    completion_text: str,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str = "length",
    created: Optional[int] = None,
) -> dict:
    return {
        "id": completion_id,
        "object": "text_completion",
        "created": created or int(time.time()),
        "model": model,
        "choices": [
            {
                "text": completion_text,
                "index": 0,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
