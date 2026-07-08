import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inference.openai_api import build_completion_response, completion_chunk_sse


def test_completion_chunk_sse_shape():
    event = completion_chunk_sse(
        completion_id="cmpl-test",
        model="gpt-test",
        text=" hi",
        finish_reason=None,
        created=1700000000,
    )
    assert event.startswith("data: ")
    assert event.endswith("\n\n")

    payload = json.loads(event.removeprefix("data: ").strip())
    assert payload["id"] == "cmpl-test"
    assert payload["object"] == "text_completion"
    assert payload["created"] == 1700000000
    assert payload["model"] == "gpt-test"
    assert len(payload["choices"]) == 1
    assert payload["choices"][0]["text"] == " hi"
    assert payload["choices"][0]["index"] == 0
    assert payload["choices"][0]["finish_reason"] is None


def test_build_completion_response_shape():
    payload = build_completion_response(
        completion_id="cmpl-abc",
        model="gpt-test",
        completion_text="hello",
        prompt_tokens=3,
        completion_tokens=2,
        created=1700000001,
    )
    assert payload["object"] == "text_completion"
    assert payload["choices"][0]["text"] == "hello"
    assert payload["choices"][0]["finish_reason"] == "length"
    assert payload["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
