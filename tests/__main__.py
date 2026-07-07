import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
MODEL = SRC / "model"
for path in (SRC, MODEL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tests.test_attention import test_attention_cache_smoke
<<<<<<< HEAD
from tests.test_engine import (
    test_engine_smoke,
    test_stream_request_matches_generate,
    test_stream_request_unregisters_callback,
    test_stream_request_with_chunked_prefill,
=======
from tests.test_engine_worker import (
    test_worker_concurrent_generates,
    test_worker_concurrent_stream_and_generate,
    test_worker_generate_matches_engine,
    test_worker_generate_stream_matches_generate,
)
from tests.test_engine import (
    test_engine_smoke,
    test_generate_stream_matches_generate,
    test_generate_stream_unregisters_callback,
    test_generate_stream_with_chunked_prefill,
>>>>>>> f172705 (Add support for token streaming)
    test_token_callback_emits_all_generated_tokens,
)
from tests.test_generate import test_model_runner_via_engine, test_static_batch_matches_single
from tests.test_kv_cache import test_kv_cache_smoke
from tests.test_backends import test_load_gpt_backend
from tests.test_scheduler import (
    test_mark_prefill_done_rejects_incomplete_prefill,
    test_scheduler_smoke,
    test_token_budget_limits_decode_batch,
    test_token_budget_limits_prefill_batch,
)
from tests.test_model_runner import (
    test_chunked_prefill_matches_single_step,
    test_engine_add_request_uses_prefill_chunk_size,
    test_engine_with_small_prefill_chunks,
)
from tests.test_observability import (
    test_engine_metrics_smoke,
    test_observability_routes,
    test_optimization_tracker,
    test_percentile,
)
from tests.test_server import (
    test_completions_non_streaming_explicit,
    test_completions_smoke,
    test_completions_stream_sse,
    test_completions_validation,
    test_health,
)


class SmokeTests(unittest.TestCase):
    def test_kv_cache(self):
        test_kv_cache_smoke()

    def test_attention(self):
        test_attention_cache_smoke()

    def test_scheduler(self):
        test_scheduler_smoke()

    def test_scheduler_chunked_prefill_guard(self):
        test_mark_prefill_done_rejects_incomplete_prefill()

    def test_scheduler_token_budget_prefill(self):
        test_token_budget_limits_prefill_batch()

    def test_scheduler_token_budget_decode(self):
        test_token_budget_limits_decode_batch()

    def test_engine(self):
        test_engine_smoke()

    def test_engine_token_callback(self):
        test_token_callback_emits_all_generated_tokens()

<<<<<<< HEAD
    def test_engine_stream_matches_generate(self):
        test_stream_request_matches_generate()

    def test_engine_stream_chunked_prefill(self):
        test_stream_request_with_chunked_prefill()

    def test_engine_stream_cleanup(self):
        test_stream_request_unregisters_callback()
=======
    def test_engine_generate_stream_matches_generate(self):
        test_generate_stream_matches_generate()

    def test_engine_generate_stream_chunked_prefill(self):
        test_generate_stream_with_chunked_prefill()

    def test_engine_generate_stream_cleanup(self):
        test_generate_stream_unregisters_callback()

    def test_worker_generate(self):
        test_worker_generate_matches_engine()

    def test_worker_generate_stream(self):
        test_worker_generate_stream_matches_generate()

    def test_worker_concurrent_generates(self):
        test_worker_concurrent_generates()

    def test_worker_concurrent_stream_and_generate(self):
        test_worker_concurrent_stream_and_generate()
>>>>>>> f172705 (Add support for token streaming)

    def test_static_batch(self):
        test_static_batch_matches_single()

    def test_model_runner(self):
        test_model_runner_via_engine()

    def test_backends(self):
        test_load_gpt_backend()

    def test_server_health(self):
        test_health()

    def test_server_validation(self):
        test_completions_validation()

    def test_server_completions(self):
        test_completions_smoke()

    def test_server_non_streaming(self):
        test_completions_non_streaming_explicit()

    def test_server_stream(self):
        test_completions_stream_sse()

    def test_model_runner_chunked(self):
        test_chunked_prefill_matches_single_step()

    def test_engine_chunked_prefill(self):
        test_engine_with_small_prefill_chunks()

    def test_engine_prefill_chunk_size(self):
        test_engine_add_request_uses_prefill_chunk_size()

    def test_observability_percentile(self):
        test_percentile()

    def test_observability_engine(self):
        test_engine_metrics_smoke()

    def test_observability_tracker(self):
        test_optimization_tracker()

    def test_observability_routes(self):
        test_observability_routes()


if __name__ == "__main__":
    unittest.main()
