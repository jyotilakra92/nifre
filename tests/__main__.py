import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
MODEL = SRC / "model"
for path in (SRC, MODEL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tests.test_prefix_cache import (
    test_prefix_cache_does_not_retain_existing_key,
    test_prefix_cache_eviction_drops_stale_entry,
    test_prefix_cache_ignores_partial_tail_block,
    test_prefix_cache_init_validation,
    test_prefix_cache_insert_retains_blocks,
    test_prefix_cache_lookup_and_insert,
    test_prefix_cache_partial_match_stops_at_first_miss,
)
from tests.test_paged_kv_cache import (
    test_paged_kv_cache_block_growth,
    test_paged_kv_cache_prefix_cache_can_be_disabled,
    test_paged_kv_cache_register_and_reload_prefix,
    test_paged_kv_cache_reset_slot_frees_blocks,
    test_paged_kv_cache_smoke,
    test_paged_kv_cache_try_load_prefix_miss_returns_zero,
)
from tests.test_block_table import (
    test_clear_allows_block_reuse,
    test_clear_frees_blocks_to_allocator,
    test_import_blocks_attaches_shared_blocks,
    test_import_blocks_rejects_non_empty_table,
    test_ensure_capacity_grows_with_sequence,
    test_ensure_capacity_idempotent,
    test_ensure_capacity_rejects_negative_total,
    test_ensure_capacity_single_block,
    test_ensure_capacity_zero_tokens,
    test_init_rejects_invalid_block_size,
    test_pool_exhausted_propagates,
    test_resolve_first_and_second_block,
    test_resolve_offset_within_block,
    test_resolve_unallocated_token_raises,
)
from tests.test_block_allocator import (
    test_allocate_and_free,
    test_allocate_many_rejects_invalid_count,
    test_block_reuse_after_free,
    test_double_free_rejected,
    test_free_many_duplicate_rejected,
    test_retain_keeps_block_alive_after_one_release,
    test_retain_requires_allocated_block,
    test_free_many_empty_list,
    test_free_out_of_range_rejected,
    test_init_rejects_invalid_size,
    test_pool_exhausted,
    test_single_allocate_wrapper,
    test_utilization,
)
from tests.test_attention import test_attention_cache_smoke, test_attention_paged_cache_smoke
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

    def test_block_allocator(self):
        test_init_rejects_invalid_size()
        test_allocate_and_free()
        test_single_allocate_wrapper()
        test_block_reuse_after_free()
        test_pool_exhausted()
        test_free_out_of_range_rejected()
        test_double_free_rejected()
        test_free_many_duplicate_rejected()
        test_utilization()
        test_free_many_empty_list()
        test_allocate_many_rejects_invalid_count()
        test_retain_keeps_block_alive_after_one_release()
        test_retain_requires_allocated_block()

    def test_block_table(self):
        test_init_rejects_invalid_block_size()
        test_ensure_capacity_zero_tokens()
        test_ensure_capacity_single_block()
        test_ensure_capacity_idempotent()
        test_ensure_capacity_grows_with_sequence()
        test_ensure_capacity_rejects_negative_total()
        test_resolve_first_and_second_block()
        test_resolve_offset_within_block()
        test_resolve_unallocated_token_raises()
        test_clear_frees_blocks_to_allocator()
        test_import_blocks_attaches_shared_blocks()
        test_import_blocks_rejects_non_empty_table()
        test_clear_allows_block_reuse()
        test_pool_exhausted_propagates()

    def test_paged_kv_cache(self):
        test_paged_kv_cache_smoke()
        test_paged_kv_cache_block_growth()
        test_paged_kv_cache_reset_slot_frees_blocks()
        test_paged_kv_cache_register_and_reload_prefix()
        test_paged_kv_cache_try_load_prefix_miss_returns_zero()
        test_paged_kv_cache_prefix_cache_can_be_disabled()

    def test_prefix_cache(self):
        test_prefix_cache_lookup_and_insert()
        test_prefix_cache_partial_match_stops_at_first_miss()
        test_prefix_cache_ignores_partial_tail_block()
        test_prefix_cache_insert_retains_blocks()
        test_prefix_cache_eviction_drops_stale_entry()
        test_prefix_cache_does_not_retain_existing_key()
        test_prefix_cache_init_validation()

    def test_attention(self):
        test_attention_cache_smoke()
        test_attention_paged_cache_smoke()

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
