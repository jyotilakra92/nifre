import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tests.test_admin_tuning import (
    test_admin_tuning_enable_and_update_goal,
    test_admin_tuning_get_status,
    test_auto_tune_flag_starts_enabled,
    test_health_exposes_auto_tune_endpoint,
    test_observability_tuning_route,
)
from tests.test_server_metrics import (
    test_from_nifre_observability_extracts_key_fields,
    test_from_vllm_prometheus_extracts_throughput_and_cache,
)
from tests.test_bench import (
    test_compare_format_reports_both_labels_and_ratio,
    test_format_report,
    test_profiles_exist,
    test_run_bench_counts_completion_tokens,
    test_run_bench_unknown_profile,
    test_run_bench_with_mock_request_fn,
)
from tests.test_tuning_controller import (
    test_controller_promotes_on_latency_improvement,
    test_controller_respects_cooldown,
    test_controller_rolls_back_on_latency_regression,
    test_controller_rolls_back_on_neutral_result,
    test_controller_skips_without_min_completed_requests,
    test_controller_starts_attempt_and_applies_changes,
    test_evaluation_helpers,
)
from tests.test_tuning_policy import (
    test_policy_accepts_string_goal,
    test_policy_balanced_queue_high_reduces_chunks_after_cache_init,
    test_policy_holds_on_error_elevated,
    test_policy_latency_sensitive_decreases_chunk_and_budget,
    test_policy_prefix_friendly_enables_prefix_cache,
    test_policy_respects_lower_bound,
    test_policy_returns_none_when_no_action,
    test_policy_throughput_low_increases_token_budget,
    test_policy_throughput_queue_high_increases_concurrency_before_cache_init,
    test_policy_throughput_queue_high_skips_concurrency_after_cache_init,
)
from tests.test_workload_classifier import (
    test_classifier_error_elevated,
    test_classifier_latency_sensitive,
    test_classifier_multiple_labels,
    test_classifier_prefix_friendly,
    test_classifier_queue_high_not_triggered_by_brief_spike,
    test_classifier_queue_high_requires_sustained_depth,
    test_classifier_throughput_low_only_under_load,
    test_workload_snapshot_from_metrics,
    test_workload_snapshot_prefix_hit_rate_zero_when_no_tokens,
)
from tests.test_engine_reconfigure import (
    test_get_config_returns_current_settings,
    test_reconfigure_cache_flags_before_first_step,
    test_reconfigure_max_concurrent_before_cache_init,
    test_reconfigure_max_concurrent_decrease_after_cache_init,
    test_reconfigure_max_tokens_per_step_affects_next_schedule,
    test_reconfigure_prefill_chunk_size_applies_to_new_requests_only,
    test_reconfigure_rejects_cache_type_toggle_after_cache_init,
    test_reconfigure_rejects_invalid_values,
    test_reconfigure_rejects_max_concurrent_increase_after_cache_init,
    test_reconfigure_round_trip,
)
from tests.test_prefix_cache import (
    test_engine_prefix_cache_disabled,
    test_engine_reuses_prefix_for_shared_prompt,
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
from tests.test_backends import test_load_gpt_backend, test_load_hf_backend
from tests.test_hf_backend import (
    test_hf_auto_engine_generate,
    test_hf_concurrent_prefills_of_different_lengths,
    test_hf_prefix_cache_metrics_and_block_dedup,
    test_hf_prefix_cache_reuses_shared_prefix_without_changing_output,
)
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
from tests.test_openai_api import (
    test_build_completion_response_shape,
    test_completion_chunk_sse_shape,
)
from tests.test_server import (
    test_completions_non_streaming_openai_shape,
    test_completions_smoke,
    test_completions_stream_openai_shape,
    test_completions_stream_respects_requested_model_name,
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
        test_engine_reuses_prefix_for_shared_prompt()
        test_engine_prefix_cache_disabled()

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

    def test_engine_reconfigure(self):
        test_get_config_returns_current_settings()
        test_reconfigure_round_trip()
        test_reconfigure_prefill_chunk_size_applies_to_new_requests_only()
        test_reconfigure_max_tokens_per_step_affects_next_schedule()
        test_reconfigure_rejects_invalid_values()
        test_reconfigure_max_concurrent_before_cache_init()
        test_reconfigure_max_concurrent_decrease_after_cache_init()
        test_reconfigure_rejects_max_concurrent_increase_after_cache_init()
        test_reconfigure_rejects_cache_type_toggle_after_cache_init()
        test_reconfigure_cache_flags_before_first_step()

    def test_workload_classifier(self):
        test_workload_snapshot_from_metrics()
        test_workload_snapshot_prefix_hit_rate_zero_when_no_tokens()
        test_classifier_latency_sensitive()
        test_classifier_prefix_friendly()
        test_classifier_throughput_low_only_under_load()
        test_classifier_error_elevated()
        test_classifier_queue_high_requires_sustained_depth()
        test_classifier_queue_high_not_triggered_by_brief_spike()
        test_classifier_multiple_labels()

    def test_tuning_policy(self):
        test_policy_holds_on_error_elevated()
        test_policy_latency_sensitive_decreases_chunk_and_budget()
        test_policy_throughput_queue_high_increases_concurrency_before_cache_init()
        test_policy_throughput_queue_high_skips_concurrency_after_cache_init()
        test_policy_throughput_low_increases_token_budget()
        test_policy_prefix_friendly_enables_prefix_cache()
        test_policy_balanced_queue_high_reduces_chunks_after_cache_init()
        test_policy_returns_none_when_no_action()
        test_policy_respects_lower_bound()
        test_policy_accepts_string_goal()

    def test_tuning_controller(self):
        test_controller_starts_attempt_and_applies_changes()
        test_controller_promotes_on_latency_improvement()
        test_controller_rolls_back_on_latency_regression()
        test_controller_rolls_back_on_neutral_result()
        test_controller_skips_without_min_completed_requests()
        test_controller_respects_cooldown()
        test_evaluation_helpers()

    def test_admin_tuning(self):
        test_admin_tuning_get_status()
        test_admin_tuning_enable_and_update_goal()
        test_observability_tuning_route()
        test_health_exposes_auto_tune_endpoint()
        test_auto_tune_flag_starts_enabled()

    def test_bench(self):
        test_profiles_exist()
        test_run_bench_with_mock_request_fn()
        test_run_bench_counts_completion_tokens()
        test_run_bench_unknown_profile()
        test_format_report()
        test_compare_format_reports_both_labels_and_ratio()

    def test_server_metrics(self):
        test_from_nifre_observability_extracts_key_fields()
        test_from_vllm_prometheus_extracts_throughput_and_cache()

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
        test_load_hf_backend()

    def test_hf_backend_engine(self):
        test_hf_auto_engine_generate()

    def test_hf_prefix_cache(self):
        test_hf_prefix_cache_reuses_shared_prefix_without_changing_output()

    def test_hf_concurrent_prefills(self):
        test_hf_concurrent_prefills_of_different_lengths()

    def test_hf_prefix_cache_metrics(self):
        test_hf_prefix_cache_metrics_and_block_dedup()

    def test_openai_api(self):
        test_completion_chunk_sse_shape()
        test_build_completion_response_shape()

    def test_server_health(self):
        test_health()

    def test_server_validation(self):
        test_completions_validation()

    def test_server_completions(self):
        test_completions_smoke()

    def test_server_non_streaming(self):
        test_completions_non_streaming_openai_shape()

    def test_server_stream(self):
        test_completions_stream_openai_shape()
        test_completions_stream_respects_requested_model_name()

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
