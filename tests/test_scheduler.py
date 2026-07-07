from inference.data_model import InferenceRequest
from inference.scheduler import Scheduler


def make_req(rid: str, prompt=None, chunk_size=128) -> InferenceRequest:
    return InferenceRequest(
        request_id=rid,
        prompt_token_ids=prompt if prompt is not None else [1, 2, 3],
        max_new_tokens=2,
        prefill_chunk_size=chunk_size,
    )


def finish_prefill(request: InferenceRequest) -> None:
    """Simulate a completed prompt cache fill for scheduler unit tests."""
    request.prefill_offset = request.num_prompt_tokens


def test_scheduler_smoke():
    scheduler = Scheduler(max_concurrent_requests=2, max_tokens_per_step=4096)
    scheduler.add_request(make_req("A"))
    scheduler.add_request(make_req("B"))
    scheduler.add_request(make_req("C"))

    result = scheduler.schedule()
    assert len(result.prefill_requests) == 2
    assert len(result.decode_requests) == 0
    assert len(scheduler.waiting) == 1
    assert len(scheduler.running) == 2

    for req in result.prefill_requests:
        finish_prefill(req)
        scheduler.mark_prefill_done(req)

    result = scheduler.schedule()
    assert len(result.prefill_requests) == 0
    assert len(result.decode_requests) == 2
    assert len(scheduler.waiting) == 1

    scheduler.mark_decode_done(result.decode_requests[0], token_id=100)
    scheduler.mark_decode_done(result.decode_requests[1], token_id=200)

    result = scheduler.schedule()
    assert len(result.prefill_requests) == 0
    assert len(result.decode_requests) == 2
    assert len(scheduler.waiting) == 1

    scheduler.mark_decode_done(scheduler.running["A"], token_id=101)

    result = scheduler.schedule()
    assert len(result.prefill_requests) == 1
    assert result.prefill_requests[0].request_id == "C"
    assert len(result.decode_requests) == 1
    assert result.decode_requests[0].request_id == "B"


def test_mark_prefill_done_rejects_incomplete_prefill():
    scheduler = Scheduler(max_concurrent_requests=1, max_tokens_per_step=4096)
    scheduler.add_request(make_req("A"))
    scheduler.schedule()

    request = scheduler.running["A"]
    assert request.prefill_offset == 0

    try:
        scheduler.mark_prefill_done(request)
        raise AssertionError("expected ValueError for incomplete prefill")
    except ValueError as exc:
        assert "prefill incomplete" in str(exc)

    assert request.state.value == "prefill"


def test_token_budget_limits_prefill_batch():
    scheduler = Scheduler(max_concurrent_requests=2, max_tokens_per_step=4)
    scheduler.add_request(make_req("A", prompt=[1, 2, 3, 4, 5], chunk_size=3))
    scheduler.add_request(make_req("B", prompt=[10, 20, 30, 40, 50], chunk_size=3))
    scheduler.schedule()

    result = scheduler.schedule()
    assert [req.request_id for req in result.prefill_requests] == ["A"]
    assert result.decode_requests == []


def test_token_budget_limits_decode_batch():
    scheduler = Scheduler(max_concurrent_requests=3, max_tokens_per_step=2)
    scheduler.add_request(make_req("A"))
    scheduler.add_request(make_req("B"))
    scheduler.add_request(make_req("C"))
    scheduler.schedule()

    for req in scheduler.running.values():
        finish_prefill(req)
        scheduler.mark_prefill_done(req)

    result = scheduler.schedule()
    assert len(result.decode_requests) == 2
    assert result.prefill_requests == []
