from inference.data_model import InferenceRequest
from inference.scheduler import Scheduler


def make_req(rid: str) -> InferenceRequest:
    return InferenceRequest(
        request_id=rid,
        prompt_token_ids=[1, 2, 3],
        max_new_tokens=2,
    )


def test_scheduler_smoke():
    scheduler = Scheduler(max_concurrent_requests=2)
    scheduler.add_request(make_req("A"))
    scheduler.add_request(make_req("B"))
    scheduler.add_request(make_req("C"))

    result = scheduler.schedule()
    assert len(result.prefill_requests) == 2
    assert len(result.decode_requests) == 0
    assert len(scheduler.waiting) == 1
    assert len(scheduler.running) == 2

    for req in result.prefill_requests:
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
