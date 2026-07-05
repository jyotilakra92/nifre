from collections import deque

from data_model import InferenceRequest, RequestState, ScheduleResult

class Scheduler:
    """Assigns KV-cache slots and picks prefill/decode batches each step.

    ``waiting`` holds only brand-new requests (``WAITING``). Once scheduled, a
    request moves into ``running`` and stays there through prefill and decode
    until it finishes — it is never put back on the waiting queue.
    """

    def __init__(self, max_concurrent_requests: int):
        self.max_concurrent_requests = max_concurrent_requests
        self.waiting: deque[InferenceRequest] = deque()
        self.running: dict[str, InferenceRequest] = {}
        self.completed: dict[str, InferenceRequest] = {}
        self.failed: dict[str, InferenceRequest] = {}
        self.free_slots: list[int] = list(range(max_concurrent_requests))

    def add_request(self, request: InferenceRequest) -> None:
        if request.state != RequestState.WAITING:
            raise ValueError("new requests must start in WAITING")
        self.waiting.append(request)

    def schedule(self) -> ScheduleResult:
        self._assign_waiting_to_slots()

        prefill_requests = [
            req for req in self.running.values() if req.state == RequestState.PREFILL
        ]
        decode_requests = [
            req for req in self.running.values() if req.state == RequestState.DECODE
        ]

        return ScheduleResult(
            prefill_requests=prefill_requests,
            decode_requests=decode_requests,
        )

    def _assign_waiting_to_slots(self) -> None:
        while self.waiting and self.free_slots:
            request = self.waiting.popleft()
            batch_idx = self.free_slots.pop(0)
            request.batch_idx = batch_idx
            request.state = RequestState.PREFILL
            self.running[request.request_id] = request

    def mark_prefill_done(self, request: InferenceRequest) -> None:
        if request.state != RequestState.PREFILL:
            raise ValueError(f"expected PREFILL, got {request.state}")
        request.state = RequestState.DECODE

    def mark_decode_done(self, request: InferenceRequest, token_id: int) -> None:
        if request.state != RequestState.DECODE:
            raise ValueError(f"expected DECODE, got {request.state}")
        request.output_token_ids.append(token_id)
        if request.num_generated >= request.max_new_tokens:
            self._finish_request(request)

    def _finish_request(self, request: InferenceRequest) -> None:
        request.state = RequestState.FINISHED
        del self.running[request.request_id]
        self.completed[request.request_id] = request
        if request.batch_idx is not None:
            self.free_slots.append(request.batch_idx)
            request.batch_idx = None

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)


def _smoke_test():
    def make_req(rid: str) -> InferenceRequest:
        return InferenceRequest(
            request_id=rid,
            prompt_token_ids=[1, 2, 3],
            max_new_tokens=2,
        )

    scheduler = Scheduler(max_concurrent_requests=2)
    scheduler.add_request(make_req("A"))
    scheduler.add_request(make_req("B"))
    scheduler.add_request(make_req("C"))

    # Step 1: two slots filled from waiting; both need prefill
    result = scheduler.schedule()
    assert len(result.prefill_requests) == 2
    assert len(result.decode_requests) == 0
    assert len(scheduler.waiting) == 1
    assert len(scheduler.running) == 2

    for req in result.prefill_requests:
        scheduler.mark_prefill_done(req)

    # Step 2: same two requests decode; C still waiting (no free slots)
    result = scheduler.schedule()
    assert len(result.prefill_requests) == 0
    assert len(result.decode_requests) == 2
    assert len(scheduler.waiting) == 1

    scheduler.mark_decode_done(result.decode_requests[0], token_id=100)
    scheduler.mark_decode_done(result.decode_requests[1], token_id=200)

    # Step 3: A and B still decoding (1/2 tokens); C still waiting
    result = scheduler.schedule()
    assert len(result.prefill_requests) == 0
    assert len(result.decode_requests) == 2
    assert len(scheduler.waiting) == 1

    # Finish A completely; frees one slot
    scheduler.mark_decode_done(
        scheduler.running["A"], token_id=101
    )

    # Step 4: C gets the freed slot; B still decoding
    result = scheduler.schedule()
    assert len(result.prefill_requests) == 1
    assert result.prefill_requests[0].request_id == "C"
    assert len(result.decode_requests) == 1
    assert result.decode_requests[0].request_id == "B"

    print("scheduler smoke test passed")


if __name__ == "__main__":
    _smoke_test()
