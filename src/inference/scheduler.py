from collections import deque

from inference.data_model import InferenceRequest, RequestState, ScheduleResult


class Scheduler:
    """Assigns KV-cache slots and picks prefill/decode batches each step."""

    def __init__(self, max_concurrent_requests: int):
        self.max_concurrent_requests = max_concurrent_requests
        self.waiting: deque = deque()
        self.running: dict = {}
        self.completed: dict = {}
        self.failed: dict = {}
        self.free_slots: list = list(range(max_concurrent_requests))

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
