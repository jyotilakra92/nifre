from collections import deque

from inference.data_model import InferenceRequest, RequestState, ScheduleResult


class Scheduler:
    """Assigns KV-cache slots and picks prefill/decode batches each step."""

    def __init__(self, max_concurrent_requests: int, max_tokens_per_step: int = 2048):
        if max_concurrent_requests <= 0:
            raise ValueError(
                f"max_concurrent_requests must be positive, got {max_concurrent_requests}"
            )
        if max_tokens_per_step <= 0:
            raise ValueError(f"max_tokens_per_step must be positive, got {max_tokens_per_step}")

        self.max_concurrent_requests = max_concurrent_requests
        self.max_tokens_per_step = max_tokens_per_step
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

        budget = self.max_tokens_per_step
        decode_requests = []
        prefill_requests = []

        for request in self.running.values():
            if request.state != RequestState.DECODE:
                continue
            if budget < 1:
                break
            decode_requests.append(request)
            budget -= 1

        for request in self.running.values():
            if request.state != RequestState.PREFILL:
                continue
            chunk_tokens = self._prefill_chunk_tokens(request)
            if chunk_tokens > budget:
                continue
            prefill_requests.append(request)
            budget -= chunk_tokens

        return ScheduleResult(
            prefill_requests=prefill_requests,
            decode_requests=decode_requests,
        )

    def reconfigure(
        self,
        *,
        max_tokens_per_step: int | None = None,
        max_concurrent_requests: int | None = None,
        cache_initialized: bool = False,
    ) -> None:
        if max_tokens_per_step is not None:
            if max_tokens_per_step <= 0:
                raise ValueError(
                    f"max_tokens_per_step must be positive, got {max_tokens_per_step}"
                )
            self.max_tokens_per_step = max_tokens_per_step

        if max_concurrent_requests is not None:
            self._apply_max_concurrent(max_concurrent_requests, cache_initialized)

    def _apply_max_concurrent(self, new_max: int, cache_initialized: bool) -> None:
        if new_max <= 0:
            raise ValueError(f"max_concurrent_requests must be positive, got {new_max}")

        old_max = self.max_concurrent_requests
        if new_max == old_max:
            return

        if cache_initialized and new_max > old_max:
            raise ValueError(
                "cannot increase max_concurrent_requests after KV cache is initialized"
            )

        if len(self.running) > new_max:
            raise ValueError(
                f"cannot set max_concurrent_requests to {new_max} with "
                f"{len(self.running)} running requests"
            )

        for request in self.running.values():
            if request.batch_idx is not None and request.batch_idx >= new_max:
                raise ValueError(
                    f"running request uses slot {request.batch_idx}, "
                    f"above new max {new_max}"
                )

        if new_max > old_max:
            for slot in range(old_max, new_max):
                if slot not in self.free_slots:
                    self.free_slots.append(slot)
        else:
            self.free_slots = [slot for slot in self.free_slots if slot < new_max]

        self.free_slots.sort()
        self.max_concurrent_requests = new_max

    def _prefill_chunk_tokens(self, request: InferenceRequest) -> int:
        remaining = request.num_prompt_tokens - request.prefill_offset
        return min(request.prefill_chunk_size, remaining)

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
        if not request.prefill_complete:
            raise ValueError(
                f"prefill incomplete: offset={request.prefill_offset}, "
                f"prompt_len={request.num_prompt_tokens}"
            )
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

    def cancel_request(self, request_id: str, status: str = "timeout") -> InferenceRequest:
        """Remove a request from waiting or running and record it as failed."""
        remaining = deque()
        found = None
        while self.waiting:
            request = self.waiting.popleft()
            if request.request_id == request_id:
                found = request
            else:
                remaining.append(request)
        self.waiting = remaining

        if found is not None:
            found.state = RequestState.FINISHED
            found.status = status
            self.failed[request_id] = found
            return found

        request = self.running.pop(request_id, None)
        if request is None:
            raise KeyError(f"unknown request_id: {request_id}")

        request.state = RequestState.FINISHED
        request.status = status
        if request.batch_idx is not None:
            self.free_slots.append(request.batch_idx)
            request.batch_idx = None
        self.failed[request_id] = request
        return request

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)
