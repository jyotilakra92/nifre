"""Background worker that owns the engine step loop for concurrent serving."""

import threading
import time
from queue import Empty, Queue
from typing import Iterator, List, Optional

from inference.data_model import InferenceRequest
from inference.engine import Engine

_POLL_INTERVAL_SECONDS = 0.001


class EngineWorker:
    """Runs ``engine.step()`` on a dedicated thread so clients only submit work."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self._lock = threading.Lock()
        self._work_available = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="engine-worker", daemon=True)

    def start(self) -> None:
        if self._thread.is_alive():
            return
        self._stop.clear()
        self._thread.start()

    def stop(self, timeout: Optional[float] = 5.0) -> None:
        self._stop.set()
        with self._lock:
            self._work_available.notify_all()
        self._thread.join(timeout=timeout)

    def generate(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int,
        timeout_seconds: Optional[float] = None,
    ) -> InferenceRequest:
        """Queue one request and block until it finishes."""
        with self._lock:
            request_id = self.engine.add_request(prompt_token_ids, max_new_tokens)
            self._work_available.notify()

        deadline = time.time() + timeout_seconds if timeout_seconds else None

        while not self._stop.is_set():
            with self._lock:
                if request_id in self.engine.scheduler.completed:
                    return self.engine.scheduler.completed[request_id]
                if request_id in self.engine.scheduler.failed:
                    return self.engine.scheduler.failed[request_id]
                if deadline and time.time() > deadline:
                    request = self.engine._cancel_request(request_id, status="timeout")
                    if self.engine.metrics and request is not None:
                        self.engine.metrics.on_request_failed(request, "timeout")
                    return self.engine.scheduler.failed[request_id]

            time.sleep(_POLL_INTERVAL_SECONDS)

        raise RuntimeError("engine worker stopped before request completed")

    def generate_stream(
        self,
        prompt_token_ids: List[int],
        max_new_tokens: int,
    ) -> Iterator[int]:
        """Yield output token ids as they are produced."""
        token_queue: Queue[int] = Queue()
        with self._lock:
            request_id = self.engine.add_request(prompt_token_ids, max_new_tokens)
            self.engine.register_token_callback(request_id, token_queue.put)
            self._work_available.notify()

        try:
            while not self._stop.is_set():
                finished = False
                with self._lock:
                    finished = (
                        request_id in self.engine.scheduler.completed
                        or request_id in self.engine.scheduler.failed
                    )

                while True:
                    try:
                        yield token_queue.get_nowait()
                    except Empty:
                        break

                if finished:
                    while True:
                        try:
                            yield token_queue.get_nowait()
                        except Empty:
                            break
                    break

                time.sleep(_POLL_INTERVAL_SECONDS)
        finally:
            with self._lock:
                self.engine.unregister_token_callback(request_id)

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                if self._stop.is_set():
                    break
                if self.engine.scheduler.has_work():
                    self.engine.step()
                else:
                    self._work_available.wait(timeout=0.05)
