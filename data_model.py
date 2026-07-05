from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Optional


class RequestState(Enum):
    """Lifecycle of one generation job."""

    WAITING = "waiting"    # queued, no KV-cache slot yet
    PREFILL = "prefill"    # has a slot; prompt not yet processed
    DECODE = "decode"      # prefill done; generating one token per step
    FINISHED = "finished"  # done (max tokens reached or EOS)


@dataclass
class InferenceRequest:
    request_id: str
    prompt_token_ids: list[int]
    max_new_tokens: int
    state: RequestState = RequestState.WAITING
    output_token_ids: list[int] = field(default_factory=list)
    batch_idx: Optional[int] = None
    created_at: float = field(default_factory=time.time)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_generated(self) -> int:
        return len(self.output_token_ids)

@dataclass
class ScheduleResult:
    prefill_requests: list[InferenceRequest]
    decode_requests: list[InferenceRequest] 
    