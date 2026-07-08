from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Optional


class RequestState(Enum):
    """Lifecycle of one generation job."""

    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"


@dataclass
class InferenceRequest:
    request_id: str
    prompt_token_ids: list
    max_new_tokens: int
    state: RequestState = RequestState.WAITING
    output_token_ids: list = field(default_factory=list)
    batch_idx: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    first_token_at: Optional[float] = None
    finished_at: Optional[float] = None
    last_token_at: Optional[float] = None
    prefill_duration_sec: Optional[float] = None
    prefill_offset: int = 0
    prefill_chunk_size: int = 128
    slot_prepared: bool = False
    prefix_cache_hit_tokens: int = 0
    status: str = "ok"

    @property
    def prefill_complete(self) -> bool:
        return self.prefill_offset >= self.num_prompt_tokens

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_generated(self) -> int:
        return len(self.output_token_ids)


@dataclass
class ScheduleResult:
    prefill_requests: list
    decode_requests: list


@dataclass(frozen=True)
class ModelConfig:
    """Architecture metadata needed by the inference engine and KV cache."""

    num_layers: int
    max_seq_len: int
    n_heads: int
    head_dim: int
    vocab_size: int
    pad_token_id: int
    block_size: int = 16