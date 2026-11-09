"""The per-turn JSON log line. This is the interface every stream builds against: the
orchestrator writes it, and calibration, summaries, WER and the judge only read it.
All times are seconds since the run started."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class StageRecord:
    key: str  # one of config.LATENCY_KEYS, or "llm_small_router"
    inflight: int  # requests already in flight on that service when this one was sent
    t0: float
    t1: float


@dataclass
class TurnRecord:
    run_id: str
    policy: str
    n_callers: int
    caller_id: int
    turn_idx: int
    question_id: str
    question_text: str
    snr_db: float | None  # injected noise level (ground truth); None means clean
    repeat: bool = False
    est_snr_db: float | None = None
    asr_model: str = ""
    llm_model: str = ""  # small | large | turbo
    route: str = ""  # plain | answer | escalate_large | escalate_turbo | escalate_hedge
    router_decision: str = ""  # ANSWER | ESCALATE | EXPERT | UNKNOWN
    router_raw: str = ""
    transcript: str = ""
    filler_text: str = ""
    answer: str = ""
    t_start: float | None = None
    t_asr_done: float | None = None
    t_retrieval_done: float | None = None
    t_first_audio: float | None = None
    t_first_answer_audio: float | None = None
    t_audio_done: float | None = None
    t_playback_end: float | None = None
    ttfa: float | None = None
    ttsa: float | None = None
    stall_count: int = 0  # silences between clips longer than player.STALL_THRESHOLD_S
    stall_total_s: float = 0.0
    max_gap_s: float = 0.0
    stages: list[StageRecord] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)
