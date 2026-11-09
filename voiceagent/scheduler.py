from __future__ import annotations

from voiceagent.clients import Pool
from voiceagent.config import Config


class LatencyModel:
    """latency(stage) = base_s + per_inflight_s * requests already in flight. Fitted from the
    E1 calibration run by voiceagent.analysis.calibrate."""

    def __init__(self, params: dict[str, dict[str, float]]):
        self.params = params

    def predict(self, key: str, inflight: int) -> float:
        p = self.params[key]
        return p["base_s"] + p["per_inflight_s"] * inflight


class Scheduler:
    def __init__(self, cfg: Config, latency: LatencyModel, pools: dict[str, Pool]):
        self.cfg, self.latency, self.pools = cfg, latency, pools

    def _predict(self, key: str, pool: str) -> float:
        return self.latency.predict(key, self.pools[pool].inflight)

    def choose_asr(self, policy: str, est_snr_db: float) -> str:
        if policy == "always_small":
            return "small"
        if policy in ("always_large", "always_turbo"):
            return "large"
        if est_snr_db >= self.cfg.asr_snr_threshold_db:
            return "small"
        predicted_first_audio = (
            self._predict("asr_large", "asr_large")
            + self._predict("llm_small_first_sentence", "llm_small")
            + self._predict("tts", "tts")
        )
        return "large" if predicted_first_audio <= self.cfg.deadline_s else "small"

    def fits(self, tier: str, elapsed_s: float) -> bool:
        """Whether this tier's first spoken sentence lands within max_wait of turn start."""
        predicted = elapsed_s + self._predict(f"llm_{tier}_first_sentence", f"llm_{tier}") + self._predict("tts", "tts")
        return predicted <= self.cfg.max_wait_s

    def choose_escalation(self, decision: str, elapsed_s: float) -> str | None:
        """EXPERT tries turbo, then large; ESCALATE tries large. None means no tier fits in
        time and the small LLM gives a hedged answer. Without a turbo pool, EXPERT means large."""
        wanted = ("turbo", "large") if decision == "EXPERT" else ("large",)
        for tier in wanted:
            if f"llm_{tier}" in self.pools and self.fits(tier, elapsed_s):
                return tier
        return None
