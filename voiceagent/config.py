from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

POLICIES = ("always_small", "always_large", "always_turbo", "adaptive", "adaptive_no_filler")
LATENCY_KEYS = ("asr_small", "asr_large", "llm_small_first_sentence", "llm_large_first_sentence", "tts")
TURBO_LATENCY_KEY = "llm_turbo_first_sentence"


@dataclass
class LLMService:
    model: str
    urls: list[str]
    router_model: str | None = None  # small tier only: served name of the fine-tuned router (LoRA)


@dataclass
class Config:
    deadline_s: float
    max_wait_s: float
    asr_snr_threshold_db: float
    filler_mode: str
    max_tokens: int
    request_timeout_s: float
    rag: dict
    asr_small: list[str]
    asr_large: list[str]
    tts: list[str]
    llm_small: LLMService
    llm_large: LLMService
    llm_turbo: LLMService | None
    latency: dict[str, dict[str, float]]

    def all_urls(self) -> list[str]:
        turbo = self.llm_turbo.urls if self.llm_turbo else []
        return [*self.asr_small, *self.asr_large, *self.tts, *self.llm_small.urls, *self.llm_large.urls, *turbo]


def _expand(obj):
    if isinstance(obj, str):
        value = os.path.expandvars(obj)
        if "${" in value or value.startswith("$"):
            raise ValueError(f"unset environment variable in config value: {obj}")
        return value
    if isinstance(obj, list):
        return [_expand(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    return obj


def load_config(path: str | Path) -> Config:
    raw = _expand(yaml.safe_load(Path(path).read_text()))
    services = raw.pop("services")
    latency = raw.pop("latency")
    if isinstance(latency, str):
        latency = json.loads(Path(latency).read_text())
    turbo = services.get("llm_turbo")
    missing = set(LATENCY_KEYS) - set(latency)
    if turbo and TURBO_LATENCY_KEY not in latency:
        missing.add(TURBO_LATENCY_KEY)
    if missing:
        raise ValueError(f"latency model missing keys: {sorted(missing)}")
    if raw.get("filler_mode", "bank") not in ("bank", "generated"):
        raise ValueError("filler_mode must be 'bank' or 'generated'")
    return Config(
        deadline_s=float(raw.get("deadline_s", 1.0)),
        max_wait_s=float(raw.get("max_wait_s", 4.0)),
        asr_snr_threshold_db=float(raw.get("asr_snr_threshold_db", 10.0)),
        filler_mode=raw.get("filler_mode", "bank"),
        max_tokens=int(raw.get("max_tokens", 200)),
        request_timeout_s=float(raw.get("request_timeout_s", 60.0)),
        rag=raw["rag"],
        asr_small=services["asr_small"],
        asr_large=services["asr_large"],
        tts=services["tts"],
        llm_small=LLMService(**services["llm_small"]),
        llm_large=LLMService(**services["llm_large"]),
        llm_turbo=LLMService(**turbo) if turbo else None,
        latency=latency,
    )
