from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable

from voiceagent.audio.wav import wav_duration
from voiceagent.clients import TTSClient
from voiceagent.records import StageRecord

# Silences shorter than this sound like a normal pause between sentences, not a stall.
STALL_THRESHOLD_S = 0.15


@dataclass
class Clip:
    kind: str  # filler | answer
    text: str
    ready: float
    duration: float


class Player:
    """Synthesizes one turn's sentences in order and simulates playback on the caller's
    side: a clip starts when it is ready and the previous clip has finished playing."""

    def __init__(self, tts: TTSClient, clock: Callable[[], float], stages: list[StageRecord]):
        self.tts, self.clock, self.stages = tts, clock, stages
        self.clips: list[Clip] = []
        self.queue: asyncio.Queue = asyncio.Queue()
        self.worker = asyncio.create_task(self._run())

    def say(self, text: str, kind: str) -> None:
        self.queue.put_nowait((kind, text, None))

    def play_prerendered(self, text: str, duration: float, kind: str = "filler") -> None:
        self.queue.put_nowait((kind, text, duration))

    async def _run(self) -> None:
        while (item := await self.queue.get()) is not None:
            kind, text, duration = item
            if duration is None:
                t0 = self.clock()
                wav, inflight = await self.tts.synthesize(text)
                self.stages.append(StageRecord("tts", inflight, t0, self.clock()))
                duration = wav_duration(wav)
            self.clips.append(Clip(kind, text, self.clock(), duration))

    async def finish(self) -> float:
        """Waits for all synthesis and returns when the caller hears the last clip end.
        Afterwards self.gaps holds every silence between clips (the next clip wasn't ready
        when the previous one finished playing)."""
        self.queue.put_nowait(None)
        await self.worker
        end = None
        self.gaps: list[float] = []
        for clip in self.clips:
            if end is None:
                start = clip.ready
            else:
                if clip.ready > end:
                    self.gaps.append(clip.ready - end)
                start = max(clip.ready, end)
            end = start + clip.duration
        return end if end is not None else self.clock()

    def cancel(self) -> None:
        if not self.worker.done():
            self.worker.cancel()
