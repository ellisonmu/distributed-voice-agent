from __future__ import annotations

import random

from voiceagent.audio.wav import wav_duration

# v1 baseline. None of these may contain medical information.
FILLER_LINES = [
    "Good question, let me pull up the details on that.",
    "Let me look into that for you, one moment.",
    "Okay, give me a second to check on that.",
    "That's worth getting right, let me take a closer look.",
    "Sure, let me find the most accurate information for you.",
    "One moment while I look that up.",
    "Let me check the details so I give you a good answer.",
    "Thanks for asking, I'm pulling that up now.",
    "Hang on just a second, I'm checking on that.",
    "Let me make sure I have the right information for you.",
    "Good question. Give me a moment.",
    "I want to be careful with that one, let me check.",
    "Let me take a look at that for you.",
    "Just a moment, I'm finding the details.",
    "Okay, let me dig into that.",
    "Bear with me a second while I check.",
    "Let me get you a proper answer on that.",
    "I'm looking that up right now.",
    "Give me a second to find the right information.",
    "Let me double check that for you.",
]


class FillerBank:
    """Holding lines rendered once at startup, so playing one costs no TTS time per turn."""

    def __init__(self, lines: list[str] = FILLER_LINES, seed: int = 0):
        self.lines = lines
        self.durations: dict[str, float] = {}
        self.rng = random.Random(seed)

    async def prerender(self, tts) -> None:
        for line in self.lines:
            wav, _ = await tts.synthesize(line)
            self.durations[line] = wav_duration(wav)

    def pick(self) -> tuple[str, float]:
        line = self.rng.choice(self.lines)
        return line, self.durations[line]
