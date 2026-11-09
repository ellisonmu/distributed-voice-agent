import asyncio

import numpy as np

from voiceagent.audio.wav import write_wav
from voiceagent.player import Player


class FakeTTS:
    """Each sentence takes `delays[i]` seconds to synthesize and lasts `durations[i]` seconds."""

    def __init__(self, delays, durations):
        self.delays, self.durations, self.i = delays, durations, 0

    async def synthesize(self, text):
        i, self.i = self.i, self.i + 1
        await asyncio.sleep(self.delays[i])
        return write_wav(np.zeros(int(16000 * self.durations[i]), dtype=np.float32), 16000), 0


def play(delays, durations, gap_before=None):
    async def go():
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        player = Player(FakeTTS(delays, durations), lambda: loop.time() - t0, [])
        player.say("one", "answer")
        if gap_before:
            await asyncio.sleep(gap_before)
        player.say("two", "answer")
        end = await player.finish()
        return player, end
    return asyncio.run(go())


def test_buffered_clips_have_no_gap():
    player, _ = play(delays=[0.01, 0.01], durations=[0.3, 0.3])
    assert player.gaps == []


def test_late_sentence_creates_a_measured_gap():
    # clip 1 is ready at ~0.01 s and ends at ~0.11 s; sentence 2 only arrives at ~0.4 s
    player, end = play(delays=[0.01, 0.01], durations=[0.1, 0.1], gap_before=0.4)
    assert len(player.gaps) == 1 and 0.25 < player.gaps[0] < 0.4
    assert end > 0.45
