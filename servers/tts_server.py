"""Kokoro-82M TTS over HTTP. POST /synthesize {"text", "voice"?} returns a 24 kHz WAV.
Kokoro stands in for Breeze-TTS 2 from the mid-term deck (fast, small and easy to serve);
the HTTP interface is what the rest of the system depends on, so it can be swapped.
One replica synthesizes one request at a time; scale with more replicas."""
from __future__ import annotations

import argparse
import asyncio
import threading

import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response

from voiceagent.audio.wav import write_wav

app = FastAPI()
PIPELINE = None
DEFAULT_VOICE = "af_heart"
SAMPLE_RATE = 24000
LOCK = threading.Lock()


def _synthesize(text: str, voice: str) -> bytes:
    with LOCK:
        chunks = [audio for _, _, audio in PIPELINE(text, voice=voice)]
    audio = np.concatenate([c.cpu().numpy() if hasattr(c, "cpu") else np.asarray(c) for c in chunks])
    return write_wav(audio.astype(np.float32), SAMPLE_RATE)


@app.get("/health")
async def health():
    return {"ok": PIPELINE is not None}


@app.post("/synthesize")
async def synthesize(request: Request):
    body = await request.json()
    wav = await asyncio.to_thread(_synthesize, body["text"], body.get("voice", DEFAULT_VOICE))
    return Response(wav, media_type="audio/wav")


def main() -> None:
    global PIPELINE, DEFAULT_VOICE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--voice", default="af_heart")
    a = p.parse_args()

    from kokoro import KPipeline

    PIPELINE = KPipeline(lang_code="a")
    DEFAULT_VOICE = a.voice
    _synthesize("Warming up.", DEFAULT_VOICE)
    uvicorn.run(app, host="0.0.0.0", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
