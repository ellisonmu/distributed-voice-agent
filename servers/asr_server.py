"""Whisper over HTTP (faster-whisper / CTranslate2). POST /transcribe with a 16-bit WAV body
returns {"text": ...}. num_workers lets several requests decode on the GPU at once."""
from __future__ import annotations

import argparse
import asyncio

import uvicorn
from fastapi import FastAPI, Request

from voiceagent.audio.wav import read_wav, resample

app = FastAPI()
MODEL = None
BEAM_SIZE = 1


def _transcribe(audio) -> str:
    segments, _ = MODEL.transcribe(audio, beam_size=BEAM_SIZE, language="en", condition_on_previous_text=False)
    return " ".join(s.text.strip() for s in segments)


@app.get("/health")
async def health():
    return {"ok": MODEL is not None}


@app.post("/transcribe")
async def transcribe(request: Request):
    x, sr = read_wav(await request.body())
    text = await asyncio.to_thread(_transcribe, resample(x, sr, 16000))
    return {"text": text}


def main() -> None:
    global MODEL, BEAM_SIZE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="e.g. small or large-v3")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--compute-type", default="float16")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--beam-size", type=int, default=1)
    a = p.parse_args()

    from faster_whisper import WhisperModel

    MODEL = WhisperModel(a.model, device=a.device, compute_type=a.compute_type, num_workers=a.num_workers)
    BEAM_SIZE = a.beam_size
    uvicorn.run(app, host="0.0.0.0", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
