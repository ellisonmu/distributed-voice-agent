"""Fake ASR / LLM / TTS servers with the same HTTP interface as the real ones. Latency
grows with requests in flight, so load tests behave plausibly on a laptop."""
from __future__ import annotations

import argparse
import asyncio
import json
import random

import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from voiceagent.audio.wav import write_wav

ANSWER = (
    "Asthma is a long-term condition that affects the airways in the lungs. "
    "Common symptoms include wheezing, coughing, chest tightness and shortness of breath. "
    "Please talk to a clinician about a treatment plan that is right for you."
)


def make_app(role: str, base_s: float, per_inflight_s: float, tokens_per_s: float,
             p_escalate: float, seed: int, chars_per_s: float = 15.0, p_expert: float = 0.0) -> FastAPI:
    app = FastAPI()
    state = {"inflight": 0}
    rng = random.Random(seed)

    def delay() -> float:
        return base_s + per_inflight_s * state["inflight"]

    @app.get("/health")
    async def health():
        return {"ok": True}

    if role == "asr":
        @app.post("/transcribe")
        async def transcribe(request: Request):
            await request.body()
            state["inflight"] += 1
            try:
                await asyncio.sleep(delay())
            finally:
                state["inflight"] -= 1
            return {"text": "What are the early symptoms of asthma in adults?"}

    elif role == "tts":
        @app.post("/synthesize")
        async def synthesize(request: Request):
            text = (await request.json())["text"]
            state["inflight"] += 1
            try:
                await asyncio.sleep(delay())
            finally:
                state["inflight"] -= 1
            sr = 16000
            return Response(write_wav(np.zeros(int(sr * len(text) / chars_per_s), dtype=np.float32), sr), media_type="audio/wav")

    elif role == "llm":
        @app.post("/v1/chat/completions")
        async def chat(request: Request):
            body = await request.json()
            system = body["messages"][0]["content"]
            if not body.get("stream"):
                # Offline batch calls (judge, dataset builder): answer the whole thing at once.
                if "You grade answers" in system:
                    text = json.dumps({"verdict": rng.choice(["correct", "partial", "incorrect"]), "unsafe": False})
                elif "holding lines" in system:
                    text = json.dumps({"medical_claim": False})
                elif "while it looks up" in system:
                    text = rng.choice(["Let me look into that for you.", "One moment while I check on that."])
                else:
                    text = ANSWER
                await asyncio.sleep(delay())
                return {"choices": [{"message": {"role": "assistant", "content": text}}]}
            if "ESCALATE" in system:
                r = rng.random()
                if r < p_expert + p_escalate:
                    text = "EXPERT" if r < p_expert else "ESCALATE"
                    if "After ESCALATE" in system:
                        text += ": Let me look into that for you, one moment."
                else:
                    text = "ANSWER " + ANSWER
            else:
                text = ANSWER
            words = text.split(" ")

            async def events():
                state["inflight"] += 1
                try:
                    await asyncio.sleep(delay())
                    for i, word in enumerate(words):
                        chunk = {"choices": [{"delta": {"content": word if i == 0 else " " + word}}]}
                        yield f"data: {json.dumps(chunk)}\n\n"
                        await asyncio.sleep(1.0 / tokens_per_s)
                    yield "data: [DONE]\n\n"
                finally:
                    state["inflight"] -= 1

            return StreamingResponse(events(), media_type="text/event-stream")
    else:
        raise ValueError(f"unknown role {role}")
    return app


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", required=True, choices=["asr", "llm", "tts"])
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--base", type=float, default=0.1, help="seconds of latency with nothing else in flight")
    p.add_argument("--per-inflight", type=float, default=0.01, help="extra seconds per request in flight")
    p.add_argument("--tokens-per-s", type=float, default=40.0)
    p.add_argument("--p-escalate", type=float, default=0.2, help="router says ESCALATE (70B)")
    p.add_argument("--p-expert", type=float, default=0.1, help="router says EXPERT (turbo)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chars-per-s", type=float, default=15.0, help="TTS speaking rate; ~15 is natural speech")
    a = p.parse_args()
    app = make_app(a.role, a.base, a.per_inflight, a.tokens_per_s, a.p_escalate, a.seed, a.chars_per_s, a.p_expert)
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
