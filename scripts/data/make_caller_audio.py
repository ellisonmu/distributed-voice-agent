"""Turns a question split into caller audio: synthesize each question with the TTS server
in a randomly chosen voice, resample to 16 kHz, and mix in background noise at one SNR
condition (assigned round-robin so conditions are balanced). Writes <out-dir>/eval.jsonl,
which is the --eval input of voiceagent.loadgen.

Noise: any directory of 16-bit mono WAVs, e.g. the `noise/` part of MUSAN."""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

import httpx
import numpy as np

from voiceagent.audio.mix import mix_at_snr
from voiceagent.audio.wav import read_wav, resample, write_wav

VOICES = ["af_heart", "af_bella", "af_nicole", "am_adam", "am_michael"]


def load_noise(noise_dir: Path) -> list[np.ndarray]:
    clips = []
    for path in sorted(noise_dir.rglob("*.wav")):
        x, sr = read_wav(path.read_bytes())
        if np.any(x):
            clips.append(resample(x, sr, 16000))
    if not clips:
        raise SystemExit(f"no usable 16-bit WAV noise files under {noise_dir}")
    return clips


async def main_async(a: argparse.Namespace) -> None:
    with open(a.split) as f:
        questions = [json.loads(line) for line in f][: a.limit]
    conditions = [None if c == "clean" else float(c) for c in a.snrs.split(",")]
    noise = load_noise(a.noise_dir) if any(c is not None for c in conditions) else []
    audio_dir = a.out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    pyrng = random.Random(a.seed)
    urls = a.tts_url
    sem = asyncio.Semaphore(a.concurrency)
    rows: list[dict | None] = [None] * len(questions)

    async with httpx.AsyncClient(timeout=120) as http:
        async def one(i: int, q: dict) -> None:
            voice = pyrng.choice(VOICES)
            snr = conditions[i % len(conditions)]
            async with sem:
                r = await http.post(f"{urls[i % len(urls)]}/synthesize", json={"text": q["question"], "voice": voice})
                r.raise_for_status()
            x, sr = read_wav(r.content)
            x = resample(x, sr, 16000)
            if snr is not None:
                x = mix_at_snr(x, noise[int(rng.integers(len(noise)))], snr, rng)
            name = f"{i:05d}.wav"
            (audio_dir / name).write_bytes(write_wav(x, 16000))
            rows[i] = {**q, "audio_path": f"audio/{name}", "snr_db": snr, "voice": voice}

        await asyncio.gather(*(one(i, q) for i, q in enumerate(questions)))

    with open(a.out_dir / "eval.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} clips to {a.out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", required=True, type=Path, help="e.g. data/medquad/eval.jsonl")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--tts-url", required=True, nargs="+", help="one or more TTS server URLs")
    p.add_argument("--noise-dir", type=Path)
    p.add_argument("--snrs", default="clean,20,10,5,0", help="comma list; 'clean' means no noise")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.noise_dir is None and a.snrs != "clean":
        p.error("--noise-dir is required unless --snrs clean")
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
