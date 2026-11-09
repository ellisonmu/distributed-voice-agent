"""Offline, non-streaming calls to an OpenAI-compatible server (vLLM) for batch jobs: the
judge and the fine-tuning dataset builder. Results go to JSONL files keyed by `key`, and a
rerun skips keys that are already there, so a job killed by the Slurm time limit resumes."""
from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Awaitable, Callable, Iterable

import httpx

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_object(text: str) -> dict | None:
    m = _JSON_OBJECT.search(text or "")
    if not m:
        return None
    try:
        value = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


async def chat(http: httpx.AsyncClient, url: str, model: str, messages: list[dict], max_tokens: int,
               temperature: float = 0.0, json_mode: bool = False) -> str:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    r = await http.post(f"{url.rstrip('/')}/v1/chat/completions", json=body)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def done_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path) as f:
        return {json.loads(line)["key"] for line in f if line.strip()}


async def run_resumable(items: Iterable[dict], work: Callable[[dict], Awaitable[dict]], out: Path,
                        concurrency: int, label: str) -> int:
    """Runs `work` on every item whose key isn't in `out` yet and appends each result as it
    finishes. Returns how many items were processed this time."""
    out.parent.mkdir(parents=True, exist_ok=True)
    skip = done_keys(out)
    todo = [it for it in items if it["key"] not in skip]
    sem = asyncio.Semaphore(concurrency)
    done = 0
    with open(out, "a") as f:
        async def one(item: dict) -> None:
            nonlocal done
            async with sem:
                result = await work(item)
            f.write(json.dumps({"key": item["key"], **result}) + "\n")
            f.flush()
            done += 1
            if done % 200 == 0 or done == len(todo):
                print(f"{label}: {done}/{len(todo)} (skipped {len(skip)} already done)", file=sys.stderr, flush=True)

        await asyncio.gather(*(one(it) for it in todo))
    return done


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
