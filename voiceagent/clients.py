"""Clients for the model servers. Each service is a Pool of replicas; requests go to the
replica with the fewest requests in flight (our data-parallel request router). The
in-flight counts also feed the scheduler's latency predictions."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

import httpx


@dataclass
class Endpoint:
    url: str
    inflight: int = 0
    served: int = 0


class Pool:
    def __init__(self, name: str, urls: list[str]):
        if not urls:
            raise ValueError(f"pool {name} has no replicas")
        self.name = name
        self.endpoints = [Endpoint(u.rstrip("/")) for u in urls]

    @property
    def inflight(self) -> int:
        return sum(e.inflight for e in self.endpoints)

    @asynccontextmanager
    async def acquire(self):
        ep = min(self.endpoints, key=lambda e: (e.inflight, e.served))
        ep.inflight += 1
        ep.served += 1
        try:
            yield ep
        finally:
            ep.inflight -= 1


class ASRClient:
    def __init__(self, pool: Pool, http: httpx.AsyncClient):
        self.pool, self.http = pool, http

    async def transcribe(self, wav: bytes) -> str:
        async with self.pool.acquire() as ep:
            r = await self.http.post(f"{ep.url}/transcribe", content=wav, headers={"Content-Type": "audio/wav"})
            r.raise_for_status()
            return r.json()["text"]


class TTSClient:
    def __init__(self, pool: Pool, http: httpx.AsyncClient):
        self.pool, self.http = pool, http

    async def synthesize(self, text: str, voice: str | None = None) -> tuple[bytes, int]:
        """Returns (wav bytes, in-flight count when the request was sent)."""
        inflight = self.pool.inflight
        body = {"text": text} if voice is None else {"text": text, "voice": voice}
        async with self.pool.acquire() as ep:
            r = await self.http.post(f"{ep.url}/synthesize", json=body)
            r.raise_for_status()
            return r.content, inflight


class LLMClient:
    """Streams from an OpenAI-compatible server (vLLM). Closing the generator early closes
    the HTTP connection, which makes vLLM abort the request and free its batch slot."""

    def __init__(self, pool: Pool, model: str, http: httpx.AsyncClient):
        self.pool, self.model, self.http = pool, model, http

    async def stream(self, messages: list[dict], max_tokens: int, temperature: float = 0.0,
                     model: str | None = None) -> AsyncIterator[str]:
        body = {
            "model": model or self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        async with self.pool.acquire() as ep:
            async with self.http.stream("POST", f"{ep.url}/v1/chat/completions", json=body) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    choices = json.loads(data).get("choices") or []
                    if choices and (delta := choices[0].get("delta", {}).get("content")):
                        yield delta
