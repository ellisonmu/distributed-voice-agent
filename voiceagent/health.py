"""Blocks until every server in a config answers /health, or fails after --timeout."""
from __future__ import annotations

import argparse
import asyncio
import time

import httpx

from voiceagent.config import load_config


async def wait_healthy(urls: list[str], timeout_s: float, interval_s: float = 10.0) -> None:
    pending = set(urls)
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(timeout=5.0) as http:
        while pending:
            for url in sorted(pending):
                try:
                    if (await http.get(f"{url}/health")).status_code == 200:
                        pending.discard(url)
                        print(f"healthy: {url}", flush=True)
                except httpx.HTTPError:
                    pass
            if pending and time.monotonic() > deadline:
                raise SystemExit(f"not healthy after {timeout_s:.0f}s: {sorted(pending)}")
            if pending:
                await asyncio.sleep(interval_s)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", help="wait for every server in this config")
    p.add_argument("--url", nargs="*", default=[], help="or for these server URLs")
    p.add_argument("--timeout", type=float, default=1800.0)
    a = p.parse_args()
    urls = (load_config(a.config).all_urls() if a.config else []) + a.url
    if not urls:
        p.error("give --config or --url")
    asyncio.run(wait_healthy(urls, a.timeout))


if __name__ == "__main__":
    main()
