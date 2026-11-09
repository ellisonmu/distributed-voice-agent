"""Closed-loop load test. N callers start together; each asks a question, waits until the
caller would have heard the end of the reply, waits `think` seconds, then asks the next.
New turns stop starting after `duration` seconds; turns already in flight finish.
Warm-up turns are kept in the log and dropped by the analysis scripts."""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import subprocess
import time
from pathlib import Path

import httpx

from voiceagent.config import POLICIES, load_config
from voiceagent.orchestrator import Orchestrator
from voiceagent.rag import make_retriever
from voiceagent.records import TurnRecord


class QuestionQueue:
    """Draws questions without replacement; if the split runs out, it reshuffles and marks
    repeats so the analysis can see them."""

    def __init__(self, items: list[dict], seed: int):
        self.items, self.rng = items, random.Random(seed)
        self.order: list[dict] = []
        self.passes = -1

    def next(self) -> tuple[dict, bool]:
        if not self.order:
            self.order = self.items[:]
            self.rng.shuffle(self.order)
            self.passes += 1
        return self.order.pop(), self.passes > 0


def load_eval(path: Path) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            item["audio_path"] = str((path.parent / item["audio_path"]).resolve())
            items.append(item)
    if not items:
        raise ValueError(f"no questions in {path}")
    return items


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


async def run(config: str, eval_path: str, n: int, policy: str, duration: float, warmup: float,
              think: float, out: str, seed: int = 0) -> Path:
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    cfg = load_config(config)
    if policy == "always_turbo" and cfg.llm_turbo is None:
        raise ValueError("always_turbo needs an llm_turbo service in the config")
    items = load_eval(Path(eval_path))
    questions = QuestionQueue(items, seed)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = out_path.stem

    t0 = time.perf_counter()

    def clock() -> float:
        return time.perf_counter() - t0

    limits = httpx.Limits(max_connections=2000, max_keepalive_connections=500)
    async with httpx.AsyncClient(timeout=cfg.request_timeout_s, limits=limits) as http:
        orch = Orchestrator(cfg, http, make_retriever(cfg.rag), clock)
        await orch.start()
        t0 = time.perf_counter()
        meta = {
            "run_id": run_id, "policy": policy, "n_callers": n, "duration_s": duration, "warmup_s": warmup,
            "think_s": think, "config": config, "eval": eval_path, "seed": seed,
            "deadline_s": cfg.deadline_s, "filler_mode": cfg.filler_mode,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_commit": _git_commit(),
        }
        out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

        with open(out_path, "w") as log:
            async def caller(caller_id: int) -> None:
                turn = 0
                while clock() < duration:
                    item, repeat = questions.next()
                    rec = TurnRecord(
                        run_id=run_id, policy=policy, n_callers=n, caller_id=caller_id, turn_idx=turn,
                        question_id=item["question_id"], question_text=item["question"],
                        snr_db=item.get("snr_db"), repeat=repeat,
                    )
                    wav = await asyncio.to_thread(Path(item["audio_path"]).read_bytes)
                    try:
                        await orch.handle_turn(rec, wav)
                    except Exception as e:  # one failed turn must not end the run
                        rec.error = repr(e)
                    log.write(json.dumps(rec.to_dict()) + "\n")
                    if rec.error:
                        await asyncio.sleep(1.0)
                    else:
                        await asyncio.sleep(max(0.0, rec.t_playback_end - clock()) + think)
                    turn += 1

            await asyncio.gather(*(caller(i) for i in range(n)))
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--eval", required=True, help="eval jsonl with question_id, question, audio_path, snr_db")
    p.add_argument("--n", type=int, required=True, help="number of concurrent callers")
    p.add_argument("--policy", required=True, choices=POLICIES)
    p.add_argument("--duration", type=float, default=300.0, help="seconds during which new turns start")
    p.add_argument("--warmup", type=float, default=30.0, help="seconds dropped from the start by the analysis")
    p.add_argument("--think", type=float, default=0.0, help="caller pause after hearing a reply")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    path = asyncio.run(run(a.config, a.eval, a.n, a.policy, a.duration, a.warmup, a.think, a.out, a.seed))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
