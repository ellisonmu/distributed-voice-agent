"""Builds the dataset that teaches the 8B to route (ANSWER / ESCALATE / EXPERT) and to say a
natural filler line, by distilling from the 70B. Stages, each resumable, all under --work-dir:

  retrieve   CPU only. RAG passages for every question in the fine-tuning split.
  answers    Plain answers from one tier (run once with --tier small against the 8B server,
             once with --tier large against the 70B server).
  label      Against the 70B: judge both answers against MedQuAD's reference, label the
             question, write a filler line for escalated questions, and reject filler that
             contains a medical claim.
  assemble   No server. Writes train.jsonl / val.jsonl in the format training/train_router.py reads.

Labels: the 8B's answer is correct -> ANSWER; otherwise the 70B's is correct -> ESCALATE;
neither is correct -> EXPERT (the question needs more than the 70B). Turbo is not run here,
which would cost a 4-node job; the EXPERT label means "beyond the 70B", not "the 405B got it right".

ANSWER targets are the 8B's own correct answers (short and spoken, unlike MedQuAD's long
reference text), so training sharpens routing without changing how the 8B talks."""
from __future__ import annotations

import argparse
import asyncio
import json
import random
from dataclasses import asdict
from pathlib import Path

import httpx

from voiceagent.analysis.judge import filler_has_medical_claim, judge_answer
from voiceagent.llm_batch import chat, read_jsonl, run_resumable
from voiceagent.prompts import ROUTER, ROUTER_GENERATED_FILLER, build_messages
from voiceagent.rag import Passage

FILLER_GEN_SYSTEM = (
    "Write one short, natural sentence that a phone assistant says while it looks up the answer to the "
    "caller's question. It may name the topic, but it must not contain any medical information: no facts "
    "about conditions, symptoms, drugs, treatments, doses or risks. Vary the wording. Reply with the sentence only."
)


def _passages(item: dict) -> list[Passage]:
    return [Passage(**p) for p in item["passages"]]


def retrieve(split: str, work: Path, index_dir: str | None, k: int, limit: int | None, seed: int) -> None:
    questions = read_jsonl(split)
    random.Random(seed).shuffle(questions)
    questions = questions[:limit]
    retriever = None
    if index_dir:
        from voiceagent.rag import FaissRetriever
        retriever = FaissRetriever(index_dir)
    out = work / "retrieved.jsonl"
    work.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for q in questions:
            passages = retriever.search(q["question"], k) if retriever else []
            f.write(json.dumps({"key": q["question_id"], "question": q["question"], "answer": q["answer"],
                                "qtype": q.get("qtype", ""), "passages": [asdict(p) for p in passages]}) + "\n")
    print(f"retrieved passages for {len(questions)} questions -> {out}" + ("" if retriever else " (no index: empty passages)"))


async def answers(work: Path, tier: str, url: str, model: str, concurrency: int, max_tokens: int) -> None:
    items = read_jsonl(work / "retrieved.jsonl")
    async with httpx.AsyncClient(timeout=300) as http:
        async def one(item: dict) -> dict:
            text = await chat(http, url, model, build_messages(item["question"], _passages(item)), max_tokens)
            return {"answer": text.strip()}
        await run_resumable(items, one, work / f"answers_{tier}.jsonl", concurrency, f"answers {tier}")


async def label(work: Path, url: str, model: str, concurrency: int) -> None:
    items = read_jsonl(work / "retrieved.jsonl")
    small = {a["key"]: a["answer"] for a in read_jsonl(work / "answers_small.jsonl")}
    large = {a["key"]: a["answer"] for a in read_jsonl(work / "answers_large.jsonl")}
    items = [it for it in items if it["key"] in small and it["key"] in large]
    async with httpx.AsyncClient(timeout=300) as http:
        async def one(item: dict) -> dict:
            q, ref = item["question"], item["answer"]
            js = await judge_answer(http, url, model, q, ref, small[item["key"]])
            if js["verdict"] == "correct":
                return {"label": "ANSWER", "small_verdict": js["verdict"]}
            jl = await judge_answer(http, url, model, q, ref, large[item["key"]])
            if js["verdict"] is None or jl["verdict"] is None:
                return {"label": None, "small_verdict": js["verdict"], "large_verdict": jl["verdict"]}
            tag = "ESCALATE" if jl["verdict"] == "correct" else "EXPERT"
            filler = None
            for _ in range(3):
                candidate = (await chat(http, url, model, [{"role": "system", "content": FILLER_GEN_SYSTEM},
                                                            {"role": "user", "content": q}], 40, temperature=0.9)).strip()
                if candidate and await filler_has_medical_claim(http, url, model, candidate) is False:
                    filler = candidate
                    break
            return {"label": tag, "small_verdict": js["verdict"], "large_verdict": jl["verdict"], "filler": filler}
        await run_resumable(items, one, work / "labels.jsonl", concurrency, "label")


def assemble(work: Path, val_fraction: float, seed: int) -> None:
    items = {it["key"]: it for it in read_jsonl(work / "retrieved.jsonl")}
    small = {a["key"]: a["answer"] for a in read_jsonl(work / "answers_small.jsonl")}
    rows, dropped = [], 0
    for lab in read_jsonl(work / "labels.jsonl"):
        item = items[lab["key"]]
        if lab["label"] == "ANSWER":
            target = "ANSWER " + small[lab["key"]]
        elif lab["label"] in ("ESCALATE", "EXPERT") and lab.get("filler"):
            target = f"{lab['label']}: {lab['filler']}"
        else:
            dropped += 1
            continue
        messages = build_messages(item["question"], _passages(item), ROUTER + ROUTER_GENERATED_FILLER)
        rows.append({"key": lab["key"], "label": lab["label"], "messages": messages, "target": target})
    random.Random(seed).shuffle(rows)
    n_val = max(1, int(len(rows) * val_fraction)) if rows else 0
    for name, part in (("val", rows[:n_val]), ("train", rows[n_val:])):
        with open(work / f"{name}.jsonl", "w") as f:
            for r in part:
                f.write(json.dumps(r) + "\n")
    counts = {t: sum(r["label"] == t for r in rows) for t in ("ANSWER", "ESCALATE", "EXPERT")}
    print(f"{len(rows)} examples ({counts}); dropped {dropped} without a usable label or filler")
    print(f"wrote {len(rows) - n_val} train / {n_val} val to {work}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["retrieve", "answers", "label", "assemble"])
    p.add_argument("--work-dir", required=True, type=Path)
    p.add_argument("--split", help="retrieve: MedQuAD fine-tuning split, e.g. $DATA_DIR/medquad/finetune.jsonl")
    p.add_argument("--index-dir", help="retrieve: RAG index from scripts/rag/build_index.py")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--limit", type=int, default=4000, help="retrieve: how many questions to use")
    p.add_argument("--tier", choices=["small", "large"], help="answers: which model is being served")
    p.add_argument("--url", help="answers / label: OpenAI-compatible server")
    p.add_argument("--model", help="answers / label: served model name")
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.stage == "retrieve":
        if not a.split:
            p.error("retrieve needs --split")
        retrieve(a.split, a.work_dir, a.index_dir, a.k, a.limit, a.seed)
    elif a.stage == "answers":
        if not (a.tier and a.url and a.model):
            p.error("answers needs --tier, --url and --model")
        asyncio.run(answers(a.work_dir, a.tier, a.url, a.model, a.concurrency, a.max_tokens))
    elif a.stage == "label":
        if not (a.url and a.model):
            p.error("label needs --url and --model (the 70B)")
        asyncio.run(label(a.work_dir, a.url, a.model, a.concurrency))
    else:
        assemble(a.work_dir, a.val_fraction, a.seed)


if __name__ == "__main__":
    main()
