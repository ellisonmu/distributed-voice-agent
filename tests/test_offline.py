"""The judge and the distillation-set builder, run against a mock OpenAI-compatible server."""
import asyncio
import csv
import json

import pytest
import uvicorn

from servers.mock import make_app
from training import build_distill_set as distill
from voiceagent.analysis import judge
from voiceagent.analysis.common import load_run
from voiceagent.analysis.summarize import load_judgments, summarize_run
from voiceagent.llm_batch import parse_json_object, read_jsonl

PORT = 9571
URL = f"http://127.0.0.1:{PORT}"


async def with_llm(coro):
    server = uvicorn.Server(uvicorn.Config(make_app("llm", 0.0, 0.0, 1000, 0.0, seed=3), host="127.0.0.1",
                                           port=PORT, log_level="warning"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.02)
    try:
        return await coro
    finally:
        server.should_exit = True
        await task


def write_run(tmp_path, n_turns=12):
    run = tmp_path / "adaptive_n4.jsonl"
    meta = {"run_id": "adaptive_n4", "policy": "adaptive", "n_callers": 4, "duration_s": 100, "warmup_s": 0}
    run.with_suffix(".meta.json").write_text(json.dumps(meta))
    with open(run, "w") as f:
        for i in range(n_turns):
            f.write(json.dumps({
                "run_id": "adaptive_n4", "policy": "adaptive", "n_callers": 4, "caller_id": i % 4, "turn_idx": i // 4,
                "question_id": f"q{i}", "question_text": "What is asthma?", "transcript": "What is asthma?",
                "answer": "Asthma is a long-term lung condition.", "filler_text": "One moment." if i % 2 else "",
                "llm_model": "small", "route": "answer", "asr_model": "small", "repeat": False, "error": None,
                "t_start": 1.0 + i, "ttfa": 0.6, "ttsa": 0.6, "stall_count": 0, "stall_total_s": 0.0, "stages": [],
            }) + "\n")
    evals = tmp_path / "eval.jsonl"
    evals.write_text("".join(json.dumps({"question_id": f"q{i}", "question": "What is asthma?",
                                         "answer": "Asthma is a chronic lung disease."}) + "\n" for i in range(n_turns)))
    return run, evals


def test_parse_json_object_tolerates_chatter():
    assert parse_json_object('Sure! {"verdict": "correct", "unsafe": false} done') == {"verdict": "correct", "unsafe": False}
    assert parse_json_object("no json here") is None


def test_judge_grades_resumes_and_feeds_summary(tmp_path):
    run, evals = write_run(tmp_path)
    out = tmp_path / "judgments.jsonl"
    asyncio.run(with_llm(judge.grade([str(run)], str(evals), URL, "judge", out, concurrency=4)))
    graded = read_jsonl(out)
    assert len(graded) == 12 and all(g["verdict"] in judge.VERDICTS for g in graded)
    assert all((g["filler_medical_claim"] is False) == bool(i % 2) or g["filler_medical_claim"] is None
               for i, g in enumerate(sorted(graded, key=lambda g: g["question_id"])))
    asyncio.run(with_llm(judge.grade([str(run)], str(evals), URL, "judge", out, concurrency=4)))
    assert len(read_jsonl(out)) == 12  # rerun skips turns already graded

    row = summarize_run(*load_run(run), load_judgments(str(out)))
    assert row["judged"] == 12 and 0 <= row["quality"] <= 1 and row["unsafe"] == 0 and row["filler_claims"] == 0

    csv_path = tmp_path / "check.csv"
    judge.sample(str(out), [str(run)], str(evals), 5, str(csv_path), seed=0)
    rows = list(csv.DictReader(open(csv_path)))
    assert len(rows) == 5 and all(r["judge"] in judge.VERDICTS for r in rows)


def test_cohen_kappa():
    assert judge.cohen_kappa(["correct", "incorrect"] * 5, ["correct", "incorrect"] * 5) == 1.0
    # agree on 6 of 10, chance agreement 0.5 -> (0.6 - 0.5) / (1 - 0.5)
    assert judge.cohen_kappa(["correct"] * 5 + ["incorrect"] * 5, ["correct", "incorrect"] * 5) == pytest.approx(0.2)


def test_distill_pipeline(tmp_path):
    split = tmp_path / "finetune.jsonl"
    split.write_text("".join(json.dumps({"question_id": f"q{i}", "question": f"Question {i}?",
                                         "answer": "Reference answer.", "qtype": "information"}) + "\n" for i in range(30)))
    work = tmp_path / "distill"
    distill.retrieve(str(split), work, None, k=3, limit=20, seed=0)

    async def build():
        await distill.answers(work, "small", URL, "small", 8, 50)
        await distill.answers(work, "large", URL, "large", 8, 50)
        await distill.label(work, URL, "large", 8)
    asyncio.run(with_llm(build()))
    distill.assemble(work, val_fraction=0.1, seed=0)

    rows = read_jsonl(work / "train.jsonl") + read_jsonl(work / "val.jsonl")
    assert len(rows) == 20 and {r["label"] for r in rows} <= {"ANSWER", "ESCALATE", "EXPERT"}
    for r in rows:
        assert r["target"].startswith(r["label"])
        assert "ANSWER, ESCALATE or EXPERT" in r["messages"][0]["content"]  # same router prompt as serving
        if r["label"] != "ANSWER":
            assert r["target"].split(": ", 1)[1] in ("Let me look into that for you.", "One moment while I check on that.")
