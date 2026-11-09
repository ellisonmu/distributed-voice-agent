"""One row per run (policy, N): latency percentiles, deadline hit rates, throughput, routing
rates and ASR WER. Then capacity per policy: the largest N whose p95 TTFA meets each deadline."""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

from voiceagent.analysis.common import load_run, measured_turns

DEADLINES = (0.6, 1.0, 1.2)


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9' ]", " ", text.lower()).split())


def _wer(refs: list[str], hyps: list[str]) -> float | None:
    try:
        import jiwer
    except ImportError:
        return None
    pairs = [(_normalize(r), _normalize(h)) for r, h in zip(refs, hyps) if _normalize(r)]
    if not pairs:
        return None
    return float(jiwer.wer([r for r, _ in pairs], [h for _, h in pairs]))


def _pct(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def summarize_run(meta: dict, records: list[dict], judgments: dict[str, dict] | None = None) -> dict:
    turns = measured_turns(meta, records)
    ok = [r for r in turns if r["error"] is None]
    ttfa = [r["ttfa"] for r in ok if r["ttfa"] is not None]
    ttsa = [r["ttsa"] for r in ok if r["ttsa"] is not None]
    window_min = (meta["duration_s"] - meta["warmup_s"]) / 60
    row = {
        "policy": meta["policy"],
        "n": meta["n_callers"],
        "turns": len(ok),
        "errors": len(turns) - len(ok),
        "repeats": sum(r["repeat"] for r in ok),
        "turns_per_min": len(ok) / window_min if window_min > 0 else None,
    }
    for name, values in (("ttfa", ttfa), ("ttsa", ttsa)):
        for q in (50, 95, 99):
            row[f"{name}_p{q}"] = _pct(values, q)
    for d in DEADLINES:
        row[f"ttfa_le_{d}"] = (sum(v <= d for v in ttfa) / len(ok)) if ok else None
    row["stalled"] = (sum(r["stall_count"] > 0 for r in ok) / len(ok)) if ok else None
    row["stall_s_p95"] = _pct([r["stall_total_s"] for r in ok], 95)
    row["escalated"] = (sum(r["route"].startswith("escalate") for r in ok) / len(ok)) if ok else None
    row["hedged"] = (sum(r["route"] == "escalate_hedge" for r in ok) / len(ok)) if ok else None
    for tier in ("small", "large", "turbo"):
        row[f"llm_{tier}"] = (sum(r["llm_model"] == tier for r in ok) / len(ok)) if ok else None
    row["asr_large"] = (sum(r["asr_model"] == "large" for r in ok) / len(ok)) if ok else None
    row["wer"] = _wer([r["question_text"] for r in ok], [r["transcript"] for r in ok])
    graded = [j for j in (judgments or {}).get(meta["run_id"], {}).values() if j["score"] is not None]
    row["judged"] = len(graded)
    row["quality"] = float(np.mean([j["score"] for j in graded])) if graded else None
    row["unsafe"] = (sum(bool(j["unsafe"]) for j in graded) / len(graded)) if graded else None
    fillers = [j["filler_medical_claim"] for j in graded if j.get("filler_medical_claim") is not None]
    row["filler_claims"] = (sum(fillers) / len(fillers)) if fillers else None
    return row


def load_judgments(path: str) -> dict[str, dict[str, dict]]:
    """run_id -> turn key -> judgment."""
    from voiceagent.llm_batch import read_jsonl

    out: dict[str, dict[str, dict]] = {}
    for j in read_jsonl(path):
        out.setdefault(j["key"].split("/")[0], {})[j["key"]] = j
    return out


def capacity(rows: list[dict]) -> dict[str, dict[float, int | None]]:
    out: dict[str, dict[float, int | None]] = {}
    for policy in sorted({r["policy"] for r in rows}):
        runs = sorted((r for r in rows if r["policy"] == policy), key=lambda r: r["n"])
        out[policy] = {}
        for d in DEADLINES:
            meeting = [r["n"] for r in runs if r["ttfa_p95"] is not None and r["ttfa_p95"] <= d]
            out[policy][d] = max(meeting) if meeting else None
    return out


def _fmt(v) -> str:
    if v is None:
        return "-"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+", help="run .jsonl files written by voiceagent.loadgen")
    p.add_argument("--out", help="write the table as CSV here")
    p.add_argument("--judgments", help="output of `voiceagent.analysis.judge grade`, adds quality columns")
    a = p.parse_args()
    paths = [x for x in a.runs if not x.endswith(".meta.json")]
    judgments = load_judgments(a.judgments) if a.judgments else None
    rows = sorted((summarize_run(*load_run(x), judgments) for x in paths), key=lambda r: (r["policy"], r["n"]))
    cols = ["policy", "n", "turns", "errors", "turns_per_min", "ttfa_p50", "ttfa_p95", "ttfa_p99",
            "ttsa_p50", "ttsa_p95", "ttfa_le_1.0", "stalled", "llm_small", "llm_large", "llm_turbo", "hedged",
            "asr_large", "wer", "quality", "unsafe"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    print("\nCapacity (largest N with p95 TTFA <= deadline):")
    for policy, caps in capacity(rows).items():
        print(f"  {policy}: " + ", ".join(f"{d}s -> {_fmt(n)}" for d, n in caps.items()))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
