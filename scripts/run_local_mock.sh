#!/usr/bin/env bash
# Full pipeline on a laptop: mock servers, a fake eval set, a short sweep, and the summary.
# Usage: scripts/run_local_mock.sh [duration_s]   (needs: pip install -e '.[dev]')
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
DURATION=${1:-20}
OUT=logs/local-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"

pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null || true; }
trap cleanup EXIT

start() { "$PY" -m servers.mock "$@" >>"$OUT/mock.log" 2>&1 & pids+=($!); }
start --role asr --port 9101 --base 0.05 --per-inflight 0.005
start --role asr --port 9103 --base 0.05 --per-inflight 0.005
start --role asr --port 9102 --base 0.25 --per-inflight 0.02
# Speech plays 10x faster than real so a short smoke test still completes many turns per caller.
start --role tts --port 9111 --base 0.08 --per-inflight 0.01 --chars-per-s 150
start --role tts --port 9112 --base 0.08 --per-inflight 0.01 --chars-per-s 150
start --role llm --port 9201 --base 0.15 --per-inflight 0.01 --tokens-per-s 60 --seed 1
start --role llm --port 9202 --base 0.15 --per-inflight 0.01 --tokens-per-s 60 --seed 2
start --role llm --port 9301 --base 0.60 --per-inflight 0.03 --tokens-per-s 25
start --role llm --port 9401 --base 1.00 --per-inflight 0.05 --tokens-per-s 15

"$PY" scripts/data/make_mock_eval.py --out-dir data/mock_eval
"$PY" -m voiceagent.health --config configs/local_mock.yaml --timeout 30

for policy in always_small always_large always_turbo adaptive adaptive_no_filler; do
  for n in 1 8 32; do
    "$PY" -m voiceagent.loadgen --config configs/local_mock.yaml --eval data/mock_eval/eval.jsonl \
      --n "$n" --policy "$policy" --duration "$DURATION" --warmup 2 --out "$OUT/${policy}_n${n}.jsonl"
  done
done

"$PY" -m voiceagent.analysis.summarize "$OUT"/*_n*.jsonl --out "$OUT/summary.csv"
"$PY" -m voiceagent.analysis.calibrate "$OUT"/always_*.jsonl --out "$OUT/latency.json"
