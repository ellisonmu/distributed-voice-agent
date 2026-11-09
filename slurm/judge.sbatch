#!/usr/bin/env bash
# Offline judge on one node: serve the 70B (TP=4) and grade every turn of a finished sweep.
#   RUNS=logs/run-123 sbatch -A "$NERSC_ACCOUNT" slurm/judge.sbatch
# Writes $RUNS/judgments.jsonl and $RUNS/summary.csv (with quality columns). Resumable: if the
# time limit hits, submit again and it skips turns already graded.
#SBATCH -J voiceagent-judge
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -c 64
#SBATCH -t 01:00:00
#SBATCH -o logs/slurm-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
: "${RUNS:?set RUNS to a sweep directory, e.g. RUNS=logs/run-123}"
PY=$VOICEAGENT_ENV/bin/python
URL=http://$(hostname):8100

bash slurm/node2_llm.sh "$RUNS" &
"$PY" -m voiceagent.health --url "$URL" --timeout 1800
"$PY" -m voiceagent.analysis.judge grade "$RUNS"/*_n*.jsonl --eval "$DATA_DIR/eval/eval.jsonl" \
  --url "$URL" --model "$LLM_LARGE" --out "$RUNS/judgments.jsonl"
"$PY" -m voiceagent.analysis.summarize "$RUNS"/*_n*.jsonl --judgments "$RUNS/judgments.jsonl" --out "$RUNS/summary.csv"
"$PY" -m voiceagent.analysis.judge sample "$RUNS"/*_n*.jsonl --judgments "$RUNS/judgments.jsonl" \
  --eval "$DATA_DIR/eval/eval.jsonl" --n 50 --out "$RUNS/handcheck.csv"
kill $(jobs -p) 2>/dev/null || true
