#!/usr/bin/env bash
# Builds the router + filler fine-tuning set on one node (training/build_distill_set.py):
#   1. RAG passages for LIMIT fine-tuning questions (CPU)
#   2. 8B answers (one vLLM on one GPU)
#   3. 70B answers, then labels and filler lines judged by the 70B (vLLM, TP=4)
#   4. train.jsonl / val.jsonl in $DATA_DIR/distill
#   LIMIT=4000 sbatch -A "$NERSC_ACCOUNT" slurm/distill.sbatch
# Every stage is resumable, so a job that hits the time limit can just be resubmitted.
#SBATCH -J voiceagent-distill
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -c 64
#SBATCH -t 02:00:00
#SBATCH -o logs/slurm-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
PY=$VOICEAGENT_ENV/bin/python
WORK=$DATA_DIR/distill
LOGS=logs/distill-$SLURM_JOB_ID
mkdir -p "$LOGS"
HOST=$(hostname)
BUILD=("$PY" training/build_distill_set.py)

if [ ! -f "$WORK/retrieved.jsonl" ]; then
  "${BUILD[@]}" retrieve --work-dir "$WORK" --split "$DATA_DIR/medquad/finetune.jsonl" \
    --index-dir "$DATA_DIR/rag_index" --limit "${LIMIT:-4000}"
fi

CUDA_VISIBLE_DEVICES=0 "$VLLM_ENV/bin/vllm" serve "$LLM_SMALL" --port 8001 --max-model-len 8192 \
  >"$LOGS/llm_small.log" 2>&1 &
SMALL_PID=$!
"$PY" -m voiceagent.health --url "http://$HOST:8001" --timeout 1200
"${BUILD[@]}" answers --work-dir "$WORK" --tier small --url "http://$HOST:8001" --model "$LLM_SMALL"
kill "$SMALL_PID"; wait "$SMALL_PID" 2>/dev/null || true
sleep 10  # let the GPU memory free before the 70B starts

bash slurm/node2_llm.sh "$LOGS" &
LARGE_PID=$!
"$PY" -m voiceagent.health --url "http://$HOST:8100" --timeout 1800
"${BUILD[@]}" answers --work-dir "$WORK" --tier large --url "http://$HOST:8100" --model "$LLM_LARGE"
"${BUILD[@]}" label --work-dir "$WORK" --url "http://$HOST:8100" --model "$LLM_LARGE"
kill "$LARGE_PID" 2>/dev/null || true

"${BUILD[@]}" assemble --work-dir "$WORK"
