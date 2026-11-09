#!/usr/bin/env bash
# Node 2: the large LLM, tensor-parallel across the node's 4 GPUs over NVLink.
set -euo pipefail
RUN_DIR=$1
source slurm/env.sh
TP=${LARGE_TP:-4}

exec "$VLLM_ENV/bin/vllm" serve "$LLM_LARGE" --port 8100 --tensor-parallel-size "$TP" \
  --gpu-memory-utilization 0.90 --max-model-len 8192 >"$RUN_DIR/llm_large.log" 2>&1
