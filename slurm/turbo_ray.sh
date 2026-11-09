#!/usr/bin/env bash
# Nodes 3-4: the turbo LLM across two nodes. Usage: turbo_ray.sh head|worker RUN_DIR
#   head (node 3):   starts the Ray head, waits until node 4 joins (8 GPUs), then runs
#                    vLLM with TP=4 inside each node and PP=2 across the two nodes.
#   worker (node 4): joins the Ray cluster and blocks.
# TP stays inside a node because it all-reduces every layer (NVLink is fast enough);
# PP crosses Slingshot because it only sends one activation tensor per stage boundary.
set -euo pipefail
ROLE=$1
RUN_DIR=$2
source slurm/env.sh
RAY=$VLLM_ENV/bin/ray
PORT=6379
TP=${TURBO_TP:-4}
PP=${TURBO_PP:-2}

if [ "$ROLE" = head ]; then
  "$RAY" start --head --port=$PORT --num-gpus=4 >"$RUN_DIR/ray_head.log" 2>&1
  for _ in $(seq 1 120); do
    gpus=$("$VLLM_ENV/bin/python" -c "import ray; ray.init(address='auto', logging_level='ERROR'); print(int(ray.cluster_resources().get('GPU', 0)))" 2>/dev/null || echo 0)
    [ "$gpus" -ge $((TP * PP)) ] && break
    sleep 5
  done
  echo "ray cluster has $gpus GPUs" >>"$RUN_DIR/ray_head.log"
  exec "$VLLM_ENV/bin/vllm" serve "$LLM_TURBO" --port 8200 \
    --tensor-parallel-size "$TP" --pipeline-parallel-size "$PP" --distributed-executor-backend ray \
    --gpu-memory-utilization 0.90 --max-model-len 4096 >"$RUN_DIR/llm_turbo.log" 2>&1
else
  for _ in $(seq 1 60); do
    "$RAY" start --address="$NODE3:$PORT" --num-gpus=4 --block >>"$RUN_DIR/ray_worker.log" 2>&1 && exit 0
    sleep 5
  done
  echo "could not join the Ray head on $NODE3" >&2
  exit 1
fi
