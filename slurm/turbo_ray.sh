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
source slurm/job_env.sh
# Cross-node NCCL over Slingshot through NERSC's NCCL OFI plugin (the settings `module load nccl`
# applies). Without it NCCL falls back to TCP ("NET/Socket" in the log). Set before `ray start` so
# the vLLM workers Ray spawns inherit it. TURBO_NCCL_SOCKET=1 keeps TCP, for the E5 comparison.
NERSC_NCCL=${NERSC_NCCL:-/global/common/software/nersc9/nccl/2.29.2-cu13}
if [ "${TURBO_NCCL_SOCKET:-0}" != 1 ] && [ -d "$NERSC_NCCL/plugin/lib" ]; then
  export LD_LIBRARY_PATH=$NERSC_NCCL/plugin/lib:${LD_LIBRARY_PATH:-}
  export NCCL_NET="AWS Libfabric" NCCL_SOCKET_IFNAME=hsn NCCL_CROSS_NIC=2 NCCL_NET_GDR_LEVEL=PHB
  export FI_CXI_DISABLE_HOST_REGISTER=1 FI_CXI_RDZV_THRESHOLD=0
fi
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
  # The 405B AWQ checkpoint's generation_config.json stops only on <|end_of_text|> (128001), not on
  # <|eot_id|> (128009), which ends every Llama 3 chat reply; without it the model keeps generating
  # to max_tokens. Give vLLM a generation config with the same stop tokens as the 8B and 70B.
  GEN_CFG=$RUN_DIR/turbo_generation_config
  mkdir -p "$GEN_CFG"
  echo '{"bos_token_id": 128000, "eos_token_id": [128001, 128008, 128009]}' >"$GEN_CFG/generation_config.json"
  exec "$VLLM_ENV/bin/vllm" serve "$LLM_TURBO" --port 8200 --generation-config "$GEN_CFG" \
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
