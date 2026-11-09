#!/usr/bin/env bash
# Bring-up of the turbo tier alone, before any 4-node job: the Ray cluster across 2 nodes, the
# 405B (AWQ 4-bit) with TP=4 x PP=2, one test request, and a check of which network NCCL used.
#   sbatch -A "$NERSC_ACCOUNT" slurm/turbo_2node.sbatch
#   TURBO_TP=8 TURBO_PP=1 sbatch -A "$NERSC_ACCOUNT" slurm/turbo_2node.sbatch   # the fallback layout
# Loading ~203 GiB of weights takes a while, hence 1 h on the regular queue (8 GPU-hours).
# Logs: logs/turbo-<jobid>/{ray_head.log, ray_worker.log, llm_turbo.log}
#SBATCH -J voiceagent-turbo
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -c 64
#SBATCH -t 01:00:00
#SBATCH -o logs/slurm-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
export NODE3=${NODES[0]} NODE4=${NODES[1]}
export NCCL_DEBUG=INFO  # vLLM's log then says which transport NCCL picked
RUN_DIR=logs/turbo-$SLURM_JOB_ID
mkdir -p "$RUN_DIR"
PY=$VOICEAGENT_ENV/bin/python
echo "head=$NODE3 worker=$NODE4 TP=${TURBO_TP:-4} PP=${TURBO_PP:-2}"

srun --overlap -N1 -n1 -w "$NODE3" --gpus-per-node=4 bash slurm/turbo_ray.sh head "$RUN_DIR" &
srun --overlap -N1 -n1 -w "$NODE4" --gpus-per-node=4 bash slurm/turbo_ray.sh worker "$RUN_DIR" &
"$PY" -m voiceagent.health --url "http://$NODE3:8200" --timeout 3000

echo "test request:"
"$PY" - "http://$NODE3:8200" "$LLM_TURBO" <<'EOF'
import sys, time, httpx
url, model = sys.argv[1], sys.argv[2]
body = {"model": model, "max_tokens": 60, "temperature": 0,
        "messages": [{"role": "user", "content": "In two sentences, what does warfarin do?"}]}
t0 = time.perf_counter()
r = httpx.post(f"{url}/v1/chat/completions", json=body, timeout=600)
r.raise_for_status()
print(f"  {time.perf_counter() - t0:.2f}s: {r.json()['choices'][0]['message']['content']!r}")
EOF

echo "GPU memory per node (MiB used / total):"
for node in "$NODE3" "$NODE4"; do
  srun --overlap -N1 -n1 -w "$node" nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader \
    | sed "s/^/  $node GPU/"
done
echo "NCCL transport (want NET/OFI or the Slingshot provider for cross-node, not NET/Socket):"
grep -ho "NCCL INFO.*NET/[A-Za-z]*[^ ]*" "$RUN_DIR"/*.log | sort | uniq -c | head -5 || echo "  no NCCL lines found"
kill $(jobs -p) 2>/dev/null || true
