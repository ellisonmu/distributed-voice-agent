#!/usr/bin/env bash
# Shared body of run_8gpu.sbatch and run_16gpu.sbatch: start every server, wait for health,
# run the closed-loop sweep, summarize. 2 nodes = 8-GPU layout, 4 nodes = 16-GPU layout.
set -euo pipefail
source slurm/env.sh

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
export NODE1=${NODES[0]} NODE2=${NODES[1]}
RUN_DIR=logs/run-$SLURM_JOB_ID
mkdir -p "$RUN_DIR"
SWEEP_N=${SWEEP_N:-"1 4 8 16 32 64"}
DURATION=${DURATION:-180}
WARMUP=${WARMUP:-30}
PY=$VOICEAGENT_ENV/bin/python

srun --overlap -N1 -n1 -w "$NODE1" --gpus-per-node=4 bash slurm/node1_services.sh "$RUN_DIR" &
srun --overlap -N1 -n1 -w "$NODE2" --gpus-per-node=4 bash slurm/node2_llm.sh "$RUN_DIR" &

if [ "${#NODES[@]}" -ge 4 ]; then
  export NODE3=${NODES[2]} NODE4=${NODES[3]}
  CONFIG=${CONFIG:-configs/cluster_16gpu.yaml}
  SWEEP_POLICIES=${SWEEP_POLICIES:-"always_small always_large always_turbo adaptive adaptive_no_filler"}
  srun --overlap -N1 -n1 -w "$NODE3" --gpus-per-node=4 bash slurm/turbo_ray.sh head "$RUN_DIR" &
  srun --overlap -N1 -n1 -w "$NODE4" --gpus-per-node=4 bash slurm/turbo_ray.sh worker "$RUN_DIR" &
else
  CONFIG=${CONFIG:-configs/cluster_8gpu.yaml}
  SWEEP_POLICIES=${SWEEP_POLICIES:-"always_small always_large adaptive adaptive_no_filler"}
fi
echo "nodes=${NODES[*]} config=$CONFIG run_dir=$RUN_DIR"

"$PY" -m voiceagent.health --config "$CONFIG" --timeout 2400

for policy in $SWEEP_POLICIES; do
  for n in $SWEEP_N; do
    "$PY" -m voiceagent.loadgen --config "$CONFIG" --eval "$DATA_DIR/eval/eval.jsonl" \
      --n "$n" --policy "$policy" --duration "$DURATION" --warmup "$WARMUP" \
      --out "$RUN_DIR/${policy}_n${n}.jsonl"
  done
done

"$PY" -m voiceagent.analysis.summarize "$RUN_DIR"/*_n*.jsonl --out "$RUN_DIR/summary.csv"
kill $(jobs -p) 2>/dev/null || true
