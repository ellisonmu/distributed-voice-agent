#!/usr/bin/env bash
# LoRA fine-tune of the 8B router on 4 GPUs with DDP or FSDP. Run both for the E5 comparison:
#   STRATEGY=ddp  sbatch -A "$NERSC_ACCOUNT" slurm/train.sbatch
#   STRATEGY=fsdp sbatch -A "$NERSC_ACCOUNT" slurm/train.sbatch
# Same data, seed and hyperparameters; only the parallelism differs. Extra flags go in
# TRAIN_ARGS, e.g. TRAIN_ARGS="--max-steps 200" for a short comparison run.
# Output: $SCRATCH/router-<strategy>/{metrics.jsonl,summary.json,adapter/}. The run checkpoints every
# 50 steps, so if the job hits its time limit, resubmit the same command and it resumes there.
# For a fresh run instead, pick a new directory: OUT=$SCRATCH/router-fsdp-v2
#SBATCH -J voiceagent-train
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -c 64
#SBATCH -t 01:30:00
#SBATCH -o logs/slurm-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
STRATEGY=${STRATEGY:-fsdp}
OUT=${OUT:-$SCRATCH/router-$STRATEGY}

# shellcheck disable=SC2086
"$VOICEAGENT_ENV/bin/torchrun" --standalone --nproc_per_node=4 training/train_router.py \
  --strategy "$STRATEGY" --model "$LLM_SMALL" --data-dir "$DATA_DIR/distill" --out "$OUT" ${TRAIN_ARGS:-}
echo "adapter: $OUT/adapter  (serve it with ROUTER_LORA=$OUT/adapter LLM_ROUTER=router FILLER_MODE=generated)"
