#!/usr/bin/env bash
# 16-GPU layout (nodes 1-4, turbo LLM on nodes 3-4). Submit from the repo root:
#   sbatch -A "$NERSC_ACCOUNT" slurm/run_16gpu.sbatch
# A full sweep (5 policies x 6 N x ~3.5 min) is ~2 h x 16 GPUs = ~32 GPU-hours, so start
# small, e.g.  SWEEP_N="1 16 64" DURATION=120 sbatch -A ... slurm/run_16gpu.sbatch
#SBATCH -J voiceagent-16gpu
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -c 64
#SBATCH -t 02:30:00
#SBATCH -o logs/slurm-%j.out
cd "$SLURM_SUBMIT_DIR"
exec bash slurm/sweep.sh
