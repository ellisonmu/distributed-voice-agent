#!/usr/bin/env bash
# 8-GPU layout (nodes 1-2). Submit from the repo root:
#   sbatch -A "$NERSC_ACCOUNT" slurm/run_8gpu.sbatch
# Override the sweep with env vars, e.g.
#   SWEEP_N="1 8 32" SWEEP_POLICIES="always_small always_large" DURATION=120 sbatch -A ... slurm/run_8gpu.sbatch
#SBATCH -J voiceagent-8gpu
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -c 64
#SBATCH -t 02:00:00
#SBATCH -o logs/slurm-%j.out
cd "$SLURM_SUBMIT_DIR"
exec bash slurm/sweep.sh
