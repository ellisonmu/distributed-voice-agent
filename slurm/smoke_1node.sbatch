#!/usr/bin/env bash
# First contact with Perlmutter: node 1's services only (Whisper x4, 8B x3, Kokoro x3), a short
# load test, and a GPU memory trace to check the node 1 packing. Fits the debug queue.
#   sbatch -A "$NERSC_ACCOUNT" slurm/smoke_1node.sbatch
# Uses $DATA_DIR/eval/eval.jsonl if it exists; otherwise it synthesizes 12 test questions with
# Kokoro (clean audio, or clean / 10 dB / 0 dB if NOISE_DIR points at a folder of noise WAVs).
#SBATCH -J voiceagent-smoke
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -c 64
#SBATCH -t 00:30:00
#SBATCH -o logs/slurm-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/env.sh
export NODE1=$(hostname)
RUN_DIR=logs/smoke-$SLURM_JOB_ID
mkdir -p "$RUN_DIR"
PY=$VOICEAGENT_ENV/bin/python
CONFIG=configs/cluster_1node.yaml

nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -l 5 \
  >"$RUN_DIR/gpu.csv" &
srun --overlap -N1 -n1 --gpus-per-node=4 bash slurm/node1_services.sh "$RUN_DIR" &
"$PY" -m voiceagent.health --config "$CONFIG" --timeout 1200

EVAL=$DATA_DIR/eval/eval.jsonl
if [ ! -f "$EVAL" ]; then
  SNRS=clean
  NOISE_ARGS=()
  if [ -n "${NOISE_DIR:-}" ]; then SNRS=clean,10,0; NOISE_ARGS=(--noise-dir "$NOISE_DIR"); fi
  "$PY" scripts/data/make_caller_audio.py --split scripts/data/smoke_questions.jsonl --out-dir "$RUN_DIR/eval" \
    --tts-url "http://$NODE1:9011" "http://$NODE1:9012" "http://$NODE1:9013" --snrs "$SNRS" "${NOISE_ARGS[@]}"
  EVAL=$RUN_DIR/eval/eval.jsonl
fi

"$PY" -m voiceagent.loadgen --config "$CONFIG" --eval "$EVAL" --n 1 --policy always_small \
  --duration 60 --warmup 10 --out "$RUN_DIR/always_small_n1.jsonl"
"$PY" -m voiceagent.loadgen --config "$CONFIG" --eval "$EVAL" --n 8 --policy adaptive \
  --duration 90 --warmup 15 --out "$RUN_DIR/adaptive_n8.jsonl"
"$PY" -m voiceagent.analysis.summarize "$RUN_DIR"/*_n*.jsonl --out "$RUN_DIR/summary.csv"

echo "peak GPU memory (MiB used / total) and peak utilization per GPU:"
"$PY" - "$RUN_DIR/gpu.csv" <<'EOF'
import csv, sys
peak = {}
for row in csv.reader(open(sys.argv[1])):
    i, used, total, util = row[1].strip(), int(row[2]), int(row[3]), int(row[4])
    u, t, p = peak.get(i, (0, total, 0))
    peak[i] = (max(u, used), total, max(p, util))
for i, (u, t, p) in sorted(peak.items()):
    print(f"  GPU{i}: {u} / {t} MiB, peak util {p}%")
EOF
echo "first error lines, if any:"
grep -h '"error": "' "$RUN_DIR"/*_n*.jsonl | head -3 || echo "  none"
kill $(jobs -p) 2>/dev/null || true
