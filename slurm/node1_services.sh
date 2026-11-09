#!/usr/bin/env bash
# Node 1: ASR, TTS and small-LLM replicas. Runs as one srun step and stays up until killed.
#   GPU0:      all ASR. Whisper small (~0.5 GB fp16) and large-v3 (~3 GB) are tiny next to 40 GB,
#              and each turn runs ASR once, so one GPU's compute covers it; 2 replicas of each.
#   GPU1..3:   one small LLM + one Kokoro TTS each. Every turn hits the small LLM (it routes),
#              so it gets the GPUs ASR doesn't need.
# With ROUTER_LORA set (an adapter dir from training/train_router.py), vLLM also serves the
# fine-tuned router under the name "router"; set LLM_ROUTER=router and FILLER_MODE=generated.
set -euo pipefail
RUN_DIR=$1
source slurm/env.sh
PY=$VOICEAGENT_ENV/bin/python
VLLM=$VLLM_ENV/bin/vllm
LORA_ARGS=()
if [ -n "${ROUTER_LORA:-}" ]; then
  LORA_ARGS=(--enable-lora --lora-modules "router=$ROUTER_LORA" --max-lora-rank 16)
fi

for i in 0 1; do
  CUDA_VISIBLE_DEVICES=0 "$PY" -m servers.asr_server --model small --port $((9001 + 2 * i)) >"$RUN_DIR/asr_small_$i.log" 2>&1 &
  CUDA_VISIBLE_DEVICES=0 "$PY" -m servers.asr_server --model large-v3 --port $((9002 + 2 * i)) >"$RUN_DIR/asr_large_$i.log" 2>&1 &
done

for i in 0 1 2; do
  gpu=$((1 + i))
  CUDA_VISIBLE_DEVICES=$gpu "$PY" -m servers.tts_server --port $((9011 + i)) >"$RUN_DIR/tts_$i.log" 2>&1 &
  # 0.75 of the GPU for vLLM (8B weights ~16 GB + KV cache) leaves room for Kokoro on the same GPU.
  CUDA_VISIBLE_DEVICES=$gpu "$VLLM" serve "$LLM_SMALL" --port $((8001 + i)) "${LORA_ARGS[@]}" \
    --gpu-memory-utilization 0.75 --max-model-len 8192 >"$RUN_DIR/llm_small_$i.log" 2>&1 &
done

wait
