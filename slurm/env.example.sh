# Copy to slurm/env.sh (git-ignored) and edit. Sourced by every script in slurm/.
export NERSC_ACCOUNT=m0000_g                          # pass at submit: sbatch -A $NERSC_ACCOUNT ...
export VOICEAGENT_ENV=$SCRATCH/envs/voiceagent       # venv: pip install -e '.[serve,asr,tts,rag,eval,train,plots]'
export VLLM_ENV=$SCRATCH/envs/vllm                   # separate venv: pip install vllm
export HF_HOME=$SCRATCH/hf                           # model weights cache; run `huggingface-cli login` once
export DATA_DIR=$SCRATCH/voiceagent-data             # medquad/, eval/, rag_index/
export LLM_SMALL=meta-llama/Llama-3.1-8B-Instruct
# After fine-tuning, submit with ROUTER_LORA=<adapter dir> LLM_ROUTER=router FILLER_MODE=generated
export LLM_ROUTER=${LLM_ROUTER:-$LLM_SMALL}
export FILLER_MODE=${FILLER_MODE:-bank}
export RAG_KIND=${RAG_KIND:-faiss}   # RAG_KIND=none turns retrieval off (E2)
export LLM_LARGE=casperhansen/llama-3.3-70b-instruct-awq   # any 4-bit AWQ/GPTQ Llama 3.3 70B; check it fits TP=4
export LLM_TURBO=hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4  # ~203 GiB: too big for one node's 160 GB.
# 4-bit weights, fp16 math: runs on A100. (bf16 405B is ~810 GB; FP8 builds need H100-class GPUs.)
# Cross-node NCCL (turbo's PP=2) must go over Slingshot, not TCP. Check NERSC's current guidance
# for NCCL on Perlmutter (the libfabric/OFI plugin) before the first 4-node run; without it,
# the pipeline hop between nodes 3 and 4 is far slower.
