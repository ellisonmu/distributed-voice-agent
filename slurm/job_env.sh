# Settings every job needs on Perlmutter. Tracked in git (unlike env.sh); sourced right after env.sh.

# Compile caches on the node's local disk. vLLM, FlashInfer, Triton and torch.compile lock files in
# their caches, and the home filesystem (where they go by default) doesn't support those locks:
#   OSError: [Errno 524] Unknown error 524   (from fcntl.flock)
export XDG_CACHE_HOME=/tmp/$USER/cache
export VLLM_CACHE_ROOT=$XDG_CACHE_HOME/vllm
export TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton
export TORCHINDUCTOR_CACHE_DIR=$XDG_CACHE_HOME/inductor
export FLASHINFER_WORKSPACE_BASE=$XDG_CACHE_HOME
mkdir -p "$XDG_CACHE_HOME"

# Separate env for the Whisper servers (CUDA 12 libraries); see README setup.
export ASR_ENV=${ASR_ENV:-$SCRATCH/envs/asr}

# Load models from the downloaded copies in $HF_HOME instead of asking Hugging Face; several
# servers starting at once otherwise get rate-limited (HTTP 429). Set HF_HUB_OFFLINE=0 to download.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
