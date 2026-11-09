# Distributed voice agent (Team Stoat)

A medical-information voice agent that serves many concurrent callers on 16 Perlmutter
GPUs. Per turn it picks a speech-recognition model, one of three LLM tiers and whether to play
filler, so that the first audio reaches the caller within the deadline (~1 s). 

```
caller WAV ─► blind SNR estimate ─► ASR small|large ─► RAG (FAISS, CPU) ─► 8B: ANSWER | ESCALATE | EXPERT
                                                                              │ ANSWER   → 8B answers               (node 1)
                                                                              │ ESCALATE → filler now + 70B answer  (node 2, TP=4)
                                                                              │ EXPERT   → filler now + 405B answer (nodes 3–4, TP=4 × PP=2)
                                                                              │ fallback when a tier is backed up: 405B → 70B → hedged 8B answer
                              every LLM sentence ─► TTS as soon as it is complete ─► caller
```

Animated version: open [docs/pipeline-diagram.html](docs/pipeline-diagram.html) in a browser.

## Resources
 - **Llama-3.1-8B** https://huggingface.co/meta-llama/Llama-3.1-8B
 - **Llama-3.3-70B** https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
 - **Llama-3.1-405B** https://huggingface.co/meta-llama/Llama-3.1-405B
 - **MedQuAD** https://github.com/abachaa/MedQuAD/tree/master

## Layout

| Path | What | Stream |
|---|---|---|
| `voiceagent/records.py` | **The interface**: one JSON log line per turn. Everything downstream only reads this. | all |
| `voiceagent/orchestrator.py` | One turn end to end: SNR → ASR → RAG → router → filler → streaming TTS | C |
| `voiceagent/scheduler.py` | ASR choice and the turbo → 70B → hedge fallback chain, from a fitted latency model | C |
| `voiceagent/clients.py` | HTTP clients + replica pools (least-in-flight routing = our data-parallel router) | C |
| `voiceagent/player.py` | Sentence-level TTS and simulated playback timeline per turn | C |
| `voiceagent/loadgen.py` | Closed-loop load test: N callers, next question only after the reply finishes playing | C |
| `voiceagent/analysis/` | `summarize` (p50/p95/p99, deadline hit rate, capacity, WER) and `calibrate` (E1 latency fit) | C |
| `voiceagent/audio/` | WAV I/O, WADA-SNR blind estimator, noise mixing | C / B |
| `servers/` | `asr_server.py` (faster-whisper), `tts_server.py` (Kokoro), `mock.py` (fakes for local dev) | A |
| `slurm/` | Perlmutter launch: node 1 services, node 2 70B with TP=4, sweep driver | A |
| `slurm/turbo_ray.sh`, `turbo_2node.sbatch`, `run_16gpu.sbatch` | Nodes 3–4: Ray cluster across both nodes, Llama 3.1 405B (4-bit) with TP=4 × PP=2 | D |
| `scripts/data/` | MedQuAD splits, caller audio synthesis + noise, mock eval set | B |
| `scripts/rag/build_index.py` | MedlinePlus → chunks → bge-small embeddings → FAISS | B |
| `voiceagent/analysis/judge.py` | Offline LLM judge (70B vs. MedQuAD reference), filler medical-claim check, hand-check sample + kappa | B |
| `voiceagent/analysis/plots.py` | Report figures: latency vs N, quality vs latency, routing mix, stalls, DDP vs FSDP | C |
| `training/build_distill_set.py` | Router + filler fine-tuning set, distilled from the 70B | Training |
| `training/train_router.py` | LoRA fine-tune of the 8B, `--strategy ddp` or `fsdp` (same code path) | Training |
| `configs/` | `local_mock.yaml` (laptop), `cluster_1node.yaml` (first test), `cluster_8gpu.yaml`, `cluster_16gpu.yaml` | — |
| `docs/pipeline-diagram.html` | Animated block diagram of the 16-GPU system | — |

## Run locally (no GPU)

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pip install -e '.[train]'                 # optional: torch/transformers/peft for the training tests
.venv/bin/python -m pytest -q                       # 43 tests, ~1 min: every policy and tier end to end, judge,
                                                    # dataset builder, DDP and FSDP on CPU, plots, configs
PYTHON=.venv/bin/python scripts/run_local_mock.sh   # mock servers + sweep + summary table + calibration
```

The mock servers speak the same HTTP API as the real ones, so anything that works here
should only need a config change on Perlmutter.

## Run on Perlmutter, in order

Every job is `sbatch -A $NERSC_ACCOUNT slurm/<job>`; settings go in front as environment
variables. Each step below says what it needs from the steps before it. A 4-node hour costs
16 GPU-hours, so debug on 1–2 nodes and keep 4-node runs short.

### 1. One-time setup
```bash
cp slurm/env.example.sh slurm/env.sh                # set account, paths, model names
module load python
python -m venv $SCRATCH/envs/voiceagent && $SCRATCH/envs/voiceagent/bin/pip install -e '.[serve,asr,tts,rag,eval,train,plots]'
python -m venv $SCRATCH/envs/vllm && $SCRATCH/envs/vllm/bin/pip install vllm
huggingface-cli login                               # Llama weights are gated: accept the license on HF first
mkdir -p logs
```
Download the 405B (~203 GiB) to `$HF_HOME` on `$SCRATCH` early; it takes a while.

### 2. Data (needs: setup)
```bash
git clone https://github.com/abachaa/MedQuAD $DATA_DIR/MedQuAD-src
python scripts/data/prepare_medquad.py --medquad-dir $DATA_DIR/MedQuAD-src --out-dir $DATA_DIR/medquad
python scripts/rag/build_index.py --mplus-xml <mplus_topics_*.xml> --out-dir $DATA_DIR/rag_index
# needs a running TTS server (servers/tts_server.py) and a noise set such as MUSAN's noise/ folder
python scripts/data/make_caller_audio.py --split $DATA_DIR/medquad/eval.jsonl --out-dir $DATA_DIR/eval \
    --tts-url http://<node>:9011 --noise-dir <noise wavs>
```

### 3. Bring-up (needs: setup; data optional)
```bash
sbatch -A $NERSC_ACCOUNT slurm/smoke_1node.sbatch     # 1 node, debug queue: node 1 services, short load test,
                                                      # GPU memory trace (escalations go to an 8B stand-in)
sbatch -A $NERSC_ACCOUNT slurm/turbo_2node.sbatch     # 2 nodes: Ray + 405B (TP=4 x PP=2), one test request,
                                                      # which network NCCL used
SWEEP_N="1 8" DURATION=60 sbatch -A $NERSC_ACCOUNT slurm/run_8gpu.sbatch    # nodes 1-2: 8B + 70B
SWEEP_N="1 8" DURATION=60 sbatch -A $NERSC_ACCOUNT slurm/run_16gpu.sbatch   # first full 16-GPU job
```
If the 405B misbehaves with TP=4 × PP=2, rerun `turbo_2node.sbatch` with `TURBO_TP=8 TURBO_PP=1`.

### 4. Router dataset and fine-tune (needs: data; one node per job)
```bash
LIMIT=4000 sbatch -A $NERSC_ACCOUNT slurm/distill.sbatch   # -> $DATA_DIR/distill/{train,val}.jsonl
STRATEGY=ddp  sbatch -A $NERSC_ACCOUNT slurm/train.sbatch  # -> $SCRATCH/router-ddp/
STRATEGY=fsdp sbatch -A $NERSC_ACCOUNT slurm/train.sbatch  # -> $SCRATCH/router-fsdp/  (E5 training comparison)
```
Both jobs are resumable: if one hits its time limit, resubmit the same command. The dataset
stages skip what they have done; training restarts from its last checkpoint (every 50 steps).
Check the label counts `distill` prints before training (see Limitations below).

### 5. Full 16-GPU evaluation (needs: bring-up, data; E3/E4 need the fine-tuned router)
```bash
# E1 calibration: fits the scheduler's latency model from the full system
SWEEP_POLICIES="always_small always_large always_turbo" SWEEP_N="1 8 32" \
  sbatch -A $NERSC_ACCOUNT slurm/run_16gpu.sbatch
python -m voiceagent.analysis.calibrate logs/run-<id>/always_*.jsonl --out configs/latency_16gpu.json
# then set `latency: configs/latency_16gpu.json` in configs/cluster_16gpu.yaml

# The fine-tuned router is used when these three are set; leave them out for the base 8B router
export ROUTER_LORA=$SCRATCH/router-fsdp/adapter LLM_ROUTER=router FILLER_MODE=generated

# E4 main result: every policy, N from 1 to 64
SWEEP_N="1 8 32 64" DURATION=120 sbatch -A $NERSC_ACCOUNT slurm/run_16gpu.sbatch

# E2 RAG off (compare with the adaptive rows of E4)
RAG_KIND=none SWEEP_POLICIES=adaptive SWEEP_N="8 32" sbatch -A $NERSC_ACCOUNT slurm/run_16gpu.sbatch

# E3 router and filler: run adaptive three ways
#   base 8B router + filler bank:         unset ROUTER_LORA LLM_ROUTER FILLER_MODE first
#   fine-tuned router + filler bank:      FILLER_MODE=bank
#   fine-tuned router + generated filler: the E4 run already has it
SWEEP_POLICIES=adaptive SWEEP_N="8 32" sbatch -A $NERSC_ACCOUNT slurm/run_16gpu.sbatch

# E5 parallelism layouts (each compared with the default E4 run)
LARGE_TP=2                SWEEP_POLICIES=adaptive SWEEP_N="1 32" sbatch -A $NERSC_ACCOUNT slurm/run_16gpu.sbatch
TURBO_TP=8 TURBO_PP=1     SWEEP_POLICIES=adaptive SWEEP_N="1 32" sbatch -A $NERSC_ACCOUNT slurm/run_16gpu.sbatch
TURBO_TP=2 TURBO_PP=4     SWEEP_POLICIES=adaptive SWEEP_N="1 32" sbatch -A $NERSC_ACCOUNT slurm/run_16gpu.sbatch

# E6 scaling: the same adaptive sweep on 8 GPUs (no 405B)
SWEEP_POLICIES="adaptive" SWEEP_N="1 8 32 64" DURATION=120 sbatch -A $NERSC_ACCOUNT slurm/run_8gpu.sbatch
```
For `TURBO_TP=2 TURBO_PP=4`, check in `llm_turbo.log` that stages 1–2 landed on node 3 and
stages 3–4 on node 4. Each `run-<id>/summary.csv` holds that job's latency table.

### 6. Quality and figures (needs: evaluation runs)
```bash
RUNS=logs/run-<id> sbatch -A $NERSC_ACCOUNT slurm/judge.sbatch   # judgments.jsonl, summary.csv, handcheck.csv
# fill the `human` column of handcheck.csv, then:
python -m voiceagent.analysis.judge agreement logs/run-<id>/handcheck.csv
python -m voiceagent.analysis.plots --summary logs/run-<id>/summary.csv --out-dir figures \
    --train-runs $SCRATCH/router-ddp $SCRATCH/router-fsdp
```

## What the evaluation measures

**N is the control variable.** N callers talk at once in a closed loop: each caller asks its
next question only after the previous reply has finished playing. Each run sweeps
N = 1, 8, 32, 64 per policy, and every metric below is computed per (policy, N). Turns from
the first `WARMUP` seconds are dropped.

### Accuracy

| Metric | How | Compared against | Where |
|---|---|---|---|
| **ASR accuracy** | Word error rate (WER) of the Whisper transcript | The question text the caller audio was synthesized from (exact ground truth) | `summarize`, every run |
| **Answer correctness** | Offline 70B judge after the run: correct = 1, partial = 0.5, incorrect = 0; the mean is `quality` | MedQuAD's reference answer | `judge.sbatch`, then `summarize --judgments` |
| **Judge agreement** | Cohen's kappa between the judge and a human on 50 random rows | Hand grades in `handcheck.csv` | `judge agreement` |
| **Safety** | Share of answers the judge flags as unsafe; filler lines that make a medical claim | — | judge output |
| **Routing accuracy** | Router's first word (ANSWER / ESCALATE / EXPERT) vs. the label | `val.jsonl` from the router dataset | script still to build |

### Latency

| Metric | Meaning |
|---|---|
| **TTFA** (p50 / p95 / p99) | Time to first audio: when the caller hears anything, filler included |
| **TTSA** | Time to first answer audio: when the real answer starts after any filler |
| **Deadline hit rate** | Share of turns with first audio within 0.6 / 1.0 / 1.2 s |
| **Stalls** | Gaps over 0.15 s between sentences during playback: count, total and longest |
| **Routing mix** | Which tier answered each turn, and how often the turbo → 70B → hedge fallback fired |

### Significance of these measures

- **Latency vs. N:** p95 TTFA per policy. **Capacity** is the largest N whose p95 TTFA meets
  the 1 s deadline; it is read off this curve, not run as a separate experiment.
- **Quality vs. latency** (the main figure): one point per (policy, N), judge quality against
  p95 TTFA.
- **Routing mix vs. N:** how the adaptive policy shifts between tiers as load rises.
- **Gap:** WER is currently one number per run. Setting the ASR switch threshold (E1) needs
  WER by noise level (clean / 20 / 10 / 5 / 0 dB) for Whisper small and large separately; each
  turn's log already has its SNR and ASR model, so this is a small addition to `summarize`.

## How the router is trained

The 8B decides each turn's tier from the first word it writes: `ANSWER` (it answers itself),
`ESCALATE` (the 70B answers) or `EXPERT` (the 405B answers). With `FILLER_MODE=generated`, it
also says a filler line after the escalation word. A LoRA fine-tune of the 8B teaches both.

### The dataset: `training/build_distill_set.py`

MedQuAD has no difficulty labels, so we define difficulty by outcome: **a question is hard for
a model if that model answers it wrong.** MedQuAD's reference answers make this checkable. The
script runs in four resumable stages, all writing to `--work-dir` (`slurm/distill.sbatch` runs them all):

| Stage | Needs | What it does |
|---|---|---|
| `retrieve` | CPU | Samples `--limit` questions (default 4,000) from the fine-tuning split and attaches the top-k RAG passages, as the live system would |
| `answers` | 8B, then 70B server | Each model answers every question with the same prompt and passages as in serving |
| `label` | 70B server | The 70B judges both answers against the MedQuAD reference and labels the question (rules below). For escalated questions, it also writes a filler line, keeping one only if a second 70B check finds no medical claim in it (up to 3 tries) |
| `assemble` | nothing | Writes `train.jsonl` / `val.jsonl` (5% val): each row is the router prompt plus a target output |

| 8B answer | 70B answer | Label | Training target |
|---|---|---|---|
| correct | — | `ANSWER` | `ANSWER <the 8B's own answer>` |
| wrong / partial | correct | `ESCALATE` | `ESCALATE: <filler line>` |
| wrong / partial | wrong / partial | `EXPERT` | `EXPERT: <filler line>` |

The `ANSWER` targets are the 8B's own correct answers, not MedQuAD's long reference text, so
training changes when the 8B escalates but not how it talks. Escalated questions with no
filler line that passes the check, and questions the judge couldn't grade, are dropped.

### The fine-tune: `training/train_router.py`

- **LoRA** (rank 16, alpha 32, dropout 0.05) on every attention and MLP projection of
  Llama 3.1 8B Instruct; the base weights stay frozen in bf16.
- **Loss only on the target.** Prompt tokens are masked out, so the model learns what to say,
  not to repeat the prompt (`training/data.py`).
- AdamW, lr 2e-4, linear warmup then decay, batch 1 per GPU × 8 accumulation steps,
  gradient checkpointing, sequences up to 2,048 tokens, 1 epoch.
- **DDP vs FSDP (E5).** The same script, seed and data run under either strategy with
  `--strategy`, on 4 GPUs via `torchrun`:
  - **DDP:** each GPU holds the whole 8B; gradients are all-reduced, but only on the last
    accumulation step (`no_sync` on the others).
  - **FSDP:** each GPU holds a quarter of each layer, gathered just before the layer runs.
    The wrap policy gives the LoRA weights their own shards, since FSDP can't mix frozen and
    trainable weights in one shard.

  Both log loss, tokens/s and peak GPU memory to `metrics.jsonl` and `summary.json`; the
  E5 figure comes from those.
- **Output:** a LoRA adapter that vLLM serves next to the base 8B (`--enable-lora`). Only the
  routing call uses the adapter; the baseline policies and hedged answers use the base 8B.

### Limitations

- **`EXPERT` means "beyond the 70B", not "the 405B gets it right."** Labeling never runs the
  405B, which would need the 4-node job. We accept this because the 405B is the largest, most
  capable model we have, so sending the questions the 70B misses to it is the best available
  choice. We don't measure whether it actually answers them correctly.
- **The labels may be lopsided.** MedQuAD questions are mostly templated ("What are the
  symptoms of X?"), so the 8B with RAG may label most of them `ANSWER`. `assemble` prints the
  label counts, so check them before training.
- **The labels inherit the judge's errors.** The 70B also judges the system, and its agreement
  with a human (Cohen's kappa on `handcheck.csv`) bounds how much to trust both.

## Status

Done (tested locally against mocks): orchestrator, three-tier router, filler v1 + generated
filler path, turbo → 70B → hedge fallback chain, scheduler, replica routing, closed-loop load
generator, playback stall measurement, summary, calibration, SNR estimator, data and RAG
scripts, offline judge, distillation set builder, DDP/FSDP training with checkpoint/resume
(2 CPU processes on a tiny model; DDP and FSDP give identical losses, and a killed and resumed
run matches an uninterrupted one), plots, real ASR/TTS servers, all Slurm scripts.

Not yet run on Perlmutter: everything under `servers/` and `slurm/`, and training at 8B scale.
Expect the first job to need fixes (module versions, ports, memory packing on node 1), and the
first 2-node job to need work on Ray and cross-node NCCL over Slingshot.

## Tasks

Six workstreams, each owned end to end by one person: running the jobs, fixing what breaks,
and writing that part of the report. Commands are in the numbered steps above.

```
1 Bring-up ──┐
             ├─► 5 System evaluation ─► 6 Demo + report
2 Data ──────┼─► 3 Router dataset ─► 4 Router fine-tune ─┘
```

**1. Cluster bring-up** (steps 1 and 3). Done when a short `run_16gpu.sbatch` finishes with
all three tiers answering.
- [ ] Perlmutter envs; model downloads (8B, 70B AWQ, 405B AWQ ~203 GiB)
- [ ] `smoke_1node.sbatch`: node 1 packing fits in memory, one caller completes a turn
- [ ] `turbo_2node.sbatch`: 405B serves with TP=4 × PP=2; NCCL uses Slingshot, not sockets
- [ ] `run_8gpu.sbatch`, then the first short `run_16gpu.sbatch`

**2. Data and RAG** (step 2). Done when `$DATA_DIR/eval/eval.jsonl` and the RAG index exist.
- [ ] MedQuAD splits; MedlinePlus FAISS index
- [ ] Pick a noise set; synthesize and noise-mix caller audio

**3. Router dataset** (step 4, `distill.sbatch`). Done when `train.jsonl` / `val.jsonl` exist
with usable label counts.
- [ ] Run it; if `ANSWER` dominates, rebalance in `assemble` (subsample, or raise `--limit`)
- [ ] Spot-check 20 generated filler lines for medical claims

**4. Router fine-tune** (step 4, `train.sbatch`). Done when the adapter is served in a 16-GPU run.
- [ ] DDP and FSDP runs on the same data (the E5 training comparison: memory, tokens/s)
- [ ] Routing accuracy on `val.jsonl`: fine-tuned vs. the base 8B with the prompted router
      (still to build: a short script that sends each val prompt and compares the first word)

**5. System evaluation** (steps 5 and 6). Done when every figure in the report exists.
- [ ] E1 calibration and fitting the latency model; WER by noise level per Whisper size
      (still to build in `summarize`) to set the ASR switch threshold
- [ ] E4 main sweep, then E2, E3, E5, E6
- [ ] Judge every run; hand-check 50 rows (kappa); figures
- [ ] E5 "small LLM 1 vs. 2 replicas" still needs a config with fewer `llm_small` URLs

**6. Demo and report.**
- [ ] Demo: the four calls from `docs/pipeline-diagram.html` (easy → 8B; medium → filler + 70B;
      expert → filler + 405B; noisy → large ASR). Still to build: a script that stitches a
      turn's filler and answer audio into one WAV at their real play times.
- [ ] Report and slides
