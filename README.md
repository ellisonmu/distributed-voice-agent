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
 - **Llama-3.1-8B-Instruct** https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
 - **Llama-3.3-70b-instruct-awq** https://huggingface.co/casperhansen/llama-3.3-70b-instruct-awq
 - **Meta-Llama-3.1-405B-Instruct-AWQ-INT4** https://huggingface.co/hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4 
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
python3 -m venv .venv && .venv/bin/pip install -e '.[dev,eval]'
.venv/bin/pip install -e '.[train]'                 # optional: torch/transformers/peft for the training tests
.venv/bin/python -m pytest -q                       # every policy and tier end to end, judge, ASR summaries,
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
See [Task 2: Data and RAG](#task-2-data-and-rag) for ownership, source snapshots,
commands and acceptance checks. MedQuAD preparation can proceed alongside cluster
bring-up; caller audio needs a working TTS endpoint from that work.

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

## Task 2: Data and RAG

**Owner: Thomas Wei.** Prepare the reference QA splits, retrieval corpus and caller audio
used by router training and system evaluation. This work does not require the 405B server.
Coordinate with the cluster owner for a TTS endpoint and with the team for the final shared
data location. Personal scratch paths may not be readable by every teammate.

### Deliverables and evidence

| Artifact under `$DATA_DIR` | Purpose | Acceptance evidence |
|---|---|---|
| `medquad/{finetune,dev,eval}.jsonl` | Answered QA pairs, separated by source document | Split audit passes; actual row/document counts, hashes and source revision recorded |
| `rag_index/{index.faiss,chunks.jsonl,meta.json}` | MedlinePlus passages and bge-small retrieval index | FAISS vector count, chunk count and metadata agree; inspect retrieval for 10 dev questions |
| `eval/eval.jsonl` and its `audio/*.wav` | Caller questions with clean/noisy audio | References and question IDs preserved; every relative path resolves; WAVs readable as 16 kHz mono PCM16 |
| `manifests/` and `logs/` | Reproduce the data snapshot and preparation | Source URLs/revisions, SHA-256 hashes, seed, commands, package versions and real execution logs |

Implementation and execution are tracked separately. Thomas ran the split audit's 18 fixture
tests on macOS with Python 3.14.6 and pytest 9.2.0: **18 passed in 0.10 s**. The command was
`.venv/bin/python -m pytest tests/test_validate_medquad.py -v --junitxml=logs/task2-validation.xml`;
the XML log is local and not committed. All real dataset/index/audio acceptance checks remain
pending. These fixture tests do not show that artifacts have been produced on Perlmutter.

The first real MedQuAD preparation ran on Perlmutter with source revision
`577bd37b96c02d1833b2c9eed2de9f96964e96cb` and seed 0. It produced 14,344 answered
QA pairs from 5,371 documents: finetune 10,838/4,052 documents, dev 1,002/404 documents,
and eval 2,504/915 documents. The audit **failed** on six duplicate/overlapping question-ID
checks; it reported no document or normalized-question overlap. Shared focus labels were
51 (finetune/dev), 164 (finetune/eval) and 17 (dev/eval).

The parser had treated upstream `qid` values as globally unique, but different collections
reuse them (for example, [GHR](https://github.com/abachaa/MedQuAD/blob/577bd37b96c02d1833b2c9eed2de9f96964e96cb/3_GHR_QA/0000001.xml)
and [NIDDK](https://github.com/abachaa/MedQuAD/blob/577bd37b96c02d1833b2c9eed2de9f96964e96cb/5_NIDDK_QA/0000001.xml)).
The first fix added `doc_id` as a question-ID namespace. Thomas reran the parser regression and split-audit fixtures on
macOS with Python 3.14.6 and pytest 9.2.0: **24 passed in 0.09 s**. The command was
`.venv/bin/python -m pytest tests/test_prepare_medquad.py tests/test_validate_medquad.py -v --junitxml=logs/task2-id-fix.xml`;
the XML log is local and not committed. The real `medquad-v2` rebuild retained the same
row/document counts and focus warnings, and passed the cross-split ID checks, but its audit
still **failed**: eval contained eight duplicated question-ID values. Some CancerGov XML
files reuse both document IDs and question IDs, for example
[`0000013_2.xml`](https://github.com/abachaa/MedQuAD/blob/577bd37b96c02d1833b2c9eed2de9f96964e96cb/1_CancerGov_QA/0000013_2.xml)
and [`0000013_2_1.xml`](https://github.com/abachaa/MedQuAD/blob/577bd37b96c02d1833b2c9eed2de9f96964e96cb/1_CancerGov_QA/0000013_2_1.xml).

Prepared question IDs now use `<relative XML path>#pair-<1-based XML pair position>`.
They do not rely on upstream IDs being unique. `source_file`, `source_pair_index` and
`source_question_id` retain the source location and original `qid` (null when absent).
Positions are assigned before filtering, so dropping a question does not renumber later
questions. IDs are reproducible for the recorded source revision regardless of checkout
location; editing/reordering source XML can change them. The original `doc_id` grouping
remains unchanged so related XML files with a shared document ID stay in the same split.
Thomas ran the updated parser and split-audit fixtures on macOS with Python 3.14.6 and
pytest 9.2.0: **26 passed in 0.10 s**. The command was
`.venv/bin/python -m pytest tests/test_prepare_medquad.py tests/test_validate_medquad.py -v --junitxml=logs/task2-source-key-fix.xml`;
the XML log is local and not committed.

The real Perlmutter `medquad-v3` rebuild **passed with 0 errors** using the source-file/pair
keys. It retained all 14,344 prepared QA pairs and the same split sizes:

| Split | Valid QA pairs | Document groups |
|---|---:|---:|
| finetune | 10,838 | 4,052 |
| dev | 1,002 | 404 |
| eval | 2,504 | 915 |

The audit found no duplicate question IDs or normalized questions within splits and no
question-ID, question-text or document-group overlap across splits. Shared focus-label
warnings remain 51/164/17 for finetune-dev/finetune-eval/dev-eval. These are integrity checks,
not measurements of ASR accuracy, retrieval relevance or answer quality.

The validated data is at `/pscratch/sd/y/ycw71/voiceagent-data/task2-LaJrsN/medquad-v3/`.
Its audit is `manifests/medquad-validation-v3.json`, with execution logs in
`logs/{prepare,validate}-medquad-v3.log` under the same snapshot root. The code revision
was recorded in `manifests/medquad-v3-code-commit.txt`; the fix pushed for this run was
`371cc32`. The original `medquad/` and `medquad-v2/` are failed attempts retained for evidence;
consumers must explicitly use `medquad-v3/` until a validated handoff layout is established.

Keep both failed audits as evidence; rebuild into a new output directory and use new
log/report paths. Do not mix old caller audio, distillation records or judgments with the
new question IDs.

### A. MedQuAD snapshot and split audit

Use a fresh versioned directory. The preparation and audio scripts can overwrite files in
their output directories, so preserve completed snapshots and use a new path for a rebuild.
The following is intended for the Perlmutter **login node**: lightweight source download and
XML/JSON preparation only. Do not run model embedding or TTS inference there.

```bash
cd "$SCRATCH/distributed-voice-agent"
module load cray-python/3.12.12
source .venv/bin/activate
export HF_HOME="$SCRATCH/hf"
export PIP_CACHE_DIR="$SCRATCH/voiceagent-cache/pip"
export DATA_DIR="$SCRATCH/voiceagent-data/task2-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$DATA_DIR/raw" "$DATA_DIR/manifests" "$DATA_DIR/logs"
printf 'export DATA_DIR=%q\n' "$DATA_DIR" > "$DATA_DIR/paths.sh"
git rev-parse HEAD > "$DATA_DIR/manifests/code-commit.txt"
git status --short > "$DATA_DIR/manifests/code-status.txt"
python -m pip freeze > "$DATA_DIR/manifests/packages.txt"
git clone https://github.com/abachaa/MedQuAD.git "$DATA_DIR/raw/MedQuAD-src"
git -C "$DATA_DIR/raw/MedQuAD-src" rev-parse HEAD > "$DATA_DIR/manifests/medquad-commit.txt"
```

Confirm the source download and recorded revision before preparing data. Reuse the printed
`DATA_DIR` on later logins by sourcing its `paths.sh`; do not recreate a new timestamp each
time you resume the same snapshot. The existing `.venv` is reused rather than reinstalled.

```bash
set -o pipefail
python scripts/data/prepare_medquad.py \
    --medquad-dir "$DATA_DIR/raw/MedQuAD-src" --out-dir "$DATA_DIR/medquad" \
    --seed 0 --eval-size 2500 --dev-size 1000 \
    2>&1 | tee "$DATA_DIR/logs/prepare-medquad.log"
python -m voiceagent.analysis.validate_medquad \
    --data-dir "$DATA_DIR/medquad" \
    --source-commit "$(cat "$DATA_DIR/manifests/medquad-commit.txt")" \
    --out "$DATA_DIR/manifests/medquad-validation.json" \
    2>&1 | tee "$DATA_DIR/logs/validate-medquad.log"
```

The splitter skips empty answers and deduplicates questions. Whole-document assignment can
overshoot the requested dev/eval sizes; record actual counts rather than assuming exact
sizes. The usable answered subset is smaller than the original MedQuAD collection because
some upstream subsets have no answer text. Follow the [upstream source and citation
instructions](https://github.com/abachaa/MedQuAD).

The audit checks nonempty string IDs/questions/answers/document IDs, malformed rows, empty
splits, duplicate question IDs and normalized question text, and document/question overlap
between splits. It writes a JSON report even on validation failure, exits nonzero on errors,
and records split hashes and source/qtype counts. It refuses to replace an existing report
unless `--overwrite` is explicit, and never allows the report to replace a split file.
Stop on failure and inspect the report before using the data for training or evaluation.

Shared disease/topic `focus` labels are reported as warnings. A document-disjoint split
does **not** guarantee topic isolation: multiple source documents can discuss the same
disease. Exact text checks also do not detect paraphrases. Keep this limitation in the report;
do not describe the current split as a topic-held-out benchmark.

### B. MedlinePlus retrieval index

Source preparation completed on Perlmutter: `raw/mplus_topics_2026-11-13.xml` downloaded
successfully (about 29 MB), with SHA-256
`895f6ec17a32ad051db55248656446f164b1d9a47475103bffa79e17288231d2`.
Its source URL, download timestamp and checksum are recorded in
`manifests/medlineplus-{source-url,downloaded-utc,sha256}.txt`; `mplus-paths.sh` restores
`MPLUS_XML`. BGE model files were also cached successfully under `$SCRATCH/hf` at revision
`5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`, recorded in `manifests/bge-snapshot.json`.

GPU execution and the index build subsequently succeeded in allocation `61462525` on
`nid200325`, using an NVIDIA A100-SXM4-80GB, PyTorch `2.15.0+cu132` and CUDA build 13.2.
The CUDA tensor check passed. The builder ran with `--device cuda`, 250-word chunks and
50-word overlap, producing **2,047 vectors of dimension 384**. The saved FAISS vector count,
passage-row count and metadata count agree; the inner-product index type and embedding
dimension checks passed. Evidence is in `logs/rag-build-61462525.log`,
`logs/rag-check-61462525.log` and `manifests/rag-validation.json` under the data snapshot.

Ten dev questions sampled with seed 0 each returned three nonempty passages with finite
scores. Several titles directly matched the query (for example, Volcanoes, Cancer
Chemotherapy and Myelodysplastic Syndromes), while others did not: the Lynch syndrome
query returned Myasthenia Gravis first, and Anonychia congenita returned Ataxia
Telangiectasia first. This establishes a working retrieval pipeline, not adequate relevance.
The saved report retains full passage text for human review. Corpus coverage and relevance
need investigation before the index is accepted; no retrieval-accuracy percentage or
answer-quality improvement has been measured. Similarity scores are not correctness probabilities.

A login-node keyword inspection was saved as `manifests/rag-keyword-coverage.json`:

| Keyword | English XML records mentioning it | Indexed chunks mentioning it |
|---|---:|---:|
| epidermolysis bullosa | 3 | 1 |
| anonychia | 1 | 0 |
| lynch | 5 | 4 |
| rapadilino | 2 | 0 |
| ornithine transcarbamylase | 3 | 0 |
| glomerular | 6 | 1 |
| hemochromatosis | 8 | 4 |

The XML scan includes metadata, link titles and URLs, whereas the current builder indexes
only title + full summary. A raw XML mention therefore does not establish that explanatory
passage text is available to retrieval. Three inspected rare-condition keywords were absent
from indexed passages. Lynch appeared in four chunks but did not occur in the top-three
retrieved titles for its sampled dev question; passage suitability and ranking both need review.
These keyword counts are diagnostic observations, not corpus recall or retrieval accuracy.

Next candidate: add the complete official [MedlinePlus Genetics XML
compendium](https://medlineplus.gov/about/developers/geneticsdatafilesapi/) as another
source, with its own snapshot/checksum. Keep the 2,047-chunk index as the baseline and build
any expanded corpus into a separate directory. Use the full public source and fixed dev
probes to investigate coverage; record its relationship to MedQuAD's source material in any
evaluation.

The Genetics download subsequently succeeded: `raw/ghr-summaries.xml` is about 7.7 MB,
with SHA-256 `ba7afc4590afa2caf2ac907bbabf414f70281da11315a342f63d6e682dc7122c`.
HTTP headers, source URL, timestamp and checksum are saved in `manifests/genetics-*`;
`genetics-paths.sh` restores `GENETICS_XML`. Inspection of this actual file found namespace
`https://medlineplus.gov/download/ghr-summaries-20250602.xsd`, 1,305 health-condition records,
1,507 genes, 24 chromosomes and one mitochondrial-DNA record. These are XML record counts,
not generated passage counts.

`scripts/rag/build_index.py --genetics-xml` now optionally adds all four record types.
The parser handles the XML namespace and both nested and escaped HTML, chunks each
`text-list/text/html` section independently, and retains the section role, source URL,
record ID, source filename and synonyms. The embedding context includes the section title
and up to eight synonyms. Link listings and database pointers are not substituted for
missing narrative text. No MedQuAD questions or reference answers are inputs to this builder.
The standard Health Topic summaries remain included, with their original embedding text
format. Metadata now includes source hashes and per-corpus chunk counts, and the builder
refuses an existing output directory. Thomas ran the Genetics corpus, MedQuAD parser and
split-audit fixture tests on macOS with Python 3.14.6 and pytest 9.2.0:
**47 passed in 0.13 s**. The command was
`.venv/bin/python -m pytest tests/test_rag_corpus.py tests/test_prepare_medquad.py tests/test_validate_medquad.py -v --junitxml=logs/task2-genetics-parser.xml`;
the XML log is local and not committed.

Real XML preparation on Perlmutter with code `a9d4910` subsequently **passed**. Both source
checksums matched the download manifests, and the original 2,047 Health Topic passages
matched the baseline's title/text/URL values in order. The combined candidate contains
**5,689 chunks**: 2,047 Health Topic and 3,642 Genetics chunks. Genetics chunk counts are
2,054 condition, 1,562 gene, 25 chromosome and one mitochondrial-DNA chunk.
Evidence is in `manifests/rag-genetics-preflight.json` and
`logs/rag-genetics-preflight.log`, with the code SHA in
`manifests/rag-genetics-preflight-code-commit.txt`.

Literal keyword chunk counts in the candidate are 15 for epidermolysis bullosa, one for
anonychia, five for lynch, six for rapadilino, zero for ornithine transcarbamylase, two for
glomerular and eight for hemochromatosis. These counts include titles and narrative text;
they do not establish semantic coverage or retrieval quality. Alternative terminology and
remaining coverage gaps still need review.

The combined index subsequently built successfully in allocation `61464498` on `nid200308`,
using the cached BGE model and `--device cuda`. Its integrity checks **passed**:
5,689 vectors of dimension 384, matching passage/metadata/preparation counts, source
manifests, inner-product metric and embedding dimension. Model name and query prefix
match the baseline. The existing ten dev probes were reused without resampling;
full baseline/candidate passages are saved in `manifests/rag-genetics-comparison.json`.
The build log is `logs/rag-genetics-build-61464498.log`.

| Dev query topic | Baseline first title | Candidate first title | Preliminary observation |
|---|---|---|---|
| Epidermolysis bullosa acquisita | Hidradenitis Suppurativa | Epidermolysis bullosa simplex (description) | The requested condition name is absent from candidate top-3; similar names do not establish relevance |
| Volcanoes | Volcanoes | Volcanoes | Same top-3 titles and scores |
| Anonychia congenita | Ataxia Telangiectasia | Anonychia congenita (description) | Exact topic now first; assess symptom support in full passage |
| Lynch syndrome treatments | Myasthenia Gravis | Myasthenia Gravis | Lynch syndrome appears second; first result remains off-topic |
| RAPADILINO inheritance | Tay-Sachs Disease | RAPADILINO syndrome (description) | Exact topic now first; inheritance support still needs full-passage review |
| Ornithine transcarbamylase deficiency | Alpha-1 Antitrypsin Deficiency | Carbamoyl phosphate synthetase I deficiency (description) | Requested condition name absent from candidate top-3 |
| Hereditary hemochromatosis | Hemochromatosis | Hereditary hemochromatosis (description) | More specific topic title first; review inheritance support |
| Glomerular Diseases | Kidney Tests | Polycystic kidney disease (description) | Candidate titles require relevance review for this broad query |
| Cancer Chemotherapy | Cancer Chemotherapy | Cancer Chemotherapy | Same top-3 titles and scores |
| Myelodysplastic Syndromes | Myelodysplastic Syndromes | 5q minus syndrome (description) | Exact topic title moves to second; rank change needs review |

These observations use the returned titles and displayed excerpts, not completed human
judgments of all full passages. The expanded corpus helps some topic matches but does not
establish consistently better retrieval, answer support or answer quality. No accuracy
percentage is reported. Preserve both indexes and this diagnostic sample; complete passage
review before selecting the handoff index or making further retrieval changes.

After fixture tests and real XML preparation pass, build a separate candidate on a
compute node using the same model and chunk settings:

```bash
python scripts/rag/build_index.py --mplus-xml "$MPLUS_XML" \
    --genetics-xml "$GENETICS_XML" --out-dir "$DATA_DIR/rag_index_genetics" \
    --model BAAI/bge-small-en-v1.5 --chunk-words 250 --overlap 50 --device cuda
```

Compare the saved baseline dev probes with the same questions and top-k on the candidate,
checking actual passage relevance and source coverage. Keep eval references out of tuning.
Do not claim improved retrieval or answer quality until the comparison has been run and
reviewed; adding more source text alone is not a measured improvement.

The saved full passages were subsequently inspected; see
[the preliminary passage-support review](docs/task2-retrieval-review.md). Anonychia's
first passage directly supports its symptom question, but the Lynch treatment and
RAPADILINO inheritance questions still lack answering evidence despite matching titles.
The existing Hemochromatosis inheritance evidence remains available but moves from rank 1
to rank 3, and the broad MDS overview moves from rank 1 to rank 2. This is an assistant
inspection of actual output, not a human grade or held-out accuracy measurement. Thomas's
confirmation and broader validation are pending.

Inspection of the original XML then confirmed the missing source fields: RAPADILINO's
`inheritance-pattern-list` contains the publisher label `Autosomal recessive`, and its
related-gene list contains `RECQL4`. Hereditary hemochromatosis has both `Autosomal recessive`
and `Autosomal dominant` labels and five listed genes. The scan returned those two records
and no record with gene symbol `OTC`; this targeted scan does not establish absence under
all possible names or aliases.

The builder now offers **`--include-genetics-metadata`**, default off. It adds separately
labelled source-metadata passages for the publisher's inheritance `memo` labels and
related-gene symbols, preserving source codes, URLs and record IDs. It retains multiple
inheritance labels without assigning them to individual genes or subtypes. Unlabelled
inheritance codes are not translated into inferred medical statements; related genes are
explicitly identified as associations, not specific mutations or variant effects.
Existing narrative passages remain unchanged when this mode is enabled, and the previous
narrative-only mode remains available. Thomas ran the corpus, MedQuAD parser and split-audit
fixtures on macOS with Python 3.14.6 and pytest 9.2.0: **54 passed in 0.14 s**. The command was
`.venv/bin/python -m pytest tests/test_rag_corpus.py tests/test_prepare_medquad.py tests/test_validate_medquad.py -v --junitxml=logs/task2-genetics-metadata.xml`;
the XML log is local and not committed.

Real metadata preparation on Perlmutter with code `a01e129` subsequently **passed**:
8,227 total chunks (2,047 Health Topic and 6,180 Genetics), with 5,689 preserved narrative
chunks, 1,305 inheritance-pattern metadata chunks and 1,233 related-gene metadata chunks.
Both XML source hashes match the previous snapshots. The saved RAPADILINO metadata
includes the publisher's `Autosomal recessive` label and `RECQL4` entry, and the
hemochromatosis metadata retains both inheritance labels and all five listed genes.
Evidence is in `manifests/rag-metadata-preflight.json` and
`logs/rag-metadata-preflight.log`, with the code SHA in
`manifests/rag-metadata-preflight-code-commit.txt`.

The metadata-enabled index subsequently built successfully in allocation `61466521` on
`nid200261`: **8,227 vectors of dimension 384**. Integrity checks passed against preparation
counts, source manifests, metadata mode, model/query prefix, index metric and model
dimension. Evidence is in `logs/rag-metadata-build-61466521.log`,
`logs/rag-metadata-check-61466521.log` and `manifests/rag-metadata-comparison.json`.

Slurm accounting confirmed allocation `61466521` and both job steps **COMPLETED** with
exit code `0:0`. The allocation elapsed time was 00:08:09; index-build step `.0` took
00:00:41 and integrity/retrieval-check step `.1` took 00:00:11. These are job-step wall
times including their setup/loading work, not isolated encoder latency or a system
serving benchmark. The allocation has been released.

The same ten dev questions were compared across all three saved indexes. RAPADILINO's
inheritance question now retrieves the publisher's `Autosomal recessive` metadata first,
and hereditary hemochromatosis retrieves both listed inheritance labels first. The latter
is a list of possible patterns, not a per-gene/subtype mapping. The Lynch treatment query
still lacks answering evidence, and the ornithine transcarbamylase query retrieves other
conditions' related-gene metadata. The epidermolysis acquisita and broad glomerular
queries also retain the earlier limitations. These fields improve support for the two
observed inheritance probes but do not establish overall retrieval or answer accuracy.
The [passage review](docs/task2-retrieval-review.md) records the three-way findings.
Human confirmation and broader validation remain pending.

After validation, use a new output directory for this candidate:

```bash
python scripts/rag/build_index.py --mplus-xml "$MPLUS_XML" \
    --genetics-xml "$GENETICS_XML" --include-genetics-metadata \
    --out-dir "$DATA_DIR/rag_index_genetics_metadata" \
    --model BAAI/bge-small-en-v1.5 --chunk-words 250 --overlap 50 --device cuda
```

Download a dated **Health Topic XML** snapshot from the [official MedlinePlus XML
page](https://medlineplus.gov/xml.html), not the Health Topic Group XML. Record its exact
URL, download date and SHA-256 in `manifests/`. Keep the XML in `raw/`. Choose the snapshot
explicitly: the upstream download listing changes over time.

Cache [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) using the
same `HF_HOME` during login-node setup, recording the downloaded model revision. This public
embedding model is separate from the gated Llama models. Run the embedding/index build in a
**compute-node allocation**, after confirming the input XML and model cache are ready:

```bash
# Compute node only. Set MPLUS_XML to the downloaded Health Topic XML's full path.
python scripts/rag/build_index.py --mplus-xml "$MPLUS_XML" \
    --out-dir "$DATA_DIR/rag_index" --model BAAI/bge-small-en-v1.5 \
    --chunk-words 250 --overlap 50 --device cuda
```

The existing builder keeps English topics, embeds title + passage, normalizes vectors and
uses FAISS inner-product search. Before handoff, verify `index.ntotal` equals the number of
rows in `chunks.jsonl` and `meta.json`'s `chunks`. Record vector dimension and index/model
settings. Inspect top-3 passages for 10 **dev** questions and save question IDs, titles,
scores and relevance notes. Do not tune retrieval using eval reference answers. A successful
index build is infrastructure evidence; it is not a measured answer-quality improvement.

### C. Caller audio and noise

Obtain the TTS endpoint from the cluster owner; model serving/inference belongs on allocated
compute nodes. Start with 10 clean questions in a separate pilot directory, then inspect
their audio and metadata before the full build:

```bash
# With the allocated TTS service reachable; TTS_URL is its http://host:port URL.
python scripts/data/make_caller_audio.py --split "$DATA_DIR/medquad/eval.jsonl" \
    --out-dir "$DATA_DIR/eval-pilot" --tts-url "$TTS_URL" \
    --snrs clean --limit 10 --concurrency 2 --seed 0
```

Agree on a noise corpus/subset before downloading it. [MUSAN](https://www.openslr.org/17/)
is one option; its full archive is about 11 GB and also contains speech/music, so coordinate
reuse and select the `noise/` WAVs rather than treating all files as background noise.
Record the corpus license, source, subset and hashes. Once the pilot passes and `NOISE_DIR`
points to the chosen noise WAVs, run audio preparation from an allocated compute node:

```bash
python scripts/data/make_caller_audio.py --split "$DATA_DIR/medquad/eval.jsonl" \
    --out-dir "$DATA_DIR/eval" --tts-url "$TTS_URL" --noise-dir "$NOISE_DIR" \
    --snrs clean,20,10,5,0 --seed 0
```

The current generator assigns **one** condition per question in round-robin order; it does
not produce all five SNRs for every question. Clean rows have `snr_db: null`. Record actual
counts by injected SNR and voice. Keep these assignments and the generated WAVs fixed across
policy comparisons. A seed alone does not establish byte-identical TTS outputs.

Check every manifest path and WAV format, and compare question IDs/text/reference answers
with the source eval split. Listen to a small sample from every condition. Record the TTS
model/voice settings and noise selection. Audio generation validates the input pipeline;
Whisper WER and system latency still require their own real evaluation runs.

### D. Team handoff and status

Give the team the agreed readable `$DATA_DIR`, artifact counts, validation logs and source
manifest. Set `DATA_DIR` in each consumer's ignored `slurm/env.sh` to that snapshot.
Task 3 consumes `medquad/finetune.jsonl` and `rag_index/`; system evaluation consumes
`eval/eval.jsonl` and the same index. Commit code and documentation, not raw datasets,
environments, caches or model weights. Preserve important artifacts in approved persistent
storage: Perlmutter scratch is temporary storage.

- [x] Local Genetics corpus, MedQuAD parser and split-audit fixture tests: 54 passed, including optional source metadata; command and local log path recorded above.
- [x] Real MedQuAD v3 split audit passes on Perlmutter: 14,344 valid rows, 0 errors; counts and evidence paths recorded above, hashes in the JSON audit.
- [x] Real MedlinePlus index built and integrity checked: 2,047 vectors, dimension 384; ten dev retrieval probes saved.
- [ ] Review corpus coverage and passage relevance, including the mismatched dev examples.
- [ ] Clean pilot checked; noise source and full caller audio checked.
- [ ] Team can read the frozen snapshot and reproduce preparation from its manifest.

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
- **ASR breakdown (E1):** `summarize --asr-out` writes one row per run, ASR model and
  injected noise level (clean / 20 / 10 / 5 / 0 dB). Only successful turns starting in the
  measurement window are included, matching the overall WER. WER is total substitutions,
  deletions and insertions divided by total reference words, rather than a mean of per-turn
  WERs. The CSV includes turn counts, scored turns, reference words, error counts and median
  estimated SNR. Empty references are excluded from scoring; empty transcripts count as
  deletions. Missing noise labels are `unknown`, distinct from clean audio.

```bash
python -m voiceagent.analysis.summarize logs/run-<id>/*_n*.jsonl \
    --out logs/run-<id>/summary.csv --asr-out logs/run-<id>/asr_by_snr.csv
```

Install the `eval` extra for this report. Compare always-small and always-large runs with
the same evaluation set and load: adaptive runs select different audio for each ASR model,
so their rows alone are not a controlled small-vs-large comparison. Grouping uses injected
SNR for evaluation; the live scheduler continues to use blind estimated SNR. Real E1 runs
and validation against estimated SNR are still needed to choose the switching threshold.

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

**2. Data and RAG — Thomas Wei** (step 2; [detailed plan](#task-2-data-and-rag)). Done when
the reference splits, index and caller audio pass the documented checks and are shared.
- [ ] MedQuAD splits and integrity audit; record source revision and actual counts
- [ ] MedlinePlus FAISS index and dev retrieval review
- [ ] Noise selection, clean audio pilot, full caller audio and manifest checks
- [ ] Reproducible artifact manifest and team handoff

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
      (`summarize --asr-out`) to set the ASR switch threshold from real runs
- [ ] E4 main sweep, then E2, E3, E5, E6
- [ ] Judge every run; hand-check 50 rows (kappa); figures
- [ ] E5 "small LLM 1 vs. 2 replicas" still needs a config with fewer `llm_small` URLs

**6. Demo and report.**
- [ ] Demo: the four calls from `docs/pipeline-diagram.html` (easy → 8B; medium → filler + 70B;
      expert → filler + 405B; noisy → large ASR). Still to build: a script that stitches a
      turn's filler and answer audio into one WAV at their real play times.
- [ ] Report and slides
