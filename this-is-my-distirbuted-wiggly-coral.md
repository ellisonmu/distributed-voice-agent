# Plan: Latency-aware medical voice agent (Team Stoat final project)

## Context

The mid-term deck proposes a general voice agent that picks a model size per stage (ASR → LLM → TTS) and per call so that many concurrent callers get a first reply within ~1 s, on 8–16 Perlmutter GPUs. Three changes now:

1. **One domain: a medical-information voice agent** (consumer health questions), so quality can be judged against reference answers and RAG has a clear knowledge base.
2. **RAG scaffolding**: every LLM size answers from retrieved medical passages. This also gives a research result: how much RAG closes the small-vs-large quality gap.
3. **"Filler, then hand off"**: when the small LLM judges a question too hard, it speaks a short holding line with no medical claims ("Good question, let me pull up the details on that...") while the large LLM writes the real answer, which is spoken right after. This meets the 1 s first-audio deadline without making anything up. That matters in a medical setting.

**Constraint: about 2 weeks, exams and research competing for time, and teammates may not deliver.**

**The system is the full 16-GPU, three-tier design from the mid-term deck.** Small models on node 1, the 70B on node 2, and the turbo LLM (Llama 3.1 405B, 4-bit) across nodes 3–4 with tensor parallelism inside each node and pipeline parallelism between them. The turbo tier is core to the project's complexity and is not optional. The 8-GPU layout (nodes 1–2) is the first milestone on the way there, because it's where the pipeline gets debugged cheaply, not a separate deliverable. Interfaces are defined early so any piece can be picked up by whoever on the team is free.

Tell the professor about the compressed timeline early.

## Scope

Everything from the mid-term deck is a **must** except the live phone demo, which Perlmutter's batch-job model makes impractical regardless of time.

| Must | Nice (if time allows) | Cut |
|---|---|---|
| **Three LLM tiers on 16 GPUs:** Llama 3.1 8B (node 1), Llama 3.3 70B 4-bit with TP=4 (node 2), Llama 3.1 405B AWQ 4-bit (~203 GiB) with TP=4 inside each of nodes 3–4 and PP=2 across them | Rebalancer: switch nodes 3–4 between one turbo copy and two 70B copies as load shifts | Live phone demo (Perlmutter batch jobs can't take calls; simulated/recorded calls stand in) |
| Whisper small + large-v3, Kokoro TTS; the parallelism techniques in "Distributed techniques" below | Emergency fast-path (keyword → "call emergency services") | |
| RAG over a medical corpus | | |
| Three-level self-assessment router: the small LLM's first output token is `ANSWER` (it answers), `ESCALATE` (70B) or `EXPERT` (turbo). This *is* the difficulty classifier; no separate embedding-based classifier | | |
| Templated filler bank (~20 pre-rendered holding lines) as the v1 baseline, then the small LLM fine-tuned (FSDP) for natural filler as v2 | | |
| **ASR routing by estimated spectral quality** (SNR estimated from the audio itself, e.g. WADA-SNR, not from the known injected noise level) | | |
| Deadline-aware scheduler with fallback chain (turbo → 70B → hedged 8B) vs always-small / always-large / always-turbo | | |
| Load test with N simulated callers, N fixed as a controlled variable per run; latency + quality + capacity, on 8 and 16 GPUs | | |

## System design

### Data
- **Questions with reference answers: MedQuAD** (NIH consumer health QA, ~47k pairs; roughly a third still ship with answers after the copyright removal — verify the exact count on download). That's far more than the ~1k originally planned, so use it in three disjoint splits:
  - **Fine-tuning split** (most of it): training data for the small LLM's router + filler fine-tune.
  - **Router/threshold dev split** (~1k): tuning the `ESCALATE` behavior and the ASR SNR threshold.
  - **Eval split** (~2–3k, stratified easy/hard): the questions the load generator draws from. In closed-loop runs, callers draw without replacement until the split is exhausted, so no question repeats inside a run.
- **Extra eval questions (optional, for out-of-distribution checks):** MedicationQA (~690 drug questions with expert answers) and the TREC LiveQA medical test set (real questions sent to NLM, with reference answers). Both are small but are *real* consumer phrasings rather than MedQuAD's templated ones. HealthSearchQA (3,173 questions) has no reference answers, so it's only usable for latency load, not quality.
- **RAG corpus: MedlinePlus health topic summaries** (public XML download), chunked to about 200–300 words, embedded with a small embedding model (e.g. bge-small or e5-small), and indexed with FAISS on CPU. Retrieval runs once per turn after ASR, and its result is shared by whichever LLM answers.
- **Caller audio:** synthesize every question with TTS in several voices, then mix in background noise at SNR 20 / 10 / 5 / 0 dB using a public noise set. This creates the "noisy audio needs the large ASR" condition, with known ground-truth text for WER. Optionally record about 20 real questions for the demo.

### Pipeline (one turn)
```
caller audio → [ASR small|large] → retrieve top-k passages → [router] ─ easy ─→ small LLM ─┐
                                                                    └ hard ─→ play filler clip now,
                                                                              large LLM ─────┤
                         stream tokens → split into sentences → TTS per sentence → audio ←───┘
```
- **Streaming:** LLM tokens are split into sentences, and each sentence goes to TTS as soon as it's complete. First audio = ASR + retrieval + time to the first LLM sentence + TTS of that sentence.
- **Router (must):** the small LLM's prompt makes its first output token `ANSWER` or `ESCALATE`. On `ANSWER` it keeps generating. On `ESCALATE` the orchestrator plays a filler clip and sends the request to the large LLM. This self-assessment step *is* the difficulty classifier — there's no separate embedding-based classifier scoring the question before the small LLM sees it; that would be a second model to build, tune and profile, which the timeline doesn't support.
- **Filler v1 (must):** a bank of about 20 holding lines, rendered to audio **ahead of time**, so the filler costs about 0 ms of TTS. The lines make no medical claims. This is the baseline for every experiment.
- **Filler v2 (must, after v1 works):** templated lines are a tell. Fine-tune the small LLM (LoRA SFT) so that on `ESCALATE` it says one short, natural, question-specific holding line with no medical claims ("Drug interactions with warfarin are worth getting exactly right, give me a second..."). Training data is distilled: the 70B labels each fine-tuning question as answerable-by-8B or not (using the judge against the reference answer) and writes a filler line for the hard ones; a filter rejects any line containing a medical claim. One fine-tune therefore teaches both the router token and the filler. Its first sentence goes straight to TTS, so it costs one short TTS call instead of ~0 ms; that cost is measured, and v1 stays in as the ablation.

### RAG (no training needed for the core)
RAG here is retrieval plus prompting: embed the transcript, pull the top-k MedlinePlus chunks from FAISS, and put them in the prompt. **None of the three LLMs needs training for this to work.** The only RAG-related training is optional: include retrieved passages in the small LLM's fine-tuning examples (RAFT-style) so it learns to answer *from* the passages. That rides along with the filler fine-tune, so it's not a separate job. Fine-tuning the 70B or the 405B isn't worth the GPU-hours for this project.
- **Max wait:** if the large LLM's predicted completion is beyond a cap (e.g. 4 s, because of load), the scheduler falls back to the small LLM with a hedged answer plus "please check with a clinician".
- **Safety prompt (all sizes):** information only, grounded in the retrieved passages, no diagnosis, and advise emergency services for emergencies.

### Scheduler (deadline-aware, per turn)
- **Inputs:** per-stage latency as a function of concurrent requests (fitted from the E1 calibration run), current in-flight requests per model server (tracked by the orchestrator), the remaining budget against the deadline, and the router's decision.
- **ASR choice (must):** small by default. Large when the audio's *estimated* quality is low and the budget allows — estimated from the signal itself (a blind SNR estimator, e.g. WADA-SNR, run on the incoming audio), not from the known injected noise level used to build the test set. The known level is ground truth for evaluating the estimator and for labeling test conditions, but the router only ever sees the blind estimate, same as it would with a real caller. Threshold tuned on the E1 calibration run (WER vs. estimated SNR for both ASR sizes).
- **LLM choice:** `ANSWER` → small. `ESCALATE` → 70B plus filler if its predicted first sentence lands within the max wait, otherwise small with a hedge. `EXPERT` → turbo plus filler if it fits the max wait, else the 70B if that fits, else small with a hedge. Turbo is the slowest tier, so under load the chain degrades quality step by step instead of missing the deadline.
- **Baselines:** always-small, always-large, always-turbo, and adaptive without filler (ablation).

### GPU layout (8 GPUs, 2 nodes)
- **Node 1:** GPU0 holds all ASR: 2 replicas each of Whisper small (~0.5 GB fp16) and large-v3 (~3 GB). Memory is not the limit on a 40 GB A100; compute is, and ASR runs once per turn for a fraction of a second, so one GPU covers it. GPU1–3 each hold one small-LLM replica (vLLM, with the fine-tuned LoRA) plus one Kokoro replica, since every turn hits the small LLM. The orchestrator, retriever and load generator run on Node 1's CPUs. The first runs check this packing: if GPU0 saturates (ASR latency climbing with N while GPU1–3 sit idle), move a large-v3 replica next to an 8B.
- **Node 2:** large LLM, Llama 3.3 70B 4-bit (AWQ/GPTQ), on vLLM with tensor parallelism across 4 GPUs over NVLink.
- **Serving:** each LLM is a vLLM OpenAI-compatible server; Whisper uses faster-whisper behind a thin HTTP service; TTS is a thin HTTP service. One `sbatch` script starts every server, waits for health checks, runs the load test, then shuts everything down.

### GPU layout, 16 GPUs (the full system)
Nodes 1–2 as above, plus:
- **Nodes 3 + 4:** turbo LLM, Llama 3.1 405B Instruct in AWQ 4-bit (~203 GiB, too big for one node's 160 GB). The original bf16 weights are ~810 GB and don't fit even all 16 GPUs; FP8 builds need H100-class GPUs. AWQ stores weights in 4 bits but does the math in fp16, which A100s run natively. All three tiers are Llama 3.x, so model size is the only thing that changes between them. Tensor-parallel (TP=4) inside each node over NVLink, pipeline-parallel (PP=2) across the two nodes over Slingshot, on a Ray cluster spanning both nodes inside one Slurm job (`slurm/turbo_ray.sh`). TP stays inside a node because it all-reduces every layer; PP crosses nodes because it only sends one activation tensor per stage boundary.
- **Same harness:** turbo is a third pool in the orchestrator and a third tier in the scheduler. `slurm/run_16gpu.sbatch` runs the same sweep with `always_turbo` added.
- **Cross-node NCCL:** the PP hop must use Slingshot through NERSC's NCCL/libfabric setup, not TCP. Check NERSC's current guidance before the first 4-node run.

### Distributed techniques (class requirement)
FSDP and DDP are **training** techniques: both keep a full set of gradients/optimizer state in sync across GPUs. At inference time there are no gradients, and FSDP's per-layer all-gather of weights would be slower than tensor parallelism for serving the 70B. So each technique goes where it actually belongs:

| Technique | Where | What we build / measure |
|---|---|---|
| **Data parallelism (replicas)** | Inference: small ASR, small LLM and TTS run as several replicas on Node 1 | Our own replica router in the orchestrator (least-outstanding-requests). Measured: latency vs. N with 1 vs. 2 replicas of the small LLM |
| **Tensor parallelism** | Inference: 70B with TP=4 over NVLink on Node 2; 405B TP=4 inside each of Nodes 3–4 | Measured: TP=2 vs. TP=4 for the 70B (per-token latency vs. how many concurrent requests fit) |
| **Pipeline parallelism** | Inference: 405B split across Nodes 3–4 over Slingshot (vLLM with a Ray cluster spanning both nodes) | Measured: TP=8 across both nodes vs. TP=4 × PP=2 vs. TP=2 × PP=4, to show why TP stays inside a node and PP crosses nodes, and what extra pipeline stages cost at low load |
| **FSDP** | Training: the small-LLM router + filler fine-tune on 4 GPUs (`torchrun`, PyTorch FSDP + LoRA) | Our own training script. Measured: throughput and memory per GPU vs. DDP |
| **DDP** | Training: same job, as the baseline FSDP is compared against | Shows the memory trade-off: DDP keeps a full copy of the 8B per GPU, FSDP shards it |

vLLM implements TP and PP internally, so for those the work is the launch scripts, the multi-node Ray setup, and the layout comparisons. The parts written from scratch are the DDP/FSDP training script, the replica router and the scheduler.

## Metrics and experiments
- **Latency:** time to first audio (TTFA, which filler can satisfy) **and** time to substantive answer (TTSA), both p50 and p95.
- **Quality:** ASR word error rate (`jiwer`). LLM answer correctness from an **offline LLM judge** (the 70B model, given the reference answer, after the runs, so it never affects latency). Hand-check about 50 judgments to validate the judge.
- **Load model: closed loop with N concurrent callers.** N is the control parameter (swept 1, 4, 8, 16, 32, 64). All N callers start at the same moment. Each caller asks a question, waits until the last TTS clip of the reply has finished, then immediately asks its next question (optionally after a short fixed "think time"). This matches a real phone call: a caller doesn't ask the next question until they've heard the answer. Each run lasts a fixed window (e.g. 5 minutes); the first ~30 s are discarded as warm-up, since the first requests hit cold caches and an artificially synchronized start.
- **What's reported per (N, policy):** TTFA and TTSA as p50 / p95 / p99 over every turn completed in the window (averages hide the tail, which is what the deadline is about); the fraction of turns with TTFA ≤ deadline (0.6 / 1.0 / 1.2 s); turns completed per minute (throughput); and quality on those same turns.
- **Capacity** is read off those curves: the largest N whose p95 TTFA meets the deadline. It isn't a separate experiment.
- **Why the question count matters less now:** in a closed loop, a 5-minute window at N = 64 uses on the order of a few thousand turns, which the eval split covers without repeats.

**We evaluate the system, not the models.** There are no standalone single-model latency runs. The scheduler still needs a latency estimate per model at a given load. That comes from a short **calibration run** of the full system (always-small, always-large and always-turbo policies at a few N), fitted from the per-stage timestamps the system already logs. Per-model quality (WER per SNR for each ASR size, judge accuracy per LLM size) falls out of the always-small and always-large baseline runs, so it doesn't need its own experiment either.

| # | Experiment | Output |
|---|---|---|
| E1 | Calibration: full system on 16 GPUs, always-small / always-large / always-turbo, N = 1, 8, 32 | Per-stage latency-vs-load fits for all three tiers; WER vs. estimated SNR (sets the ASR threshold); answer/escalate/expert labels for the router |
| E2 | RAG on/off, full system, fixed N | How much RAG narrows the gap between tiers |
| E3 | Router + filler: base 8B with the prompted three-way router vs. the fine-tuned 8B; filler v1 vs. v2 | Per-tier routing accuracy; filler quality (judge: natural? any medical claim?) and its TTFA cost |
| E4 | Main result: closed loop on 16 GPUs, N = 1…64 × policies (always-small, always-large, always-turbo, adaptive, adaptive without filler) | The deck's latency-vs-quality figure with three tiers and real data; latency-vs-N, capacity, and how often the fallback chain fires |
| E5 | Parallelism layouts: small LLM 1 vs. 2 replicas; 70B TP=2 vs. TP=4; 405B TP=8 vs. TP=4 × PP=2 vs. TP=2 × PP=4; FSDP vs. DDP training throughput/memory | The class-technique results, each tied to a system-level effect |
| E6 | Scaling: adaptive on 8 GPUs (nodes 1–2, no turbo) vs. 16 GPUs | What the turbo tier buys in quality and costs in capacity |

**GPU budget:** 80+ GPU-hours left. Every hour of a 4-node job costs 16 GPU-hours, so one full E4 sweep at 3-minute windows (5 policies × 6 values of N) is about 32 GPU-hours on its own. Estimate: distillation 4, fine-tuning 8 (DDP and FSDP runs), development on 1–2 nodes 10, E1 calibration on 16 GPUs 8, E2–E3 6, E4 on 16 GPUs 24 (trimmed: N = 1, 8, 32, 64 and 2-minute windows), E5 8, E6 8: about 76, so it fits with little spare. Ask for more allocation if the first 4-node jobs need reruns. Never debug on 4 nodes: everything except the turbo tier is debugged on 1–2 nodes first, and the first 4-node job only checks that the turbo model comes up and answers.

## Two-week schedule

Compressed from three weeks to two because of exams and other research competing for time — which means milestones have to land in week 1, not get pushed into a final crunch. Work splits into four streams with a fixed interface, so any stream can be reassigned. The interface is a JSON log line per turn: caller id, timestamps per stage, model choices, transcript, answer, filler used.

- **Stream A, serving (Perlmutter):** vLLM servers, faster-whisper and TTS services, `sbatch` launch scripts, health checks.
- **Stream B, data and quality:** question subset, audio synthesis plus noise, RAG corpus and index, WER and judge scripts.
- **Stream C, Ellison (critical path):** orchestrator, streaming and sentence TTS, router and filler, scheduler, load generator, plots.
- **Stream D, Ellison (turbo tier):** the 405B on nodes 3–4 with Ray (TP=4 × PP=2), the 4-node launch, cross-node NCCL over Slingshot. Starts on day 1 alongside A, because it has the longest setup tail.
- **Training (whoever is free, Ellison by default):** the FSDP/DDP fine-tuning script and the distillation dataset.

**Days 1–4, foundations.** Done when one caller completes end to end on 2 nodes (small models + 70B, with RAG), the SNR estimator produces numbers, and the FSDP training script runs a few steps on the base 8B.
- A: small LLM on vLLM; Whisper small and large; TTS service; 70B 4-bit with TP=4 on Node 2; two-node `sbatch` script.
- B: MedQuAD splits; MedlinePlus chunks, embeddings and FAISS; synthesized and noise-mixed audio; WER script.
- C: orchestrator v0 (sequential pipeline), retrieval call, per-turn JSON log, blind SNR estimator, replica router.
- Training: FSDP + LoRA script with `torchrun`, DDP variant of the same script, smoke test on a tiny dataset.
- D: download the 405B to `$SCRATCH`; one 2-node job that brings up the Ray cluster and serves the 405B with TP=4 × PP=2; confirm cross-node NCCL is using Slingshot.

**Days 5–8, the system.** Done when a closed-loop N-caller run on 16 GPUs produces logs for all policies, including ASR routing by estimated SNR and all three LLM tiers, and the fine-tuned 8B is served by vLLM.
- B: judge script, validated on 50 hand-checked answers; E2.
- C: streaming plus sentence-level TTS; filler clip bank (v1); scheduler v1 (LLM tier + ASR tier); closed-loop load generator; E1 calibration.
- Training: generate the distillation set with the 70B (escalation labels + filler lines + medical-claim filter); fine-tune the 8B with FSDP and with DDP (the E5 training comparison comes from these same runs).
- D: first 4-node job (`run_16gpu.sbatch`) with a short sweep; E1 calibration on 16 GPUs.

**Days 9–11, experiments.**
- E3 (router + filler v1 vs. v2), E4 on 16 GPUs, E5 layout runs, E6 (8 vs. 16 GPUs), all as `sbatch` jobs; then offline judging and plots.

**Days 12–14, write-up and demo.**
- Report and slides.
- Demo: recordings of four calls (the same scenarios as `docs/pipeline-diagram.html`): an easy question answered by the 8B; a medium one where the 8B says a filler line and the 70B answers; an expert one answered by the 405B behind the filler; and a noisy one that goes to the large ASR via the estimated-SNR router.

**If teammates fall through:** Ellison takes A's minimum (vLLM for all three tiers, faster-whisper, TTS) plus C and D, with B reduced to about 300 MedQuAD questions, a single noise level plus clean audio, and judge-only quality. The three-tier, 16-GPU system stays in scope either way.

## The hard parts

Ranked roughly by how likely each is to eat days:

1. **Getting the models up on Perlmutter across nodes.** Placing the right number of ASR / TTS / small-LLM replicas on Node 1's GPUs without running out of memory, TP=4 for the 70B, and especially the 405B across two nodes, which needs a Ray cluster spanning both nodes inside one Slurm job. Plus the unglamorous parts: gated Llama weights downloaded to `$SCRATCH`, containers, ports, health checks.
2. **Making the nodes talk, and the data flow.** The orchestrator on Node 1 has to reach servers on the other nodes (hostnames from Slurm, HTTP or gRPC over Slingshot), stream LLM tokens into sentence-level TTS, and play filler while the 70B is still generating. That's async code with several things in flight per caller, and it's where most subtle bugs live.
3. **The scheduler.** Per-turn choices from a latency model fitted on the calibration run plus live in-flight counts. Conceptually simple; getting the predictions accurate enough under load is the hard bit.
4. **The fine-tune (router + filler).** Building the distillation dataset well (good escalation labels, filler that's natural but makes no medical claim) matters more than the training itself. The FSDP/DDP script is standard PyTorch and well documented.
5. **Audio analysis.** Simple: a blind SNR estimate from spectral features plus one threshold. Its error against the known noise level gets reported.
6. **RAG.** Simple: embed, FAISS search, put passages in the prompt. No training.
7. **Latency measurement.** Simple once every stage logs timestamps; the closed-loop generator is a small async script.

Two you didn't list:
- **Evaluation (the judge).** Running the 70B offline over thousands of answers, and validating it against hand-checked ones, is real work and real GPU-hours.
- **Perlmutter logistics.** Queue waits for 2- and 4-node jobs can take hours. Submit measurement runs early and in batches; debug on 1 node with the small models only.

## Risks
- **Two weeks plus exams means less slack than a three-week plan had.** Days 1–4 and 5–8 milestones are the forcing function. If either slips, trim the sweep (fewer N values, shorter windows) and cut E5 down to one comparison per technique before cutting any system component.
- **The fine-tuned 8B gets worse at answering, or its filler sneaks in medical claims.** It's a LoRA adapter, so the base weights are untouched and the prompted base 8B is always available as a fallback; filler lines are filtered in the training data and checked again by the judge in E3.
- **The 405B across two nodes is the riskiest piece of setup.** Multi-node Ray inside Slurm, AWQ with pipeline parallelism in vLLM, and NCCL over Slingshot can each eat a day; the model card's documented setup is TP=8, which is also our E5 comparison layout, so it's the fallback if TP=4 × PP=2 misbehaves. That's why Stream D starts on day 1, with its own 2-node bring-up job before any 4-node run. Weights are ~203 GiB, so download them to `$SCRATCH` early.
- **GPU-hours run out before the 16-GPU sweeps finish.** See the budget above: request more allocation early, and never debug on 4 nodes.
- **Multi-node and vLLM setup on Perlmutter takes longer than expected.** Get the 70B on one node running in week 1, as a side task.
- **The self-assessment router barely escalates, or escalates everything.** Tune the prompt on E1 labels, and fall back to a retrieval-score threshold.
- **The judge is unreliable on medical answers.** Use the reference answer in the judge prompt, plus a hand-checked sample. Report agreement.
- **The filler can feel deceptive.** Keep it non-substantive, and report TTSA alongside TTFA so filler can't hide slow answers.
- **TTS model choice.** If Breeze-TTS 2 is slow or awkward to serve, swap in a fast open TTS model, and keep the interface the same.

## Verification
- **Single call:** a scripted call yields an audio file plus a log line whose per-stage timestamps add up to the measured TTFA.
- **Quality checks:** WER on clean audio with large-v3 should be low. If it isn't, the ASR or audio path is broken.
- **Router check:** on about 30 hand-labeled questions, easy ones mostly get `ANSWER`, medium ones `ESCALATE` and expert ones `EXPERT`.
- **Load sanity:** at N = 1, end-to-end latency ≈ the sum of the logged per-stage latencies (nothing unaccounted for between stages). As N grows, always-turbo crosses the deadline first, then always-large, then adaptive.
- **Judge:** agreement with the 50 hand-checked answers is reported.
