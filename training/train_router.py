"""LoRA fine-tune of the small LLM on the distillation set (training/build_distill_set.py),
with DDP or FSDP. Same model, data, seed and hyperparameters for both, so the E5 comparison
isolates the parallelism strategy:

  DDP   every GPU holds the full 8B (~16 GB in bf16); gradients are all-reduced each step.
  FSDP  every GPU holds 1/world of each layer's weights; a layer is all-gathered just before it
        runs and freed after, and gradients are reduce-scattered.

  torchrun --standalone --nproc_per_node=4 training/train_router.py --strategy fsdp \
      --model meta-llama/Llama-3.1-8B-Instruct --data-dir $DATA_DIR/distill --out $SCRATCH/router-fsdp

Writes <out>/metrics.jsonl (loss, tokens/s, peak GPU memory per logged step), <out>/summary.json
and the LoRA adapter, which vLLM serves with --enable-lora.

Checkpoints: every --ckpt-every optimizer steps it saves <out>/checkpoint (LoRA weights, optimizer,
LR schedule, data position, RNG state; not the frozen 8B). Rerunning the same command with the
same --out resumes from it, with the same math as an uninterrupted run; if summary.json already
exists the run is complete and it exits."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from training.data import IGNORE, collate, load_examples

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def setup() -> tuple[int, int, int, torch.device]:
    cuda = torch.cuda.is_available()
    dist.init_process_group("nccl" if cuda else "gloo")
    rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0))
    if cuda:
        torch.cuda.set_device(local)
    return rank, world, local, torch.device(f"cuda:{local}" if cuda else "cpu")


def build_model(args, device: torch.device):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                      target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)

    if args.strategy == "fsdp":
        from peft.utils.other import fsdp_auto_wrap_policy
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

        # Each decoder layer is its own FSDP unit (all-gathered just in time); peft's policy also
        # gives the trainable LoRA weights their own units, since FSDP can't mix frozen and
        # trainable or bf16 and fp32 tensors in one flat parameter.
        return FSDP(model, auto_wrap_policy=fsdp_auto_wrap_policy(model), use_orig_params=True,
                    sharding_strategy=ShardingStrategy.FULL_SHARD, limit_all_gathers=True, device_id=device)
    model.to(device)
    return torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)


def peak_memory_gb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    t = torch.tensor([torch.cuda.max_memory_allocated(device) / 2**30], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


@torch.no_grad()
def val_loss(model, loader, device) -> float:
    model.eval()
    total = torch.zeros(2, device=device)
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        n = (batch["labels"] != IGNORE).sum()
        total += torch.stack([model(**batch).loss.float() * n, n.float()])
    dist.all_reduce(total)
    model.train()
    return float(total[0] / total[1]) if total[1] > 0 else float("nan")


def save_adapter(model, args, rank: int) -> None:
    if args.strategy == "fsdp":
        from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType

        from safetensors.torch import save_file

        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            state = model.state_dict()
        if rank == 0:
            # peft's save_pretrained finds adapter weights by module name, which FSDP's wrapping
            # changes, so it would save nothing. Write the same file format it writes for DDP.
            out = args.out / "adapter"
            out.mkdir(parents=True, exist_ok=True)
            lora = {k.replace(".default.", "."): v.contiguous() for k, v in state.items() if "lora_" in k}
            save_file(lora, str(out / "adapter_model.safetensors"))
            model.module.peft_config["default"].save_pretrained(str(out))
    elif rank == 0:
        model.module.save_pretrained(args.out / "adapter")


def _ckpt_options():
    from torch.distributed.checkpoint.state_dict import StateDictOptions

    # Only the LoRA weights train; the frozen base model is reloaded from the hub copy on resume,
    # so loading is non-strict about the (deliberately) missing frozen keys.
    return StateDictOptions(ignore_frozen_params=True, strict=False)


def save_checkpoint(model, optimizer, scheduler, out: Path, rank: int, device: torch.device, meta: dict) -> None:
    """Writes <out>/checkpoint.tmp, then swaps it in, so a job killed mid-save keeps the last good one."""
    import shutil

    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    tmp, final, old = out / "checkpoint.tmp", out / "checkpoint", out / "checkpoint.old"
    if rank == 0:
        shutil.rmtree(tmp, ignore_errors=True)
    dist.barrier()
    model_sd, optim_sd = get_state_dict(model, optimizer, options=_ckpt_options())
    dcp.save({"model": model_sd, "optim": optim_sd}, checkpoint_id=str(tmp))  # each rank writes its shard
    rng = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        rng["cuda"] = torch.cuda.get_rng_state(device)
    torch.save(rng, tmp / f"rng_{rank}.pt")
    dist.barrier()
    if rank == 0:
        torch.save(scheduler.state_dict(), tmp / "scheduler.pt")
        (tmp / "meta.json").write_text(json.dumps(meta))  # written last: marks the checkpoint complete
        shutil.rmtree(old, ignore_errors=True)
        if final.exists():
            final.rename(old)
        tmp.rename(final)
        shutil.rmtree(old, ignore_errors=True)
    dist.barrier()


def find_checkpoint(out: Path) -> Path | None:
    for name in ("checkpoint", "checkpoint.old"):  # .old only survives a crash during the swap
        if (out / name / "meta.json").exists():
            return out / name
    return None


def load_checkpoint(model, optimizer, scheduler, path: Path, rank: int, device: torch.device) -> dict:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    model_sd, optim_sd = get_state_dict(model, optimizer, options=_ckpt_options())
    state = {"model": model_sd, "optim": optim_sd}
    dcp.load(state, checkpoint_id=str(path))
    set_state_dict(model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optim"],
                   options=_ckpt_options())
    scheduler.load_state_dict(torch.load(path / "scheduler.pt"))
    meta = json.loads((path / "meta.json").read_text())
    rng_file = path / f"rng_{rank}.pt"
    meta["rng"] = torch.load(rng_file) if meta["world_size"] == dist.get_world_size() and rng_file.exists() else None
    return meta


def restore_rng(rng: dict | None, device: torch.device) -> None:
    if rng:
        torch.set_rng_state(rng["cpu"])
        if "cuda" in rng and device.type == "cuda":
            torch.cuda.set_rng_state(rng["cuda"], device)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy", required=True, choices=["ddp", "fsdp"])
    p.add_argument("--model", required=True)
    p.add_argument("--data-dir", required=True, type=Path, help="holds train.jsonl and val.jsonl")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps (0 = no cap)")
    p.add_argument("--batch-size", type=int, default=1, help="per GPU")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-every", type=int, default=50, help="checkpoint every this many optimizer steps (0 = never)")
    p.add_argument("--fail-after-step", type=int, default=0, help=argparse.SUPPRESS)  # tests: simulate a killed job
    args = p.parse_args()
    if (args.out / "summary.json").exists():
        print(f"{args.out}/summary.json exists: this run already finished", flush=True)
        return

    rank, world, _, device = setup()
    torch.manual_seed(args.seed)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    train, dropped = load_examples(str(args.data_dir / "train.jsonl"), tokenizer, args.max_len)
    val, _ = load_examples(str(args.data_dir / "val.jsonl"), tokenizer, args.max_len)
    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=True)
        print(f"{len(train)} train / {len(val)} val examples ({dropped} longer than {args.max_len} tokens dropped); "
              f"strategy={args.strategy} world={world}", flush=True)

    model = build_model(args, device)
    params = [q for q in model.parameters() if q.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sampler = DistributedSampler(train, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
    loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, collate_fn=partial(collate, pad_id=pad_id))
    val_loader = DataLoader(val, batch_size=args.batch_size,
                            sampler=DistributedSampler(val, num_replicas=world, rank=rank, shuffle=False),
                            collate_fn=partial(collate, pad_id=pad_id))
    total_steps = args.epochs * math.ceil(len(loader) / args.grad_accum)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = max(1, total_steps // 30)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, 1 - s / max(1, total_steps)))

    step, start_epoch, start_batch, rng = 0, 0, 0, None
    ckpt = find_checkpoint(args.out)
    if ckpt:
        meta = load_checkpoint(model, optimizer, scheduler, ckpt, rank, device)
        step, start_epoch, start_batch, rng = meta["step"], meta["epoch"], meta["next_batch"], meta["rng"]
        if rank == 0:
            print(f"resuming from {ckpt} at step {step}", flush=True)
    metrics_path = args.out / "metrics.jsonl"
    if rank == 0:
        # Drop rows logged after the checkpoint; the resumed run logs those steps again.
        kept = [r for r in metrics_path.read_text().splitlines() if json.loads(r)["step"] <= step] \
            if ckpt and metrics_path.exists() else []
        metrics = open(metrics_path, "w")
        metrics.write("".join(r + "\n" for r in kept))
    else:
        metrics = None
    window_tokens, window_loss, window_micro = 0, 0.0, 0
    t_window = time.perf_counter()
    model.train()
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)  # same shuffle as the interrupted run
        batches = iter(loader)  # draws a random seed, so the RNG is restored after this, not before
        if epoch == start_epoch:
            restore_rng(rng, device)
        for i, batch in enumerate(batches):
            if epoch == start_epoch and i < start_batch:
                continue
            batch = {k: v.to(device) for k, v in batch.items()}
            stepping = (i + 1) % args.grad_accum == 0 or i + 1 == len(loader)
            # DDP can skip the gradient all-reduce on accumulation steps. FSDP's no_sync would keep
            # unsharded gradients around, which defeats its memory savings, so it always syncs.
            sync = model.no_sync() if args.strategy == "ddp" and not stepping else contextlib.nullcontext()
            with sync:
                loss = model(**batch).loss
                (loss / args.grad_accum).backward()
            window_tokens += int(batch["attention_mask"].sum())
            window_loss += float(loss.detach())
            window_micro += 1
            if not stepping:
                continue
            if args.strategy == "fsdp":
                model.clip_grad_norm_(1.0)
            else:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0 or step == total_steps:
                stats = torch.tensor([window_tokens, window_loss, window_micro], dtype=torch.float64, device=device)
                dist.all_reduce(stats)
                elapsed = time.perf_counter() - t_window
                row = {"strategy": args.strategy, "world_size": world, "step": step,
                       "loss": float(stats[1] / stats[2]), "tokens_per_s": float(stats[0] / elapsed),
                       "peak_mem_gb": peak_memory_gb(device), "lr": scheduler.get_last_lr()[0]}
                if metrics:
                    metrics.write(json.dumps(row) + "\n")
                    metrics.flush()
                    print(json.dumps(row), flush=True)
                window_tokens, window_loss, window_micro = 0, 0.0, 0
                t_window = time.perf_counter()
            if step >= total_steps:
                break
            if args.ckpt_every and step % args.ckpt_every == 0:
                save_checkpoint(model, optimizer, scheduler, args.out, rank, device,
                                {"step": step, "epoch": epoch, "next_batch": i + 1, "world_size": world})
                if args.fail_after_step and step >= args.fail_after_step:
                    os._exit(1)
                t_window = time.perf_counter()  # don't count checkpoint time as training time
        if step >= total_steps:
            break

    vloss = val_loss(model, val_loader, device) if val else float("nan")
    save_adapter(model, args, rank)
    if rank == 0:
        metrics.close()
        rates = [json.loads(r)["tokens_per_s"] for r in metrics_path.read_text().splitlines()]  # includes pre-resume rows
        summary = {"strategy": args.strategy, "world_size": world, "steps": step, "train_examples": len(train),
                   "val_loss": vloss, "mean_tokens_per_s": sum(rates) / len(rates) if rates else None,
                   "peak_mem_gb": peak_memory_gb(device) if device.type == "cuda" else None}
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
    elif device.type == "cuda":
        peak_memory_gb(device)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
